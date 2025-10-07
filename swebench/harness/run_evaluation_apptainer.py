"""
Apptainer-based evaluation runner to replace Docker-based run_evaluation.py.

This module provides Apptainer-specific implementations for running
SWE-bench evaluations using Apptainer containers instead of Docker.
"""

from __future__ import annotations

import json
import platform
import threading
import traceback

if platform.system() == "Linux":
    import resource

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path, PurePosixPath
from tqdm.auto import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.apptainer_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from swebench.harness.apptainer_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.modal_eval import (
    run_instances_modal,
    validate_modal_credentials,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
    optional_str,
)
from swebench.harness.apptainer_client import ApptainerClient

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def run_instance(
    test_spec: TestSpec,
    pred: dict,
    rm_image: bool,
    force_rebuild: bool,
    client: ApptainerClient,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
) -> dict:
    """
    Run a single instance with the given prediction using Apptainer.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (ApptainerClient): Apptainer client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        rewrite_reports (bool): True if eval run is just to reformat existing report
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    # Set up report file
    report_path = log_dir / LOG_REPORT
    if report_path.exists() and not rewrite_reports:
        # Load existing report if it exists and we're not rewriting
        with open(report_path, "r") as f:
            report = json.load(f)
        return report

    # Set up logger
    logger = setup_logger(instance_id, log_dir / LOG_INSTANCE)
    logger.info(f"Starting evaluation for {instance_id}")

    eval_completed = False
    report = {}
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(
            test_spec, client, run_id, logger, rm_image, force_rebuild
        )
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        # Attempt to apply patch to container
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f"{git_apply_cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
                applied_patch = True
                break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}")
        if not applied_patch:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}",
                logger,
            )

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run(
                "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
            )
            .output.decode(UTF8)
            .strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}, now copying to container..."
        )
        copy_to_container(container, eval_file, PurePosixPath("/tmp/eval.sh"))

        # Run the evaluation script
        logger.info(f"Running evaluation script for {instance_id}...")
        if timeout is not None:
            output, timed_out, duration = exec_run_with_timeout(
                container,
                f"bash /tmp/eval.sh",
                timeout=timeout,
            )
            if timed_out:
                logger.warning(f"Evaluation timed out after {timeout} seconds")
        else:
            val = container.exec_run(
                f"bash /tmp/eval.sh",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            output = val.output.decode(UTF8)
            duration = 0  # We don't track duration without timeout

        # Write test output to file
        test_output_file = log_dir / LOG_TEST_OUTPUT
        test_output_file.write_text(output)
        logger.info(f"Test output for {instance_id} written to {test_output_file}")

        # Get git diff after running eval script
        git_diff_output_after = (
            container.exec_run(
                "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
            )
            .output.decode(UTF8)
            .strip()
        )
        logger.info(f"Git diff after:\n{git_diff_output_after}")

        # Generate evaluation report
        report = get_eval_report(
            test_spec,
            pred,
            output,
            git_diff_output_before,
            git_diff_output_after,
            duration,
        )

        # Write report to file
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report for {instance_id} written to {report_path}")

        eval_completed = True

    except Exception as e:
        logger.error(f"Error during evaluation of {instance_id}: {e}")
        logger.error(traceback.format_exc())
        report = {
            "instance_id": instance_id,
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    finally:
        # Clean up container
        try:
            cleanup_container(client, container, logger)
        except Exception as e:
            logger.error(f"Error cleaning up container for {instance_id}: {e}")

        # Remove image if requested
        if rm_image and eval_completed:
            try:
                remove_image(client, test_spec.instance_image_key, logger)
            except Exception as e:
                logger.error(f"Error removing image for {instance_id}: {e}")

        close_logger(logger)

    return report


def run_instances(
    predictions: list[dict],
    dataset: list,
    max_workers: int = 4,
    timeout: int | None = None,
    rm_image: bool = False,
    force_rebuild: bool = False,
    namespace: str = None,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    cache_level: str = "env",
    clean: bool = False,
    rewrite_reports: bool = False,
) -> list[dict]:
    """
    Run evaluation on a list of predictions using Apptainer.

    Args:
        predictions (list[dict]): List of predictions to evaluate
        dataset (list): Dataset instances
        max_workers (int): Maximum number of parallel workers
        timeout (int): Timeout for running tests
        rm_image (bool): Whether to remove images after running
        force_rebuild (bool): Whether to force rebuild images
        namespace (str): Docker namespace for images
        instance_image_tag (str): Tag for instance images
        env_image_tag (str): Tag for environment images
        cache_level (str): Cache level for images
        clean (bool): Whether to clean cached images
        rewrite_reports (bool): Whether to rewrite existing reports

    Returns:
        list[dict]: List of evaluation reports
    """
    # Set open file limit for Linux
    if platform.system() == "Linux":
        open_file_limit = 65536
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    
    client = ApptainerClient()

    existing_images = list_images(client)

    # Build environment images
    print("Building environment images...")
    successful_env, failed_env = build_env_images(
        client,
        dataset,
        force_rebuild=force_rebuild,
        max_workers=max_workers,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )

    if len(failed_env) > 0:
        print(f"Warning: {len(failed_env)} environment images failed to build")

    # Filter predictions to only include those with successful environment images
    successful_env_keys = {spec.env_image_key for spec in successful_env}
    filtered_predictions = []
    for pred in predictions:
        instance_id = pred[KEY_INSTANCE_ID]
        # Find the test spec for this instance
        test_spec = None
        for spec in dataset:
            if spec.instance_id == instance_id:
                test_spec = spec
                break
        
        if test_spec and test_spec.env_image_key in successful_env_keys:
            filtered_predictions.append(pred)
        else:
            print(f"Skipping {instance_id} due to failed environment image")

    print(f"Running evaluation on {len(filtered_predictions)} instances...")

    # Prepare arguments for parallel execution
    args_list = []
    for pred in filtered_predictions:
        instance_id = pred[KEY_INSTANCE_ID]
        # Find the test spec for this instance
        test_spec = None
        for spec in dataset:
            if spec.instance_id == instance_id:
                test_spec = spec
                break
        
        if test_spec:
            test_spec = make_test_spec(
                test_spec,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
            )
            args_list.append(
                (
                    test_spec,
                    pred,
                    rm_image,
                    force_rebuild,
                    client,
                    f"run_{int(time.time())}",
                    timeout,
                    rewrite_reports,
                )
            )

    # Run evaluations in parallel
    successful, failed = run_threadpool(run_instance, args_list, max_workers)

    # Clean up images based on cache level
    if clean:
        clean_images(client, existing_images, cache_level, clean)

    print(f"Evaluation completed: {len(successful)} successful, {len(failed)} failed")
    return successful + failed


def main():
    """Main entry point for Apptainer-based evaluation."""
    parser = ArgumentParser(
        description="Run SWE-bench evaluation using Apptainer containers",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions file (JSONL format) or 'gold' for gold predictions",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Name of the dataset to evaluate on",
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for running tests (seconds)",
    )
    
    parser.add_argument(
        "--rm_image",
        type=str2bool,
        default=False,
        help="Remove images after running",
    )
    
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Force rebuild all images",
    )
    
    parser.add_argument(
        "--namespace",
        type=optional_str,
        default=None,
        help="Apptainer namespace for images",
    )
    
    parser.add_argument(
        "--instance_image_tag",
        type=str,
        default="latest",
        help="Tag for instance images",
    )
    
    parser.add_argument(
        "--env_image_tag",
        type=str,
        default="latest",
        help="Tag for environment images",
    )
    
    parser.add_argument(
        "--cache_level",
        type=str,
        default="env",
        choices=["none", "base", "env", "instance"],
        help="Cache level for images",
    )
    
    parser.add_argument(
        "--clean",
        type=str2bool,
        default=False,
        help="Clean cached images",
    )
    
    parser.add_argument(
        "--rewrite_reports",
        type=str2bool,
        default=False,
        help="Rewrite existing reports",
    )
    
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID for this evaluation",
    )
    
    parser.add_argument(
        "--instance_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific instance IDs to evaluate",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_swebench_dataset(args.dataset_name)
    
    # Filter dataset if specific instance IDs are provided
    if args.instance_ids:
        dataset = [inst for inst in dataset if inst.instance_id in args.instance_ids]
        print(f"Filtered to {len(dataset)} instances")

    # Load predictions
    if args.predictions_path == "gold":
        print("Using gold predictions")
        predictions = [
            {
                KEY_INSTANCE_ID: inst.instance_id,
                KEY_MODEL: "gold",
                KEY_PREDICTION: inst.patch,
            }
            for inst in dataset
        ]
    else:
        print(f"Loading predictions from: {args.predictions_path}")
        predictions = get_predictions_from_file(args.predictions_path)

    # Generate run ID if not provided
    if args.run_id is None:
        import time
        args.run_id = f"apptainer_run_{int(time.time())}"

    # Run evaluation
    print(f"Starting evaluation with run_id: {args.run_id}")
    reports = run_instances(
        predictions=predictions,
        dataset=dataset,
        max_workers=args.max_workers,
        timeout=args.timeout,
        rm_image=args.rm_image,
        force_rebuild=args.force_rebuild,
        namespace=args.namespace,
        instance_image_tag=args.instance_image_tag,
        env_image_tag=args.env_image_tag,
        cache_level=args.cache_level,
        clean=args.clean,
        rewrite_reports=args.rewrite_reports,
    )

    # Generate final report
    print("Generating final report...")
    final_report = make_run_report(
        reports=reports,
        full_dataset=dataset,
        run_id=args.run_id,
        client=None,  # We don't need the client for reporting
    )

    # Save final report
    report_path = Path("evaluation_results") / f"{args.run_id}_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"Evaluation completed. Report saved to: {report_path}")
    print(f"Total instances: {len(reports)}")
    print(f"Successful: {len([r for r in reports if r.get('status') == 'PASSED'])}")
    print(f"Failed: {len([r for r in reports if r.get('status') == 'FAILED'])}")


if __name__ == "__main__":
    main()
