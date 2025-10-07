"""
Enhanced SWE-bench evaluation runner with wandb logging.

This module extends the standard evaluation runner to include comprehensive
wandb logging for tracking experiments and results.
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

if platform.system() == "Linux":
    import resource

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm.auto import tqdm

from swebench.harness.wandb_logging import EvaluationLogger

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_utils import (
    clean_images,
    list_images,
)
from swebench.harness.docker_build import (
    build_env_images,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
    optional_str,
)
from swebench.harness.run_evaluation import run_instance
import docker


# WandbLogger is now imported from wandb_logging module
# Keeping this as an alias for compatibility
WandbLogger = EvaluationLogger


def run_instances_with_wandb(
    predictions: List[Dict],
    dataset: List,
    max_workers: int = 4,
    timeout: Optional[int] = None,
    rm_image: bool = False,
    force_rebuild: bool = False,
    namespace: Optional[str] = None,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    cache_level: str = "env",
    clean: bool = False,
    rewrite_reports: bool = False,
    wandb_project: str = "swebench-evaluation",
    wandb_run_name: Optional[str] = None,
) -> List[Dict]:
    """
    Run evaluation on a list of predictions with wandb logging.
    
    Args:
        predictions: List of predictions to evaluate
        dataset: Dataset instances
        max_workers: Maximum number of parallel workers
        timeout: Timeout for running tests
        rm_image: Whether to remove images after running
        force_rebuild: Whether to force rebuild images
        namespace: Docker namespace for images
        instance_image_tag: Tag for instance images
        env_image_tag: Tag for environment images
        cache_level: Cache level for images
        clean: Whether to clean cached images
        rewrite_reports: Whether to rewrite existing reports
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        
    Returns:
        List of evaluation reports
    """
    # Set open file limit for Linux
    if platform.system() == "Linux":
        open_file_limit = 65536
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    
    client = docker.from_env()
    
    # Initialize wandb logger
    wandb_config = {
        "max_workers": max_workers,
        "timeout": timeout,
        "rm_image": rm_image,
        "force_rebuild": force_rebuild,
        "namespace": namespace,
        "instance_image_tag": instance_image_tag,
        "env_image_tag": env_image_tag,
        "cache_level": cache_level,
        "clean": clean,
        "rewrite_reports": rewrite_reports,
        "dataset_size": len(dataset),
        "predictions_size": len(predictions),
    }
    
    wandb_logger = WandbLogger(
        project=wandb_project,
        run_name=wandb_run_name,
        config=wandb_config,
    )
    
    try:
        existing_images = list_images(client)
        
        # Build environment images
        print("Building environment images...")
        wandb.log({"status": "building_images"})
        
        successful_env, failed_env = build_env_images(
            client,
            dataset,
            force_rebuild=force_rebuild,
            max_workers=max_workers,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
        )
        
        wandb.log({
            "successful_env_images": len(successful_env),
            "failed_env_images": len(failed_env),
        })
        
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
        
        wandb_logger.metrics["total_instances"] = len(filtered_predictions)
        wandb.log({"filtered_predictions": len(filtered_predictions)})
        
        print(f"Running evaluation on {len(filtered_predictions)} instances...")
        wandb.log({"status": "running_evaluation"})
        
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
        
        # Run evaluations in parallel with progress tracking
        reports = []
        with tqdm(total=len(args_list), desc="Evaluating instances") as pbar:
            def run_with_logging(*args):
                result = run_instance(*args)
                wandb_logger.log_instance_evaluation(result)
                pbar.update(1)
                return result
            
            # Use a custom wrapper to add logging
            successful, failed = run_threadpool(run_with_logging, args_list, max_workers)
            reports = successful + failed
        
        # Clean up images based on cache level
        if clean:
            clean_images(client, existing_images, cache_level, clean)
        
        # Log final results
        wandb_logger.log_final_evaluation(reports)
        
        print(f"Evaluation completed: {len(successful)} successful, {len(failed)} failed")
        
        return reports
        
    finally:
        wandb_logger.finish()


def main():
    """Main entry point for wandb-enabled evaluation."""
    parser = ArgumentParser(
        description="Run SWE-bench evaluation with wandb logging",
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
        default="ryankamiri/SWE-bench_oracle_lite",
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
        help="Docker namespace for images",
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
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="swebench-evaluation",
        help="Wandb project name",
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_swebench_dataset(args.dataset_name)
    
    # Filter dataset if specific instance IDs are provided
    if args.instance_ids:
        dataset = [inst for inst in dataset if (inst.get('instance_id') if isinstance(inst, dict) else inst.instance_id) in args.instance_ids]
        print(f"Filtered to {len(dataset)} instances")

    # Load predictions
    if args.predictions_path == "gold":
        print("Using gold predictions")
        predictions = [
            {
                KEY_INSTANCE_ID: inst.get('instance_id') if isinstance(inst, dict) else inst.instance_id,
                KEY_MODEL: "gold",
                KEY_PREDICTION: inst.get('patch') if isinstance(inst, dict) else inst.patch,
            }
            for inst in dataset
        ]
    else:
        print(f"Loading predictions from: {args.predictions_path}")
        # Determine split from dataset size (dev is smaller, test is larger)
        split = "test" if len(dataset) > 10 else "dev"
        predictions = get_predictions_from_file(args.predictions_path, args.dataset_name, split)

    # Generate run ID if not provided
    if args.run_id is None:
        args.run_id = f"wandb_run_{int(time.time())}"

    # Generate wandb run name if not provided
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.dataset_name.split('/')[-1]}_{args.run_id}"

    # Run evaluation with wandb logging
    print(f"Starting evaluation with run_id: {args.run_id}")
    print(f"Wandb project: {args.wandb_project}")
    print(f"Wandb run name: {args.wandb_run_name}")
    
    reports = run_instances_with_wandb(
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
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Generate final report
    print("Generating final report...")
    final_report = make_run_report(
        reports=reports,
        full_dataset=dataset,
        run_id=args.run_id,
        client=None,
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
