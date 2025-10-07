#!/usr/bin/env python3
"""
Complete pipeline to run Qwen2.5-Coder-7B on SWE-bench with wandb logging.

This script:
1. Loads the custom dataset
2. Runs inference with Qwen2.5-Coder-7B
3. Evaluates the predictions
4. Logs everything to wandb
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import wandb
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(
    dataset_name: str,
    model_name: str = "Qwen/Qwen2.5-Coder-7B",
    output_path: str = "predictions_qwen.jsonl",
    max_instances: Optional[int] = None,
    instance_ids: Optional[List[str]] = None,
    use_quantization: bool = False,
    wandb_project: str = "swebench-qwen",
    wandb_run_name: Optional[str] = None,
) -> str:
    """
    Run inference with Qwen2.5-Coder-7B on the dataset.
    
    Returns:
        Path to the predictions file
    """
    logger.info("Starting inference phase...")
    
    # Initialize wandb for inference
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or f"qwen-inference-{int(time.time())}",
        tags=["inference", "qwen"],
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "use_quantization": use_quantization,
            "max_instances": max_instances,
        }
    )
    
    try:
        # Run the inference script
        cmd = [
            "python", "-m", "swebench.inference.qwen_inference",
            "--dataset_name", dataset_name,
            "--output_path", output_path,
            "--model_name", model_name,
            "--wandb_project", wandb_project,
        ]
        
        if max_instances:
            cmd.extend(["--max_instances", str(max_instances)])
        
        if instance_ids:
            cmd.extend(["--instance_ids"] + instance_ids)
        
        if use_quantization:
            cmd.append("--use_quantization")
        
        logger.info(f"Running inference command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Inference failed: {result.stderr}")
            raise RuntimeError(f"Inference failed: {result.stderr}")
        
        logger.info("Inference completed successfully")
        logger.info(f"Predictions saved to: {output_path}")
        
        # Log inference results to wandb
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                predictions = [json.loads(line) for line in f]
            
            wandb.log({
                "inference/total_predictions": len(predictions),
                "inference/successful_predictions": len([p for p in predictions if p.get("model_patch")]),
            })
        
        return output_path
        
    finally:
        wandb.finish()


def run_evaluation(
    predictions_path: str,
    dataset_name: str,
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
    run_id: Optional[str] = None,
    instance_ids: Optional[List[str]] = None,
    wandb_project: str = "swebench-qwen",
    wandb_run_name: Optional[str] = None,
) -> str:
    """
    Run evaluation on the predictions.
    
    Returns:
        Path to the evaluation report
    """
    logger.info("Starting evaluation phase...")
    
    # Generate run ID if not provided
    if run_id is None:
        run_id = f"eval_{int(time.time())}"
    
    # Run the evaluation script using the simplified workspace approach
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation_workspace",
        "--predictions_path", predictions_path,
        "--dataset_name", dataset_name,
        "--workspace_dir", "swe_workspace",
        "--output_dir", "evaluation_results",
        "--run_id", run_id,
    ]
    
    logger.info(f"Running evaluation command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Evaluation failed: {result.stderr}")
        raise RuntimeError(f"Evaluation failed: {result.stderr}")
    
    logger.info("Evaluation completed successfully")
    
    # Find the report file
    report_path = Path("evaluation_results") / f"{run_id}_report.json"
    if report_path.exists():
        logger.info(f"Evaluation report saved to: {report_path}")
        return str(report_path)
    else:
        logger.warning("Evaluation report not found")
        return ""


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run complete Qwen2.5-Coder-7B evaluation on SWE-bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset and model arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ryankamiri/SWE-bench_oracle_lite",
        help="Name of the dataset to evaluate on",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B",
        help="Name of the model to use",
    )
    
    # Inference arguments
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="predictions_qwen.jsonl",
        help="Path to save predictions",
    )
    
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Maximum number of instances to process",
    )
    
    parser.add_argument(
        "--instance_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific instance IDs to process",
    )
    
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 4-bit quantization to reduce memory usage",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for evaluation",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for running tests (seconds)",
    )
    
    parser.add_argument(
        "--rm_image",
        action="store_true",
        help="Remove images after running",
    )
    
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild all images",
    )
    
    parser.add_argument(
        "--namespace",
        type=str,
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
        action="store_true",
        help="Clean cached images",
    )
    
    parser.add_argument(
        "--rewrite_reports",
        action="store_true",
        help="Rewrite existing reports",
    )
    
    # Wandb arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="swebench-qwen",
        help="Wandb project name",
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name",
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference phase (use existing predictions)",
    )
    
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation phase (only run inference)",
    )
    
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID for this evaluation",
    )

    args = parser.parse_args()
    
    # Generate run ID if not provided
    if args.run_id is None:
        args.run_id = f"qwen_run_{int(time.time())}"
    
    # Generate wandb run name if not provided
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.dataset_name.split('/')[-1]}_{args.run_id}"
    
    logger.info("Starting Qwen2.5-Coder-7B evaluation pipeline")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Wandb project: {args.wandb_project}")
    logger.info(f"Wandb run name: {args.wandb_run_name}")
    
    try:
        # Phase 1: Inference
        if not args.skip_inference:
            logger.info("=" * 50)
            logger.info("PHASE 1: INFERENCE")
            logger.info("=" * 50)
            
            predictions_path = run_inference(
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                output_path=args.predictions_path,
                max_instances=args.max_instances,
                instance_ids=args.instance_ids,
                use_quantization=args.use_quantization,
                wandb_project=args.wandb_project,
                wandb_run_name=f"{args.wandb_run_name}_inference",
            )
        else:
            logger.info("Skipping inference phase")
            predictions_path = args.predictions_path
        
        # Phase 2: Evaluation
        if not args.skip_evaluation:
            logger.info("=" * 50)
            logger.info("PHASE 2: EVALUATION")
            logger.info("=" * 50)
            
            report_path = run_evaluation(
                predictions_path=predictions_path,
                dataset_name=args.dataset_name,
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
                run_id=args.run_id,
                instance_ids=args.instance_ids,
                wandb_project=args.wandb_project,
                wandb_run_name=f"{args.wandb_run_name}_evaluation",
            )
            
            if report_path:
                logger.info(f"Final evaluation report: {report_path}")
        else:
            logger.info("Skipping evaluation phase")
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
