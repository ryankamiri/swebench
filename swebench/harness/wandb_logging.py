"""
Shared wandb logging utilities for SWE-bench inference and evaluation.

This module provides a centralized logging system for tracking metrics
across inference and evaluation phases.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import wandb


class SWEBenchWandbLogger:
    """Unified wandb logger for SWE-bench workflows."""
    
    def __init__(
        self,
        project: str = "swebench",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize wandb logger.
        
        Args:
            project: Wandb project name
            run_name: Run name (auto-generated if None)
            config: Configuration dictionary
            tags: List of tags for the run
        """
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.tags = tags or ["swebench"]
        
        # Initialize wandb run
        wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
            tags=self.tags,
        )
        
        # Track metrics
        self.metrics = {
            "start_time": time.time(),
        }
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to wandb."""
        wandb.log(metrics)
    
    def finish(self):
        """Finish the wandb run."""
        wandb.finish()


class InferenceLogger(SWEBenchWandbLogger):
    """Wandb logger for inference phase."""
    
    def __init__(
        self,
        project: str = "swebench-inference",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize inference logger."""
        super().__init__(
            project=project,
            run_name=run_name,
            config=config,
            tags=["swebench", "inference"],
        )
        
        self.metrics.update({
            "total_instances": 0,
            "completed_instances": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
        })
    
    def log_instance_inference(
        self,
        instance_id: str,
        patch_length: int,
        raw_completion_length: int,
        prompt_length: int,
        has_diff_markers: bool,
        patch_empty: bool,
        progress: float,
    ):
        """
        Log metrics for a single instance inference.
        
        Args:
            instance_id: Instance identifier
            patch_length: Length of generated patch
            raw_completion_length: Length of raw model output
            prompt_length: Length of input prompt
            has_diff_markers: Whether patch has valid diff syntax
            patch_empty: Whether patch is empty
            progress: Progress through dataset (0.0 to 1.0)
        """
        self.metrics["completed_instances"] += 1
        if not patch_empty:
            self.metrics["successful_predictions"] += 1
        else:
            self.metrics["failed_predictions"] += 1
        
        self.log_metrics({
            "inference/patch_length": patch_length,
            "inference/raw_completion_length": raw_completion_length,
            "inference/prompt_length": prompt_length,
            "inference/has_diff_markers": 1 if has_diff_markers else 0,
            "inference/patch_empty": 1 if patch_empty else 0,
            "inference/progress": progress,
            "inference/completed_instances": self.metrics["completed_instances"],
        })
    
    def log_final_inference(self, total_predictions: int, successful_predictions: int):
        """Log final inference statistics."""
        duration = time.time() - self.metrics["start_time"]
        
        self.log_metrics({
            "inference/total_predictions": total_predictions,
            "inference/successful_predictions": successful_predictions,
            "inference/failed_predictions": total_predictions - successful_predictions,
            "inference/success_rate": successful_predictions / total_predictions if total_predictions > 0 else 0,
            "inference/duration_seconds": duration,
            "inference/duration_minutes": duration / 60,
        })


class EvaluationLogger(SWEBenchWandbLogger):
    """Wandb logger for evaluation phase."""
    
    def __init__(
        self,
        project: str = "swebench-evaluation",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize evaluation logger."""
        super().__init__(
            project=project,
            run_name=run_name,
            config=config,
            tags=["swebench", "evaluation"],
        )
        
        self.metrics.update({
            "total_instances": 0,
            "completed_instances": 0,
            "passed_instances": 0,
            "failed_instances": 0,
            "error_instances": 0,
        })
    
    def log_instance_evaluation(self, report: Dict):
        """
        Log metrics for a single instance evaluation.
        
        Args:
            report: Evaluation report dictionary
        """
        instance_id = report.get("instance_id", "unknown")
        status = report.get("status", "UNKNOWN")
        
        # Update metrics
        self.metrics["completed_instances"] += 1
        if status == "PASSED":
            self.metrics["passed_instances"] += 1
        elif status == "FAILED":
            self.metrics["failed_instances"] += 1
        else:
            self.metrics["error_instances"] += 1
        
        # Extract detailed test information
        test_results = report.get("test_results", {})
        FAIL_TO_PASS = test_results.get("FAIL_TO_PASS", [])
        PASS_TO_PASS = test_results.get("PASS_TO_PASS", [])
        
        # Extract patch application info
        patch_applied = report.get("patch_applied", False)
        
        # Calculate test success rates
        f2p_count = len(FAIL_TO_PASS) if FAIL_TO_PASS else 0
        p2p_count = len(PASS_TO_PASS) if PASS_TO_PASS else 0
        total_tests = test_results.get("num_tests", 0)
        passed_tests = test_results.get("num_passed", 0)
        failed_tests = test_results.get("num_failed", 0)
        
        # Calculate F2P success (tests that should have passed after fix)
        f2p_success = passed_tests if f2p_count > 0 else 0
        f2p_rate = f2p_success / f2p_count if f2p_count > 0 else 0
        
        # Log metrics to wandb
        self.log_metrics({
            # Overall metrics
            "eval/completed_instances": self.metrics["completed_instances"],
            "eval/passed_instances": self.metrics["passed_instances"],
            "eval/failed_instances": self.metrics["failed_instances"],
            "eval/error_instances": self.metrics["error_instances"],
            "eval/pass_rate": self.metrics["passed_instances"] / max(1, self.metrics["completed_instances"]),
            
            # Per-instance metrics
            "eval/patch_applied": 1 if patch_applied else 0,
            "eval/patch_success_rate": self.metrics["passed_instances"] / max(1, self.metrics["completed_instances"]),
            
            # Test metrics
            "eval/f2p_test_count": f2p_count,
            "eval/p2p_test_count": p2p_count,
            "eval/f2p_success_count": f2p_success,
            "eval/f2p_success_rate": f2p_rate,
            "eval/total_tests": total_tests,
            "eval/tests_passed": passed_tests,
            "eval/tests_failed": failed_tests,
            "eval/test_pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
        })
    
    def log_final_evaluation(self, reports: List[Dict]):
        """
        Log final evaluation results.
        
        Args:
            reports: List of evaluation report dictionaries
        """
        # Calculate final metrics
        total_instances = len(reports)
        passed_instances = len([r for r in reports if r.get("status") == "PASSED"])
        failed_instances = len([r for r in reports if r.get("status") == "FAILED"])
        error_instances = len([r for r in reports if r.get("status") not in ["PASSED", "FAILED"]])
        
        # Patch application statistics
        patches_applied = len([r for r in reports if r.get("patch_applied", False)])
        patches_failed = len([r for r in reports if not r.get("patch_applied", False)])
        
        pass_rate = passed_instances / total_instances if total_instances > 0 else 0
        patch_apply_rate = patches_applied / total_instances if total_instances > 0 else 0
        duration = time.time() - self.metrics["start_time"]
        
        # Log final metrics
        final_metrics = {
            "final/total_instances": total_instances,
            "final/passed_instances": passed_instances,
            "final/failed_instances": failed_instances,
            "final/error_instances": error_instances,
            "final/pass_rate": pass_rate,
            "final/patches_applied": patches_applied,
            "final/patches_failed_to_apply": patches_failed,
            "final/patch_apply_rate": patch_apply_rate,
            "final/duration_seconds": duration,
            "final/duration_minutes": duration / 60,
        }
        
        self.log_metrics(final_metrics)
        
        # Create summary table
        self._create_summary_table(reports)
    
    def _create_summary_table(self, reports: List[Dict]):
        """Create a summary table in wandb."""
        table_data = []
        for report in reports:
            test_results = report.get("test_results", {})
            FAIL_TO_PASS = test_results.get("FAIL_TO_PASS", [])
            PASS_TO_PASS = test_results.get("PASS_TO_PASS", [])
            
            table_data.append([
                report.get("instance_id", "unknown"),
                report.get("status", "UNKNOWN"),
                "✓" if report.get("patch_applied", False) else "✗",
                len(FAIL_TO_PASS) if FAIL_TO_PASS else 0,
                len(PASS_TO_PASS) if PASS_TO_PASS else 0,
                test_results.get("num_tests", 0),
                test_results.get("num_failed", 0),
                test_results.get("num_passed", 0),
            ])
        
        table = wandb.Table(
            columns=[
                "Instance ID", 
                "Status", 
                "Patch Applied",
                "F2P Tests",
                "P2P Tests",
                "Total Tests", 
                "Failed Tests", 
                "Passed Tests"
            ],
            data=table_data
        )
        
        wandb.log({"evaluation_results": table})

