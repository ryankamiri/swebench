"""
Simplified workspace-based evaluation for Apptainer without complex image building.

This approach uses persistent git workspaces and simple Python containers,
similar to the working CoMLRL approach.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    SWEbenchInstance,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import (
    load_swebench_dataset,
    get_predictions_from_file,
)
from swebench.harness.wandb_logging import EvaluationLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkspaceEvaluator:
    """Simplified evaluator using persistent workspaces."""
    
    def __init__(self, workspace_dir: str = "swe_workspace"):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using workspace directory: {self.workspace_dir}")
        
        # Use a simple Python container
        self.base_image = self._get_python_image()
    
    def _get_python_image(self) -> str:
        """Get or pull a simple Python container."""
        print("🐍 Getting Python image...")
        
        cachedir = os.environ.get('APPTAINER_CACHEDIR')
        if not cachedir:
            cachedir = str((Path.cwd() / "logs/build_images/apptainer_images").resolve())
            os.makedirs(cachedir, exist_ok=True)
        
        print(f"   Using cache directory: {cachedir}")
        sif_path = Path(cachedir) / "python_3.9.sif"
        
        if not sif_path.exists():
            print("   Python 3.9 image not found, pulling from Docker Hub...")
            print("   This may take several minutes...")
            logger.info("Pulling Python 3.9 image...")
            
            env = os.environ.copy()
            env['APPTAINER_CACHEDIR'] = cachedir
            env['SINGULARITY_CACHEDIR'] = cachedir
            
            tmpdir = os.environ.get('APPTAINER_TMPDIR')
            if not tmpdir:
                tmpdir = str((Path.cwd() / "tmp/apptainer").resolve())
                os.makedirs(tmpdir, exist_ok=True)
            env['APPTAINER_TMPDIR'] = tmpdir
            env['SINGULARITY_TMPDIR'] = tmpdir
            
            print(f"   Pulling to: {sif_path}")
            print(f"   Command: apptainer pull {sif_path} docker://python:3.9")
            
            result = subprocess.run(
                ["apptainer", "pull", str(sif_path), "docker://python:3.9"],
                capture_output=True,
                text=True,
                env=env,
                timeout=1800
            )
            
            if result.returncode != 0:
                print(f"   ❌ Pull failed: {result.stderr}")
                raise RuntimeError(f"Failed to pull Python image: {result.stderr}")
            
            print(f"   ✅ Python image ready at: {sif_path}")
            logger.info(f"Python image ready at: {sif_path}")
        else:
            print(f"   ✅ Using existing Python image at: {sif_path}")
        
        return str(sif_path)
    
    def _get_repo_workspace(self, instance: SWEbenchInstance) -> Path:
        """Get or create a workspace for a repository."""
        repo_name = instance['repo'].replace('/', '_')
        repo_dir = self.workspace_dir / repo_name
        
        if not repo_dir.exists():
            print(f"📥 Cloning repository: {instance['repo']}")
            print(f"   Target directory: {repo_dir}")
            print(f"   This may take a few minutes...")
            logger.info(f"Cloning {instance['repo']}...")
            
            result = subprocess.run(
                ["git", "clone", f"https://github.com/{instance['repo']}.git", str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"   ❌ Clone failed: {result.stderr}")
                raise RuntimeError(f"Failed to clone {instance['repo']}: {result.stderr}")
            print(f"   ✅ Repository cloned successfully")
        else:
            print(f"♻️  Using existing repository: {repo_dir}")
        
        # Checkout the correct commit
        print(f"🔄 Resetting repository to clean state...")
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fdx"], cwd=repo_dir, capture_output=True)
        result = subprocess.run(
            ["git", "checkout", "-f", instance['base_commit']],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout {instance['base_commit']}: {result.stderr}")
        
        return repo_dir
    
    def _apply_patch(self, patch: str, repo_dir: Path) -> bool:
        """Apply a patch to the repository."""
        patch_file = repo_dir / ".swebench_patch.diff"
        
        try:
            with open(patch_file, 'w') as f:
                f.write(patch)
            
            result = subprocess.run(
                ["git", "apply", "--verbose", patch_file],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"\n   ⚠️  First git apply attempt failed")
                print(f"   Git apply error:")
                print(f"   {'-'*60}")
                print(f"   {result.stderr}")
                print(f"   {'-'*60}")
                
                # Show what file git was trying to patch
                print(f"\n   🔍 Patch Analysis:")
                patch_lines = patch.split('\n')
                for line in patch_lines[:10]:
                    if line.startswith('---') or line.startswith('+++'):
                        filename = line.split(' ')[1] if len(line.split()) > 1 else 'unknown'
                        if filename.startswith('a/') or filename.startswith('b/'):
                            filename = filename[2:]
                        print(f"   Target file: {filename}")
                        
                        # Show actual file content around the patch location
                        file_path = repo_dir / filename
                        if file_path.exists():
                            print(f"   File exists: ✅")
                            with open(file_path, 'r') as f:
                                file_lines = f.readlines()
                            print(f"   File has {len(file_lines)} lines")
                            
                            # Try to find the line mentioned in patch
                            for patch_line in patch_lines:
                                if patch_line.startswith('@@'):
                                    print(f"   Patch hunk: {patch_line}")
                                    break
                        else:
                            print(f"   File exists: ❌")
                        break
                
                print(f"   Trying with --reject flag...\n")
                
                # Try with --reject
                result = subprocess.run(
                    ["git", "apply", "--verbose", "--reject", patch_file],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    print(f"\n   ❌ Git apply failed (both attempts)")
                    print(f"   Error output:")
                    print(f"   {'-'*60}")
                    print(f"   {result.stderr}")
                    print(f"   {'-'*60}")
                    
                    # Extract error line number from git error message
                    import re
                    error_lines = []
                    for match in re.finditer(r'line (\d+)', result.stderr):
                        error_lines.append(int(match.group(1)))
                    
                    # Read the actual patch file that git tried to apply
                    print(f"\n   📄 Patch file: {patch_file}")
                    print(f"   Reading from disk to see exactly what git saw...")
                    print(f"   {'-'*60}")
                    
                    try:
                        # Read as binary first to detect encoding issues
                        with open(patch_file, 'rb') as f:
                            patch_bytes = f.read()
                        
                        print(f"   File size: {len(patch_bytes)} bytes")
                        
                        # Try to decode as UTF-8
                        try:
                            patch_content = patch_bytes.decode('utf-8')
                            print(f"   Encoding: UTF-8 ✅")
                        except UnicodeDecodeError as e:
                            print(f"   Encoding: UTF-8 ❌ (error: {e})")
                            patch_content = patch_bytes.decode('utf-8', errors='replace')
                        
                        # Split into lines
                        patch_lines = patch_content.split('\n')
                        print(f"   Total lines in patch file: {len(patch_lines)}")
                        
                        # Check for common corruption issues
                        has_diff_header = any(line.startswith('---') for line in patch_lines)
                        has_plus_header = any(line.startswith('+++') for line in patch_lines)
                        has_hunk_header = any(line.startswith('@@') for line in patch_lines)
                        
                        print(f"\n   Patch validation:")
                        print(f"     - Has '---' header: {'✅' if has_diff_header else '❌'}")
                        print(f"     - Has '+++' header: {'✅' if has_plus_header else '❌'}")
                        print(f"     - Has '@@' hunk: {'✅' if has_hunk_header else '❌'}")
                        
                        # Check for weird characters or encoding issues
                        non_printable = []
                        for i, line in enumerate(patch_lines, 1):
                            for j, char in enumerate(line):
                                if ord(char) < 32 and char not in '\t\n\r':
                                    non_printable.append((i, j, char, ord(char)))
                        
                        if non_printable:
                            print(f"     - Non-printable chars: ⚠️ Found {len(non_printable)}")
                            for line_no, col, char, code in non_printable[:5]:
                                print(f"         Line {line_no}, col {col}: char code {code} ({repr(char)})")
                        else:
                            print(f"     - Non-printable chars: ✅ None")
                        
                        # Check line endings
                        has_crlf = b'\r\n' in patch_bytes
                        has_lf = b'\n' in patch_bytes and not has_crlf
                        print(f"     - Line endings: {'CRLF (Windows)' if has_crlf else 'LF (Unix)' if has_lf else 'Unknown'}")
                        
                        print(f"\n   {'-'*60}")
                        print(f"   Patch content with line numbers:")
                        print(f"   {'-'*60}")
                        
                        # Show context around error lines
                        for i, line in enumerate(patch_lines, 1):
                            # Highlight problematic lines mentioned in error
                            if i in error_lines:
                                print(f"   >>> {i:3d}: {repr(line)} <<<  ⚠️ ERROR AT THIS LINE")
                                # Show hex dump for problematic line to see hidden chars
                                hex_dump = ' '.join(f'{ord(c):02x}' for c in line[:50])
                                print(f"        Hex: {hex_dump}")
                                print(f"        Length: {len(line)} chars")
                                if line:
                                    print(f"        First char: {repr(line[0])} (code: {ord(line[0])})")
                                    print(f"        Last char: {repr(line[-1])} (code: {ord(line[-1])})")
                            elif error_lines and any(abs(i - err_line) <= 3 for err_line in error_lines):
                                # Show context around error (3 lines before/after)
                                print(f"       {i:3d}: {repr(line)}")
                            elif not error_lines:
                                # If no specific line found, show all (up to limit)
                                print(f"       {i:3d}: {repr(line)}")
                            
                            # Only show first 50 lines to avoid spam (unless error is later)
                            if not error_lines and i > 50:
                                print(f"       ... ({len(patch_lines) - 50} more lines)")
                                break
                            elif error_lines and i > max(error_lines) + 10:
                                remaining = len(patch_lines) - i
                                if remaining > 0:
                                    print(f"       ... ({remaining} more lines)")
                                break
                        print(f"   {'-'*60}\n")
                        
                    except Exception as read_error:
                        print(f"   ❌ Error reading patch file: {read_error}\n")
            
            os.remove(patch_file)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            if patch_file.exists():
                os.remove(patch_file)
            return False
    
    def _run_tests(self, test_spec: TestSpec, repo_dir: Path) -> tuple[str, str]:
        """Run tests in the workspace."""
        # Use the eval_script from test_spec which contains the test commands
        test_script = f"""#!/bin/bash
set -euo pipefail
cd {repo_dir}
echo "===START_TEST_OUTPUT==="
{test_spec.eval_script}
echo "===END_TEST_OUTPUT==="
"""
        
        script_path = repo_dir / ".swebench_test.sh"
        with open(script_path, 'w') as f:
            f.write(test_script)
        os.chmod(script_path, 0o755)
        
        # Run with Apptainer
        result = subprocess.run(
            [
                "apptainer", "exec",
                "--bind", f"{repo_dir}:{repo_dir}",
                "--pwd", str(repo_dir),
                "--cleanenv",
                self.base_image,
                "bash", str(script_path)
            ],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        os.remove(script_path)
        return result.stdout, result.stderr
    
    def evaluate_instance(self, instance: SWEbenchInstance, test_spec: TestSpec, prediction: Dict) -> Dict:
        """Evaluate a single instance."""
        instance_id = instance[KEY_INSTANCE_ID]
        patch = prediction.get(KEY_PREDICTION, "")
        
        print(f"\n{'='*60}")
        print(f"📋 Evaluating instance: {instance_id}")
        print(f"{'='*60}")
        logger.info(f"Evaluating {instance_id}...")
        
        # Print problem statement and gold patch for reference
        print(f"\n📖 Problem Statement:")
        print(f"   {'-'*60}")
        problem_text = instance.get('text', instance.get('problem_statement', 'N/A'))
        print(f"   {problem_text}")
        print(f"   {'-'*60}\n")
        
        print(f"🎯 Gold Patch (for reference):")
        print(f"   {'-'*60}")
        gold_patch = instance.get('patch', 'N/A')
        print(f"   {gold_patch}")
        print(f"   {'-'*60}\n")
        
        try:
            # Get workspace
            print(f"1️⃣  Setting up repository workspace...")
            repo_dir = self._get_repo_workspace(instance)
            print(f"   ✅ Workspace ready: {repo_dir}")
            
            # Print raw completion/patch
            print(f"2️⃣  Applying patch...")
            print(f"   Patch length: {len(patch)} characters")
            if prediction.get('metadata'):
                print(f"\n   📝 Raw LLM Completion:")
                print(f"   {'-'*60}")
                raw_completion = prediction['metadata'].get('raw_completion', '')
                print(f"   {raw_completion}...")
                print(f"   {'-'*60}")
                print(f"   Full length: {len(raw_completion)} characters\n")
            
            print(f"   📄 LLM Generated Patch:")
            print(f"   {'-'*60}")
            print(f"   {patch}...")
            print(f"   {'-'*60}\n")
            
            patch_applied = self._apply_patch(patch, repo_dir)
            if not patch_applied:
                print(f"   ❌ Patch application failed")
                logger.warning(f"Failed to apply patch for {instance_id}")
                return {
                    "instance_id": instance_id,
                    "patch_applied": False,
                    "resolved": False,
                    "error": "Patch application failed"
                }
            print(f"   ✅ Patch applied successfully")
            
            # Run tests
            print(f"3️⃣  Running tests in Apptainer container...")
            print(f"   This may take a few minutes...")
            stdout, stderr = self._run_tests(test_spec, repo_dir)
            
            # Reset workspace
            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, capture_output=True)
            subprocess.run(["git", "clean", "-fdx"], cwd=repo_dir, capture_output=True)
            
            return {
                "instance_id": instance_id,
                "patch_applied": True,
                "test_output": stdout + "\n" + stderr,
                "resolved": "PASSED" in stdout or "passed" in stdout,
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {instance_id}: {e}")
            return {
                "instance_id": instance_id,
                "patch_applied": False,
                "resolved": False,
                "error": str(e)
            }


def run_evaluation_workspace(
    predictions_path: str,
    dataset_name: str = None,
    dataset: list = None,
    workspace_dir: str = "swe_workspace",
    output_dir: str = "evaluation_results",
    run_id: str = None,
    wandb_project: str = None,
    wandb_run_name: str = None,
) -> List[Dict]:
    """
    Run evaluation using workspace approach.
    
    Args:
        predictions_path: Path to predictions file
        dataset_name: Name of dataset to load
        dataset: Pre-loaded dataset (if None, will load from dataset_name)
        workspace_dir: Directory for persistent workspaces
        output_dir: Directory for output results
        run_id: Run ID for this evaluation
    
    Returns:
        List of evaluation results
    """
    # Load predictions
    logger.info(f"Loading predictions from {predictions_path}")
    predictions = get_predictions_from_file(predictions_path, dataset_name or "unknown", "test")
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Load dataset if not provided
    if dataset is None:
        logger.info(f"Loading dataset {dataset_name}")
        dataset = load_swebench_dataset(dataset_name, split="test")
        logger.info(f"Loaded {len(dataset)} instances from dataset")
    
    # Create test specs and instance mapping
    test_specs = {}
    instances_map = {}
    for instance in dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        test_specs[instance_id] = make_test_spec(instance)
        instances_map[instance_id] = instance
    
    # Initialize wandb if requested
    wandb_logger = None
    if wandb_project:
        print(f"\n📊 Initializing wandb logging...")
        print(f"   Project: {wandb_project}")
        print(f"   Run name: {wandb_run_name or 'auto-generated'}\n")
        
        wandb_logger = EvaluationLogger(
            project=wandb_project,
            run_name=wandb_run_name,
            config={
                "dataset_name": dataset_name,
                "workspace_dir": workspace_dir,
                "run_id": run_id,
                "num_predictions": len(predictions),
            }
        )
    
    # Initialize evaluator
    print(f"\n🔧 Initializing workspace evaluator...")
    evaluator = WorkspaceEvaluator(workspace_dir=workspace_dir)
    print(f"✅ Evaluator ready\n")
    
    # Evaluate each prediction
    results = []
    for i, pred in enumerate(predictions, 1):
        instance_id = pred[KEY_INSTANCE_ID]
        print(f"\n{'*'*70}")
        print(f"Processing [{i}/{len(predictions)}]: {instance_id}")
        print(f"{'*'*70}")
        logger.info(f"[{i}/{len(predictions)}] Processing {instance_id}")
        
        if instance_id not in test_specs:
            logger.warning(f"No test spec found for {instance_id}")
            continue
        
        instance = instances_map[instance_id]
        test_spec = test_specs[instance_id]
        result = evaluator.evaluate_instance(instance, test_spec, pred)
        results.append(result)
        
        # Log to wandb
        if wandb_logger:
            wandb_logger.log_instance_evaluation(result)
        
        # Save intermediate results
        if run_id:
            output_path = Path(output_dir) / f"{run_id}_results.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    # Generate summary
    total = len(results)
    resolved = sum(1 for r in results if r.get('resolved', False))
    patch_applied_count = sum(1 for r in results if r.get('patch_applied', False))
    patch_failed_count = total - patch_applied_count
    
    print(f"\n{'='*70}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total completions generated:  {total}")
    print(f"Patches applied successfully: {patch_applied_count}/{total} ({patch_applied_count/total*100:.1f}%)")
    print(f"Patches failed to apply:      {patch_failed_count}/{total} ({patch_failed_count/total*100:.1f}%)")
    print(f"Instances resolved (passed):  {resolved}/{total} ({resolved/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total instances: {total}")
    logger.info(f"Patches applied: {patch_applied_count}/{total}")
    logger.info(f"Resolved: {resolved}/{total}")
    logger.info(f"Resolution rate: {resolved/total*100:.1f}%")
    
    # Log final results to wandb
    if wandb_logger:
        wandb_logger.log_final_evaluation(results)
        wandb_logger.finish()
    
    return results


def main():
    """Main entry point."""
    parser = ArgumentParser(
        description="Run SWE-bench evaluation with workspace approach",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions file (JSONL format)",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Name of the dataset",
    )
    
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="swe_workspace",
        help="Directory for persistent workspaces",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory for output results",
    )
    
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID for this evaluation",
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project name (optional)",
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (optional)",
    )
    
    args = parser.parse_args()
    
    if args.run_id is None:
        args.run_id = f"workspace_eval_{int(time.time())}"
    
    print("\n" + "="*70)
    print("🚀 SWE-bench Workspace Evaluation (Apptainer)")
    print("="*70)
    print(f"📁 Predictions: {args.predictions_path}")
    print(f"📊 Dataset: {args.dataset_name}")
    print(f"💾 Workspace: {args.workspace_dir}")
    print(f"🆔 Run ID: {args.run_id}")
    print("="*70 + "\n")
    
    logger.info("Starting workspace-based evaluation")
    logger.info(f"Predictions: {args.predictions_path}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Workspace: {args.workspace_dir}")
    logger.info(f"Run ID: {args.run_id}")
    
    results = run_evaluation_workspace(
        predictions_path=args.predictions_path,
        dataset_name=args.dataset_name,
        workspace_dir=args.workspace_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}/{args.run_id}_results.jsonl")


if __name__ == "__main__":
    main()

