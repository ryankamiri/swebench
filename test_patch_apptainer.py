#!/usr/bin/env python3
"""
Test script to verify patch application and test execution using Apptainer.

This script tests a single SWE-bench instance by:
1. Setting up the repository environment
2. Applying a patch
3. Running tests
4. Reporting results

Usage:
    python test_patch_apptainer.py --instance_id <id> --patch <patch_file_or_string>
    python test_patch_apptainer.py --instance_id astropy__astropy-12907 --patch patch.txt
"""

import argparse
import sys
import os
from pathlib import Path
from datasets import load_dataset

# Add swebench to path
sys.path.insert(0, str(Path(__file__).parent))

from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.run_evaluation_apptainer import run_instance
from swebench.harness.apptainer_client import ApptainerClient
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)


def load_instance(instance_id: str, dataset_name: str = "princeton-nlp/SWE-bench_Lite"):
    """Load a single instance from the dataset."""
    print(f"📥 Loading instance {instance_id} from {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name, split="test")
        
        for item in dataset:
            if item[KEY_INSTANCE_ID] == instance_id:
                print(f"✅ Found instance: {item['repo']}")
                return item
        
        print(f"❌ Instance {instance_id} not found in dataset")
        return None
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def read_patch(patch_source: str) -> str:
    """Read patch from file or return as-is if it's a string."""
    if os.path.isfile(patch_source):
        print(f"📄 Reading patch from file: {patch_source}")
        with open(patch_source, 'r') as f:
            return f.read()
    else:
        print(f"📝 Using patch string ({len(patch_source)} chars)")
        return patch_source


def test_patch(
    instance_id: str,
    patch: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    workspace_dir: str = "./test_workspace",
    timeout: int = 600,
):
    """
    Test a patch on a single SWE-bench instance using Apptainer.
    
    Args:
        instance_id: SWE-bench instance ID (e.g., "astropy__astropy-12907")
        patch: Patch content (unified diff format)
        dataset_name: HuggingFace dataset name
        workspace_dir: Directory for test workspace
        timeout: Test execution timeout in seconds
        
    Returns:
        dict with test results
    """
    
    print("\n" + "=" * 80)
    print(f"TESTING PATCH FOR: {instance_id}")
    print("=" * 80 + "\n")
    
    # Load instance
    instance = load_instance(instance_id, dataset_name)
    if not instance:
        return {
            "success": False,
            "error": "Instance not found"
        }
    
    # Prepare prediction format expected by SWE-bench
    prediction = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: "test_model",
        KEY_PREDICTION: patch,
    }
    
    print(f"📊 Instance details:")
    print(f"   Repository: {instance['repo']}")
    print(f"   Base commit: {instance['base_commit']}")
    print(f"   Problem statement: {instance['problem_statement'][:100]}...")
    print()
    
    # Create test spec
    print("🔧 Creating test specification...")
    try:
        test_spec = make_test_spec(instance)
        print(f"✅ Test spec created")
        print(f"   Instance ID: {test_spec.instance_id}")
        print(f"   Repo: {test_spec.repo}")
        print(f"   Version: {test_spec.version}")
        print()
    except Exception as e:
        print(f"❌ Error creating test spec: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Test spec creation failed: {e}"
        }
    
    # Run evaluation with Apptainer
    print(f"🚀 Running evaluation with Apptainer...")
    print(f"   Workspace: {workspace_dir}")
    print(f"   Timeout: {timeout}s")
    print()
    
    try:
        # Create Apptainer client
        client = ApptainerClient(
            image_cache_dir=Path(workspace_dir) / "apptainer_images"
        )
        
        # Run the single instance
        result = run_instance(
            test_spec=test_spec,
            pred=prediction,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id="test_run",
            timeout=timeout,
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80 + "\n")
        
        resolved = result.get("resolved", False)
        print(f"✅ RESOLVED: {resolved}" if resolved else f"❌ NOT RESOLVED")
        print()
        
        # Test results
        if "test_results" in result:
            test_results = result["test_results"]
            print(f"📋 Test Results:")
            
            for test_name, test_result in test_results.items():
                status = "✅ PASS" if test_result == "PASSED" else "❌ FAIL"
                print(f"   {status} {test_name}")
            print()
        
        # Patch application
        patch_applied = result.get("patch_applied", False)
        print(f"📝 Patch Applied: {'✅ YES' if patch_applied else '❌ NO'}")
        
        if "error_message" in result and result["error_message"]:
            print(f"\n⚠️  Error: {result['error_message']}")
        
        print()
        
        return {
            "success": True,
            "resolved": resolved,
            "patch_applied": patch_applied,
            "test_results": result.get("test_results", {}),
            "error_message": result.get("error_message", ""),
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test a patch on a single SWE-bench instance"
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        required=True,
        help="SWE-bench instance ID (e.g., astropy__astropy-12907)"
    )
    parser.add_argument(
        "--patch",
        type=str,
        required=True,
        help="Path to patch file or patch string"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="./test_workspace",
        help="Workspace directory"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Test execution timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Read patch
    patch = read_patch(args.patch)
    
    if not patch or not patch.strip():
        print("❌ Error: Empty patch")
        sys.exit(1)
    
    # Test the patch
    result = test_patch(
        instance_id=args.instance_id,
        patch=patch,
        dataset_name=args.dataset,
        workspace_dir=args.workspace,
        timeout=args.timeout,
    )
    
    # Exit with appropriate code
    if result["success"] and result.get("resolved", False):
        print("\n🎉 SUCCESS: Patch resolves the issue!")
        sys.exit(0)
    elif result["success"] and result.get("patch_applied", False):
        print("\n⚠️  PARTIAL: Patch applied but tests failed")
        sys.exit(1)
    else:
        print("\n❌ FAILURE: Patch did not work")
        sys.exit(1)


if __name__ == "__main__":
    main()

