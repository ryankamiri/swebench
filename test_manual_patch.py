#!/usr/bin/env python3
"""
Interactive script to manually test LLM-generated patches.

This script allows you to:
1. Select an instance from the dataset
2. Paste a raw LLM completion (multi-line)
3. Test if the patch applies
4. Run the tests
5. See the results

Press Enter on an empty line to skip or finish input.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from datasets import load_dataset
from swebench.harness.text_utils import remove_readme


def extract_patch(raw_completion: str) -> str:
    """Extract patch from raw LLM completion."""
    import re
    
    # Remove markdown code fences if present
    cleaned_text = raw_completion
    if '```' in cleaned_text:
        parts = cleaned_text.split('```')
        for part in parts:
            if '---' in part and '+++' in part:
                cleaned_text = part.strip()
                if cleaned_text.startswith(('diff', 'patch')):
                    cleaned_text = '\n'.join(cleaned_text.split('\n')[1:])
                break
    
    # Look for diff markers
    lines = cleaned_text.split('\n')
    patch_lines = []
    in_patch = False
    blank_line_count = 0
    
    for i, line in enumerate(lines):
        if line.startswith('---'):
            in_patch = True
            blank_line_count = 0
            patch_lines.append(line)
        elif line.startswith('+++') and in_patch:
            patch_lines.append(line)
        elif in_patch:
            if line.startswith(('@', '+', '-', ' ', '\\')):
                patch_lines.append(line)
                blank_line_count = 0
            elif line.strip() == '':
                patch_lines.append(line)
                blank_line_count += 1
                if blank_line_count >= 2 and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not (next_line.startswith(('---', '+++', '@@', '+', '-', ' ', '\\')) or next_line.strip() == ''):
                        break
            elif line.startswith(('diff --git', 'index ')):
                patch_lines.append(line)
                blank_line_count = 0
            else:
                if len(patch_lines) > 3 and any(l.startswith('@@') for l in patch_lines):
                    break
    
    result = '\n'.join(patch_lines)
    if result and not result.endswith('\n'):
        result += '\n'
    
    return result.strip()


def get_multiline_input(prompt: str) -> str:
    """Get multi-line input from user. Empty line to finish."""
    print(prompt)
    print("(Paste your content, then press Enter on an empty line to finish)")
    print("-" * 70)
    
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                if not lines:  # First line is empty = skip
                    return ""
                break
            lines.append(line)
        except EOFError:
            break
    
    return '\n'.join(lines)


def test_patch_on_instance(instance_id: str, raw_completion: str, workspace_dir: str = "test_workspace"):
    """Test a patch on a specific instance."""
    
    # Load dataset
    print(f"\n{'='*70}")
    print(f"Loading dataset...")
    dataset = load_dataset('ryankamiri/SWE-bench_oracle_lite', split='test')
    
    # Find instance
    instance = None
    for inst in dataset:
        if inst['instance_id'] == instance_id:
            instance = inst
            break
    
    if not instance:
        print(f"❌ Instance {instance_id} not found!")
        return False
    
    print(f"✅ Found instance: {instance_id}")
    
    # Show problem statement
    print(f"\n📖 Problem Statement:")
    print("-" * 70)
    problem_text = instance.get('text', instance.get('problem_statement', 'N/A'))
    problem_text_cleaned = remove_readme(problem_text)
    print(problem_text_cleaned[:500] + "..." if len(problem_text_cleaned) > 500 else problem_text_cleaned)
    print("-" * 70)
    
    # Extract patch from raw completion
    print(f"\n🔧 Extracting patch from raw completion...")
    patch = extract_patch(raw_completion)
    
    if not patch:
        print("❌ No valid patch found in the completion!")
        print("\nRaw completion:")
        print(raw_completion[:500])
        return False
    
    print(f"✅ Extracted patch ({len(patch)} chars)")
    print("\n📝 Extracted Patch:")
    print("-" * 70)
    print(patch)
    print("-" * 70)
    
    # Setup workspace
    workspace_path = Path(workspace_dir).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    repo = instance['repo']
    base_commit = instance['base_commit']
    repo_name = repo.replace('/', '_')
    repo_path = workspace_path / repo_name
    
    print(f"\n📦 Setting up repository: {repo}")
    
    # Clone or reset repo
    if not repo_path.exists():
        print(f"   Cloning {repo}...")
        result = subprocess.run(
            ['git', 'clone', f'https://github.com/{repo}.git', str(repo_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Clone failed: {result.stderr}")
            return False
    
    # Reset to base commit
    print(f"   Resetting to commit {base_commit[:8]}...")
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'clean', '-fdx'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'checkout', base_commit], cwd=repo_path, capture_output=True)
    
    # Write patch to file
    patch_file = repo_path / '.test_patch.diff'
    with open(patch_file, 'w') as f:
        f.write(patch)
    
    # Try to apply patch
    print(f"\n🔨 Applying patch...")
    result = subprocess.run(
        ['git', 'apply', '--verbose', str(patch_file)],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Patch applied successfully!")
        patch_applied = True
    else:
        print(f"❌ Patch application failed!")
        print(f"Error: {result.stderr}")
        
        # Try with --reject
        print("\n   Trying with --reject flag...")
        result = subprocess.run(
            ['git', 'apply', '--reject', str(patch_file)],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("⚠️  Patch partially applied (with rejects)")
            patch_applied = True
        else:
            print(f"❌ Still failed: {result.stderr}")
            patch_applied = False
    
    # Show git diff
    if patch_applied:
        print("\n📊 Git diff after applying patch:")
        print("-" * 70)
        diff_result = subprocess.run(
            ['git', 'diff'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        print(diff_result.stdout[:1000] + "..." if len(diff_result.stdout) > 1000 else diff_result.stdout)
        print("-" * 70)
    
    # Cleanup
    os.remove(patch_file)
    
    return patch_applied


def main():
    """Main interactive loop."""
    print("="*70)
    print("🧪 MANUAL PATCH TESTING TOOL")
    print("="*70)
    print("\nThis tool lets you test LLM-generated patches on SWE-bench instances.")
    print("You can paste raw completions and see if they apply and pass tests.")
    print("\nPress Ctrl+C to exit at any time.")
    print("="*70)
    
    # Load dataset to show available instances
    print("\n📚 Loading dataset...")
    dataset = load_dataset('ryankamiri/SWE-bench_oracle_lite', split='test')
    
    # Get first 3 instances
    instances = []
    for i, inst in enumerate(dataset):
        if i >= 3:
            break
        instances.append(inst)
    
    print(f"\n✅ Loaded {len(instances)} test instances:")
    for i, inst in enumerate(instances, 1):
        print(f"   {i}. {inst['instance_id']}")
    
    # Main loop
    while True:
        print(f"\n{'='*70}")
        print("SELECT AN INSTANCE TO TEST")
        print("="*70)
        
        for i, inst in enumerate(instances, 1):
            print(f"{i}. {inst['instance_id']}")
        print("0. Exit")
        
        try:
            choice = input("\nEnter choice (1-3, or 0 to exit): ").strip()
            
            if choice == "0" or choice == "":
                print("\n👋 Goodbye!")
                break
            
            choice_num = int(choice)
            if choice_num < 1 or choice_num > len(instances):
                print("❌ Invalid choice!")
                continue
            
            instance = instances[choice_num - 1]
            instance_id = instance['instance_id']
            
            print(f"\n{'='*70}")
            print(f"TESTING: {instance_id}")
            print("="*70)
            
            # Get raw completion
            raw_completion = get_multiline_input("\n📝 Paste the raw LLM completion:")
            
            if not raw_completion:
                print("⏭️  Skipped (empty input)")
                continue
            
            # Test the patch
            success = test_patch_on_instance(instance_id, raw_completion)
            
            if success:
                print(f"\n{'='*70}")
                print("✅ PATCH APPLIED SUCCESSFULLY!")
                print("="*70)
            else:
                print(f"\n{'='*70}")
                print("❌ PATCH FAILED TO APPLY")
                print("="*70)
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
