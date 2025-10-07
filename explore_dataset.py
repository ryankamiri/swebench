#!/usr/bin/env python3
"""
Script to explore the custom SWE-bench dataset structure.
"""

import json
from datasets import load_dataset
from pathlib import Path

def explore_dataset():
    """Explore the custom SWE-bench dataset structure."""
    print("Loading dataset: ryankamiri/SWE-bench_oracle_lite")
    
    try:
        # Load the dataset
        dataset = load_dataset("ryankamiri/SWE-bench_oracle_lite")
        
        print(f"Dataset keys: {list(dataset.keys())}")
        
        # Explore each split
        for split_name, split_data in dataset.items():
            print(f"\n=== {split_name.upper()} SPLIT ===")
            print(f"Number of instances: {len(split_data)}")
            
            if len(split_data) > 0:
                # Show first instance structure
                first_instance = split_data[0]
                print(f"\nFirst instance keys: {list(first_instance.keys())}")
                
                # Show detailed structure of first instance
                print("\nFirst instance details:")
                for key, value in first_instance.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}... (truncated, length: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
                
                # Show a few more instances to understand patterns
                if len(split_data) > 1:
                    print(f"\nSample of instance IDs:")
                    for i in range(min(5, len(split_data))):
                        instance = split_data[i]
                        instance_id = instance.get('instance_id', f'instance_{i}')
                        print(f"  {i}: {instance_id}")
        
        # Save a sample to file for inspection
        sample_file = Path("dataset_sample.json")
        sample_data = {
            "dataset_info": {
                "name": "ryankamiri/SWE-bench_oracle_lite",
                "splits": list(dataset.keys()),
                "total_instances": sum(len(split) for split in dataset.values())
            },
            "sample_instances": []
        }
        
        # Add a few sample instances from each split
        for split_name, split_data in dataset.items():
            for i in range(min(2, len(split_data))):
                sample_data["sample_instances"].append({
                    "split": split_name,
                    "index": i,
                    "data": split_data[i]
                })
        
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2, default=str)
        
        print(f"\nSample data saved to: {sample_file}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def check_compatibility_with_swebench():
    """Check if the dataset is compatible with SWE-bench format."""
    print("\n=== COMPATIBILITY CHECK ===")
    
    dataset = load_dataset("ryankamiri/SWE-bench_oracle_lite")
    
    # Expected SWE-bench fields
    expected_fields = [
        'instance_id',
        'problem_statement', 
        'patch',
        'repo',
        'base_commit',
        'hints_text',
        'created_at',
        'test_patch',
        'test_file_path'
    ]
    
    for split_name, split_data in dataset.items():
        print(f"\nChecking {split_name} split:")
        if len(split_data) > 0:
            first_instance = split_data[0]
            available_fields = list(first_instance.keys())
            
            print(f"Available fields: {available_fields}")
            
            missing_fields = [field for field in expected_fields if field not in available_fields]
            extra_fields = [field for field in available_fields if field not in expected_fields]
            
            if missing_fields:
                print(f"Missing expected fields: {missing_fields}")
            else:
                print("✓ All expected fields present")
                
            if extra_fields:
                print(f"Extra fields: {extra_fields}")

if __name__ == "__main__":
    print("SWE-bench Dataset Explorer")
    print("=" * 50)
    
    dataset = explore_dataset()
    if dataset:
        check_compatibility_with_swebench()
    
    print("\nExploration complete!")
