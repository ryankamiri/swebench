#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import wandb

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        import wandb
        print(f"✓ Wandb {wandb.__version__}")
    except ImportError as e:
        print(f"✗ Wandb import failed: {e}")
        return False
    
    return True

def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        return True
    else:
        print("✗ CUDA not available")
        return False

def test_model_loading():
    """Test loading the Qwen2.5-Coder-7B model."""
    print("\nTesting model loading...")
    
    try:
        model_name = "Qwen/Qwen2.5-Coder-7B"
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        print(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Model loaded successfully")
        
        # Test a simple generation
        print("Testing model generation...")
        test_prompt = "def fibonacci(n):\n    "
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation test passed: {generated_text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_dataset_loading():
    """Test loading the custom dataset."""
    print("\nTesting dataset loading...")
    
    try:
        dataset_name = "ryankamiri/SWE-bench_oracle_lite"
        print(f"Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name)
        print(f"✓ Dataset loaded successfully")
        print(f"  Splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} instances")
        
        # Test accessing first instance
        if len(dataset) > 0:
            first_split = list(dataset.keys())[0]
            first_instance = dataset[first_split][0]
            print(f"  First instance keys: {list(first_instance.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_wandb():
    """Test wandb initialization."""
    print("\nTesting wandb...")
    
    try:
        # Test wandb login (will prompt for API key if not logged in)
        wandb.login()
        print("✓ Wandb login successful")
        return True
        
    except Exception as e:
        print(f"✗ Wandb test failed: {e}")
        print("Note: You may need to run 'wandb login' manually")
        return False

def main():
    """Run all tests."""
    print("SWE-bench Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_gpu,
        test_model_loading,
        test_dataset_loading,
        test_wandb,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Setup is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
