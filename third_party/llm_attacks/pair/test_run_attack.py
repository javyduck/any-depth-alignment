#!/usr/bin/env python3
"""
Test script for run_attack.py to verify basic functionality.
"""

import sys
from pathlib import Path

# Add PAIR modules to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly."""
    try:
        from run_attack import (
            slugify_model_name,
            load_dataset_with_targets,
            VLLMTargetModel,
            GPTAttackModel,
            GPT4oJudge
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading functionality."""
    try:
        from run_attack import load_dataset_with_targets
        
        # Test advbench
        advbench_data = load_dataset_with_targets("advbench")
        print(f"✓ Loaded {len(advbench_data)} advbench samples")
        
        # Test jailbreakbench
        jbb_data = load_dataset_with_targets("jailbreakbench")
        print(f"✓ Loaded {len(jbb_data)} jailbreakbench samples")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False

def test_slugify():
    """Test model name slugification."""
    try:
        from run_attack import slugify_model_name
        
        test_cases = [
            ("mistralai/Mistral-7B-Instruct-v0.2", "mistralai_Mistral-7B-Instruct-v0_2"),
            ("meta-llama/Llama-2-7b-chat-hf", "meta-llama_Llama-2-7b-chat-hf"),
            ("google/gemma-2-9b-it", "google_gemma-2-9b-it")
        ]
        
        for input_name, expected in test_cases:
            result = slugify_model_name(input_name)
            if result == expected:
                print(f"✓ Slugify test passed: {input_name} -> {result}")
            else:
                print(f"✗ Slugify test failed: {input_name} -> {result} (expected {expected})")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Slugify error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running tests for run_attack.py...")
    
    tests = [
        ("Import test", test_imports),
        ("Dataset loading test", test_dataset_loading),
        ("Slugify test", test_slugify)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"Test failed: {test_name}")
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
