#!/usr/bin/env python3
"""
Test script for GCG attack functionality.

This script tests the dataset loading and basic functionality without running the full attack.
"""

import sys
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

def test_dataset_loading():
    """Test loading datasets from CSV files."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is not installed. Please install it with: pip install pandas")
        return False
    
    from run_gcg_attack import load_dataset_with_targets
    
    print("=== Testing Dataset Loading ===")
    
    # Test AdvBench loading
    try:
        print("Loading AdvBench dataset...")
        advbench_data = load_dataset_with_targets("advbench")
        print(f"✓ Loaded {len(advbench_data)} AdvBench entries")
        
        # Show first example
        if advbench_data:
            prompt, target = advbench_data[0]
            print(f"  Example prompt: {prompt[:80]}...")
            print(f"  Example target: {target}")
        
    except Exception as e:
        print(f"✗ Error loading AdvBench: {e}")
        return False
    
    # Test JailbreakBench loading
    try:
        print("\nLoading JailbreakBench dataset...")
        jailbreak_data = load_dataset_with_targets("jailbreakbench")
        print(f"✓ Loaded {len(jailbreak_data)} JailbreakBench entries")
        
        # Show first example
        if jailbreak_data:
            prompt, target = jailbreak_data[0]
            print(f"  Example prompt: {prompt[:80]}...")
            print(f"  Example target: {target}")
        
    except Exception as e:
        print(f"✗ Error loading JailbreakBench: {e}")
        return False
    
    return True


def test_output_directory_creation():
    """Test output directory creation."""
    from run_gcg_attack import slugify_model_name
    
    print("\n=== Testing Output Directory Structure ===")
    
    # Test model name slugification
    test_models = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    for model in test_models:
        slug = slugify_model_name(model)
        print(f"Model: {model}")
        print(f"Slug: {slug}")
        
        # Test output path construction
        output_path = Path("../results") / "GCG" / "advbench" / slug
        print(f"Output path: {output_path}")
        print()


def main():
    """Run all tests."""
    print("GCG Attack Test Script")
    print("=" * 50)
    
    # Test dataset loading
    if not test_dataset_loading():
        print("Dataset loading tests failed!")
        return
    
    # Test output directory creation
    test_output_directory_creation()
    
    print("=== All Tests Passed! ===")
    print("\nTo run the actual GCG attack, use:")
    print("python run_gcg_attack.py --dataset advbench --model mistralai/Mistral-7B-Instruct-v0.2 --gpu 0")


if __name__ == "__main__":
    main()
