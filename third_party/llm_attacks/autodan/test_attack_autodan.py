#!/usr/bin/env python3
"""
Simple test for the modernized AutoDAN attack script.
"""

import subprocess
import sys
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality of attack_autodan.py"""
    
    print("Testing AutoDAN attack script...")
    
    # Test help message
    cmd = [sys.executable, "attack_autodan.py", "--help"]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Help message works")
            print("Available arguments:")
            help_lines = result.stdout.split('\n')
            for line in help_lines:
                if '--' in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ Help message failed")
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("✗ Help command timed out")
    except Exception as e:
        print(f"✗ Error running help: {e}")

def test_import_dependencies():
    """Test if all required dependencies can be imported"""
    
    print("\nTesting dependencies...")
    
    dependencies = [
        "torch", "numpy", "pandas", "transformers", 
        "vllm", "openai", "tqdm"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError as e:
            print(f"✗ {dep}: {e}")

def test_prompt_group_loading():
    """Test if prompt group file can be loaded"""
    
    print("\nTesting prompt group loading...")
    
    try:
        import torch
        prompt_file = Path(__file__).parent / "assets" / "prompt_group.pth"
        
        if prompt_file.exists():
            data = torch.load(prompt_file, map_location='cpu')
            print(f"✓ Loaded {len(data)} prompts from prompt_group.pth")
            print(f"  First prompt: {data[0][:100]}...")
        else:
            print(f"✗ Prompt group file not found: {prompt_file}")
            
    except Exception as e:
        print(f"✗ Error loading prompt group: {e}")

def test_dataset_loading():
    """Test dataset loading functionality"""
    
    print("\nTesting dataset loading...")
    
    try:
        # Test the dataset loading function
        test_code = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from attack_autodan import load_dataset_with_targets

try:
    # Test advbench
    advbench_data = load_dataset_with_targets("advbench")
    print(f"✓ AdvBench: {len(advbench_data)} samples")
    
    # Test jailbreakbench  
    jailbreak_data = load_dataset_with_targets("jailbreakbench")
    print(f"✓ JailbreakBench: {len(jailbreak_data)} samples")
    
except Exception as e:
    print(f"✗ Dataset loading error: {e}")
'''
        
        result = subprocess.run(
            [sys.executable, "-c", test_code], 
            cwd=Path(__file__).parent, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("✗ Dataset loading failed")
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"✗ Error testing dataset loading: {e}")

if __name__ == "__main__":
    print("=== AutoDAN Attack Script Test ===\n")
    
    test_basic_functionality()
    test_import_dependencies() 
    test_prompt_group_loading()
    test_dataset_loading()
    
    print("\n=== Test Complete ===")
    print("\nTo run a full attack test:")
    print("python attack_autodan.py --dataset advbench --target-model microsoft/DialoGPT-small --gpu 0 --num-steps 5 --num-samples 1")
