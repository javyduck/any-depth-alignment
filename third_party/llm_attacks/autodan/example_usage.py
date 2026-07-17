#!/usr/bin/env python3
"""
Example usage of the modernized AutoDAN attack script.
"""

import subprocess
import sys
from pathlib import Path

def show_usage_examples():
    """Show various usage examples."""
    
    examples = [
        {
            "name": "Basic AdvBench attack on Llama-2",
            "cmd": [
                "python attack_autodan.py",
                "--dataset advbench",
                "--target-model meta-llama/Llama-2-7b-chat-hf",
                "--gpu 0",
                "--num-steps 50",
                "--num-samples 5"
            ]
        },
        {
            "name": "JailbreakBench attack with GPT-4o assistance",
            "cmd": [
                "python attack_autodan.py",
                "--dataset jailbreakbench", 
                "--target-model google/gemma-2-9b-it",
                "--API_key YOUR_OPENAI_API_KEY",
                "--gpu 1",
                "--num-steps 100"
            ]
        },
        {
            "name": "Quick test with small model",
            "cmd": [
                "python attack_autodan.py",
                "--dataset advbench",
                "--target-model microsoft/DialoGPT-small",
                "--gpu 0",
                "--num-steps 20",
                "--num-samples 3",
                "--max-tokens 32"
            ]
        }
    ]
    
    print("=== AutoDAN Attack Usage Examples ===\n")
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}:")
        print("   " + " \\\n     ".join(example['cmd']))
        print()
    
    print("Required Environment (Optional):")
    print("- OPENAI_API_KEY: Your OpenAI API key for GPT-4o assisted optimization")
    print()
    print("Output:")
    print("- Results saved to: ../attack_results/AutoDAN/{dataset}/{model_name}/attack_results.jsonl")
    print("- Each line contains a complete attack result with optimization log")
    print()
    print("Key Parameters:")
    print("- --num-steps: Number of optimization epochs (default: 100)")
    print("- --batch-size: Genetic algorithm batch size (default: 256)")
    print("- --API_key: OpenAI API key for enhanced optimization")
    print("- --start: Starting index in dataset")
    print("- --num-samples: Limit number of samples to process")

def run_example_attack():
    """Run an example AutoDAN attack with minimal parameters."""
    
    # Example command for a small test
    cmd = [
        sys.executable, "attack_autodan.py",
        "--dataset", "advbench",
        "--target-model", "microsoft/DialoGPT-small",  # Small model for testing
        "--gpu", "0",
        "--num-steps", "10",  # Reduced for testing
        "--num-samples", "2",  # Just 2 samples
        "--max-tokens", "32"
    ]
    
    print("Running example AutoDAN attack...")
    print("Command:", " ".join(cmd))
    print("\nNote: This requires:")
    print("1. GPU access for vLLM")
    print("2. Sufficient memory for the target model")
    print("3. AutoDAN utils (opt_utils.py, string_utils.py)")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Attack completed successfully!")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        else:
            print("✗ Attack failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            
    except Exception as e:
        print(f"Error running attack: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run_example_attack()
    else:
        show_usage_examples()
        print("\nTo run a test attack, use: python example_usage.py --run")
