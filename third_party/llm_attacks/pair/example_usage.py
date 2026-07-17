#!/usr/bin/env python3
"""
Example usage of the updated PAIR attack script.
"""

import subprocess
import sys
from pathlib import Path

def run_example_attack():
    """Run an example PAIR attack with minimal parameters."""
    
    # Example command for a small test
    cmd = [
        sys.executable, "run_attack.py",
        "--dataset", "advbench",
        "--target-model", "microsoft/DialoGPT-small",  # Small model for testing
        "--attack-model", "gpt-4o",
        "--judge-model", "gpt-4o", 
        "--n-streams", "3",  # Reduced for testing
        "--n-iterations", "2",  # Reduced for testing
        "--max-tokens", "100",
        "--gpu", "0"  # Use GPU 0
    ]
    
    print("Running example PAIR attack...")
    print("Command:", " ".join(cmd))
    print("\nNote: This requires:")
    print("1. OPENAI_API_KEY environment variable set")
    print("2. GPU access for vLLM")
    print("3. Sufficient memory for the target model")
    
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

def show_usage_examples():
    """Show various usage examples."""
    
    examples = [
        {
            "name": "Basic AdvBench attack on Mistral",
            "cmd": [
                "python run_attack.py",
                "--dataset advbench",
                "--target-model mistralai/Mistral-7B-Instruct-v0.2",
                "--n-streams 20",
                "--n-iterations 5",
                "--gpu 0"
            ]
        },
        {
            "name": "JailbreakBench attack on Llama",
            "cmd": [
                "python run_attack.py", 
                "--dataset jailbreakbench",
                "--target-model meta-llama/Llama-2-7b-chat-hf",
                "--n-streams 15",
                "--n-iterations 3",
                "--gpu 0"
            ]
        },
        {
            "name": "Quick test with small model",
            "cmd": [
                "python run_attack.py",
                "--dataset advbench", 
                "--target-model microsoft/DialoGPT-small",
                "--n-streams 5",
                "--n-iterations 2",
                "--max-tokens 50",
                "--gpu 0"
            ]
        }
    ]
    
    print("=== PAIR Attack Usage Examples ===\n")
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}:")
        print("   " + " \\\n     ".join(example['cmd']))
        print()
    
    print("Required Environment Variables:")
    print("- OPENAI_API_KEY: Your OpenAI API key for GPT-4o")
    print()
    print("Output:")
    print("- Results saved to: ../results/PAIR/{dataset}/{model_name}/pair_results_{timestamp}_{hash}.jsonl")
    print("- Each line contains a complete attack result with all iterations")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run_example_attack()
    else:
        show_usage_examples()
        print("\nTo run a test attack, use: python example_usage.py --run")
