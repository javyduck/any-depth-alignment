#!/usr/bin/env python3
"""
Example usage of the GCG attack script.

This script demonstrates how to run GCG attacks on different datasets and models.
"""

import subprocess
import sys
from pathlib import Path


def run_example():
    """Run example GCG attacks."""
    
    # Example 1: Single attack on AdvBench with Mistral-7B-Instruct-v0.2
    print("=== Example 1: Single Attack - AdvBench + Mistral-7B-Instruct-v0.2 ===")
    cmd1 = [
        "python", "run_gcg_attack.py",
        "--dataset", "advbench",
        "--model", "mistralai/Mistral-7B-Instruct-v0.2",
        "--gpu", "0",
        "--num_steps", "100",  # Reduced steps for testing
    ]
    
    print(f"Command: {' '.join(cmd1)}")
    print("Run this command to test the attack on AdvBench")
    print()
    
    # Example 2: Batch attack on multiple models
    print("=== Example 2: Batch Attack - Multiple Models ===")
    cmd2 = [
        "./run_gcg_batch.sh",
        "--model", "mistralai/Mistral-7B-Instruct-v0.2",
        "--model", "meta-llama/Llama-2-7b-chat-hf"
    ]
    
    print(f"Command: {' '.join(cmd2)}")
    print("Run this command to test attacks on both AdvBench and JailbreakBench for multiple models")
    print()
    
    # Example 3: Single attack with custom parameters
    print("=== Example 3: Single Attack with Custom Parameters ===")
    cmd3 = [
        "python", "run_gcg_attack.py",
        "--dataset", "advbench",
        "--model", "mistralai/Mistral-7B-Instruct-v0.2",
        "--gpu", "0",
        "--num_steps", "500",
        "--search_width", "256",
        "--topk", "128",
        "--seed", "42",
        "--dtype", "bfloat16"
    ]
    
    print(f"Command: {' '.join(cmd3)}")
    print("Run this command for a full attack with custom parameters")
    print()
    
    print("=== Output Structure ===")
    print("Results will be saved to:")
    print("../results/GCG/{dataset}/{model_slug}/gcg_results_{num_steps}steps_{seed}seed.jsonl")
    print()
    print("Each line in the JSONL file contains:")
    print("- prompt: Original harmful prompt")
    print("- target: Target string (starts with 'Sure')")
    print("- best_string: Optimized adversarial string")
    print("- generation: Model's response to the adversarial prompt")
    print("- success: Boolean indicating if generation starts with 'Sure'")
    print("- best_loss: Final loss value from GCG optimization")
    print("- num_steps_used: Number of GCG steps actually used")


if __name__ == "__main__":
    run_example()
