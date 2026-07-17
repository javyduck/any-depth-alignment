"""inference-cost benchmark for Any-Depth Alignment (ADA).

This module measures the wall-clock (and GPU-memory) cost of the two defenses
compared in the paper's inference-cost study:

* **ADA-LP overhead** ("regular LLM" path). ADA-LP re-injects the model's
  assistant header — the "Safety Tokens", ~1-3 extra tokens — and reads a single
  hidden state via a KV-cached forward. We therefore measure, on top of an
  already-built KV cache at a given context length:
    - the time to forward **one** extra token (``next_token``), and
    - the time to forward **three** extra tokens (``forward_three_tokens``),
  which brackets the 1-3 Safety-Token cost of a single ADA-LP probe.

* **Guardrail cost** ("defense model" path). An external safety classifier
  (Llama-Guard, ShieldGemma, Granite-Guardian, ...) must run a **full,
  un-cached forward pass** over the entire context to score it. We measure that
  forward pass time (``use_cache=False``) at each context length.

Both paths sweep a set of context lengths, run Flash Attention 2 where available
(falling back to the default kernel automatically), and repeat each measurement
``num_runs`` times after a warm-up run, reporting mean/std/min/max.

Model handling comes from the shared foundation: the ADA-LP model set is the
model registry (:func:`ada.registry.list_models`) and the guardrail set is
``configs/guardrails.yaml``. Per-model loading quirks are exposed as CLI flags so
no code branches on a model name:

* ``openai/gpt-oss-*`` ship MXFP4 weights and must be loaded with ``--dtype auto``.
* ``meta-llama/Llama-Guard-4-12B`` uses chunked attention; contexts longer than a
  chunk require ``--attention-chunk-size 8192`` (harmlessly ignored by models
  without that config field).

Run with::

    python -m ada.timing.benchmark --model-type regular
    python -m ada.timing.benchmark --model-type defense --attention-chunk-size 8192
    python -m ada.timing.benchmark --models openai/gpt-oss-120b --dtype auto

Results are written as one JSON per model plus a ``combined_timing_results.json``
under ``--output-dir`` (default ``timing_results/``); ``ada.timing.make_table``
assembles them into the paper's table/CSV.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from ..models.loading import load_model_and_tokenizer, load_tokenizer
from ..registry import list_models
from ..utils.naming import slugify_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context lengths swept in the paper's inference-cost study.
DEFAULT_TOKEN_LENGTHS: List[int] = [100, 500, 1000, 2500, 5000, 10000]

# configs/ lives next to the ada/ package at the repository root.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


# --------------------------------------------------------------------------- #
# Small measurement helpers
# --------------------------------------------------------------------------- #
def generate_random_tokens(tokenizer, length: int) -> List[int]:
    """Generate a random token sequence of the requested length.

    Token ids are drawn from the first ~10k of the vocabulary to avoid special
    tokens while remaining representative of ordinary content tokens.
    """
    vocab_size = len(tokenizer)
    safe_vocab_size = min(vocab_size, 10000)
    return [random.randint(1, safe_vocab_size - 1) for _ in range(length)]


def get_gpu_memory_usage() -> float:
    """Current allocated GPU memory in MB (0 if CUDA is unavailable)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def resolve_dtype(name: str):
    """Resolve a ``--dtype`` string to a torch dtype (or the literal ``"auto"``)."""
    if name == "auto":
        return "auto"
    if not hasattr(torch, name):
        raise ValueError(f"Unknown dtype '{name}'")
    return getattr(torch, name)


# --------------------------------------------------------------------------- #
# Model / benchmark-set resolution
# --------------------------------------------------------------------------- #
def load_regular_model_ids() -> List[str]:
    """The ADA-LP model set: every model declared in the registry."""
    return list_models()


def load_guardrail_model_ids(config_dir: Path) -> List[str]:
    """The guardrail baseline set, read from ``configs/guardrails.yaml``."""
    path = Path(config_dir) / "guardrails.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Guardrail config not found: {path}. Pass --config-dir or --models."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return [entry["hf_id"] for entry in raw.get("guardrails", [])]


def load_guardrail_timing_overrides(config_dir: Path) -> Dict[str, dict]:
    """Per-guardrail timing quirks from ``guardrails.yaml``.

    Reads the optional ``timing_attention_chunk_size`` / ``timing_use_flash_attention``
    fields so models like Llama-Guard-4-12B are benchmarked with the right
    chunked-attention span and attention kernel on the default reproduction path.
    """
    path = Path(config_dir) / "guardrails.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    overrides: Dict[str, dict] = {}
    for entry in raw.get("guardrails", []):
        ov = {}
        if "timing_attention_chunk_size" in entry:
            ov["attention_chunk_size"] = entry["timing_attention_chunk_size"]
        if "timing_use_flash_attention" in entry:
            ov["use_flash_attention"] = entry["timing_use_flash_attention"]
        if ov:
            overrides[entry["hf_id"]] = ov
    return overrides


def load_timing_model(
    model_name: str,
    dtype,
    device: str,
    use_flash_attention: bool = True,
    attention_chunk_size: Optional[int] = None,
):
    """Load an ``AutoModelForCausalLM`` + tokenizer for timing.

    When ``attention_chunk_size`` is ``None`` this delegates to
    :func:`ada.models.loading.load_model_and_tokenizer` (Flash Attention 2 with
    automatic fallback). When it is set, the model config's chunked-attention
    span is patched before loading — required by Llama-Guard-4-12B at contexts
    longer than a chunk — and the value is silently ignored by configs that do
    not expose it.
    """
    if attention_chunk_size is None:
        return load_model_and_tokenizer(
            model_name, dtype=dtype, device=device, use_flash_attention=use_flash_attention
        )

    tokenizer = load_tokenizer(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "text_config") and hasattr(config.text_config, "attention_chunk_size"):
        config.text_config.attention_chunk_size = attention_chunk_size
        logger.info("Set config.text_config.attention_chunk_size = %d", attention_chunk_size)
    elif hasattr(config, "attention_chunk_size"):
        config.attention_chunk_size = attention_chunk_size
        logger.info("Set config.attention_chunk_size = %d", attention_chunk_size)
    else:
        logger.warning("%s has no attention_chunk_size field; ignoring override", model_name)

    kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": device},
        "trust_remote_code": True,
        "config": config,
    }
    if use_flash_attention:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="flash_attention_2", **kwargs
            )
            logger.info("Loaded %s with Flash Attention 2 (%s)", model_name, dtype)
        except (ImportError, ValueError) as err:
            logger.warning("Flash Attention unavailable (%s); using default attention", err)
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


# --------------------------------------------------------------------------- #
# ADA-LP overhead (KV-cached forward of 1 / 3 extra Safety Tokens)
# --------------------------------------------------------------------------- #
def benchmark_regular_llm(
    model_name: str,
    token_lengths: List[int],
    num_runs: int = 10,
    device: str = "cuda",
    dtype=torch.bfloat16,
    use_flash_attention: bool = True,
    attention_chunk_size: Optional[int] = None,
) -> Dict:
    """Benchmark ADA-LP's KV-cached overhead on a generative model.

    For each context length a KV cache is built once, then we time forwarding
    one extra token and three extra tokens through that cache. Results include
    per-measurement GPU-memory deltas. On CUDA OOM the length is recorded with
    sentinel ``-1`` values and the sweep continues.
    """
    logger.info("Benchmarking regular LLM (ADA-LP overhead): %s", model_name)

    model, tokenizer = load_timing_model(
        model_name, dtype, device, use_flash_attention, attention_chunk_size
    )
    model.eval()

    results: Dict = {
        "model_name": model_name,
        "device": device,
        "token_lengths": {},
        "model_type": "regular_llm",
    }

    for token_length in token_lengths:
        logger.info("Testing token length: %d", token_length)

        length_results: Dict[str, List[float]] = {
            "next_token_times": [],
            "forward_three_tokens_times": [],
            "next_token_memory_deltas": [],
            "forward_three_tokens_memory_deltas": [],
        }
        oom_occurred = False

        for run in tqdm(
            range(num_runs + 1),
            desc=f"Running {token_length} tokens (1 dry run + {num_runs} measured)",
        ):
            try:
                random_tokens = generate_random_tokens(tokenizer, token_length)
                input_ids = torch.tensor([random_tokens], device=device)

                with torch.inference_mode():
                    # Build the KV cache once (its own timing is not recorded).
                    outputs = model(input_ids, use_cache=True)
                    past_key_values = outputs.past_key_values
                    torch.cuda.synchronize()

                    # Test 1: forward one extra token through the KV cache.
                    one_tokens = generate_random_tokens(tokenizer, 1)
                    one_token_input = torch.tensor([one_tokens], device=device)

                    torch.cuda.synchronize()
                    gc.collect()
                    memory_before = get_gpu_memory_usage()

                    start_time = time.perf_counter()
                    next_outputs = model(
                        one_token_input, past_key_values=past_key_values, use_cache=True
                    )
                    torch.cuda.synchronize()
                    next_token_time = time.perf_counter() - start_time

                    memory_after = get_gpu_memory_usage()
                    next_token_memory_delta = memory_after - memory_before

                    if run > 0:
                        length_results["next_token_times"].append(next_token_time)
                        length_results["next_token_memory_deltas"].append(next_token_memory_delta)

                    # Test 2: forward three extra tokens through the KV cache.
                    three_tokens = generate_random_tokens(tokenizer, 3)
                    three_token_input = torch.tensor([three_tokens], device=device)

                    torch.cuda.synchronize()
                    gc.collect()
                    memory_before_three = get_gpu_memory_usage()

                    start_time = time.perf_counter()
                    forward_outputs = model(
                        three_token_input, past_key_values=past_key_values, use_cache=False
                    )
                    torch.cuda.synchronize()
                    forward_time = time.perf_counter() - start_time

                    memory_after_three = get_gpu_memory_usage()
                    three_token_memory_delta = memory_after_three - memory_before_three

                    if run > 0:
                        length_results["forward_three_tokens_times"].append(forward_time)
                        length_results["forward_three_tokens_memory_deltas"].append(
                            three_token_memory_delta
                        )

                    del input_ids, one_token_input, three_token_input
                    del outputs, past_key_values, next_outputs, forward_outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                if "out of memory" in str(err).lower():
                    logger.warning("OOM at token length %d, run %d: %s", token_length, run, err)
                    oom_occurred = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                raise

        if oom_occurred or len(length_results["next_token_times"]) == 0:
            results["token_lengths"][token_length] = {
                "next_token_mean": -1.0,
                "next_token_std": -1.0,
                "next_token_min": -1.0,
                "next_token_max": -1.0,
                "next_token_memory_mean": -1.0,
                "next_token_memory_std": -1.0,
                "next_token_memory_min": -1.0,
                "next_token_memory_max": -1.0,
                "forward_three_tokens_mean": -1.0,
                "forward_three_tokens_std": -1.0,
                "forward_three_tokens_min": -1.0,
                "forward_three_tokens_max": -1.0,
                "forward_three_tokens_memory_mean": -1.0,
                "forward_three_tokens_memory_std": -1.0,
                "forward_three_tokens_memory_min": -1.0,
                "forward_three_tokens_memory_max": -1.0,
                "num_runs": 0,
                "oom": True,
            }
        else:
            results["token_lengths"][token_length] = {
                "next_token_mean": float(np.mean(length_results["next_token_times"])),
                "next_token_std": float(np.std(length_results["next_token_times"])),
                "next_token_min": float(np.min(length_results["next_token_times"])),
                "next_token_max": float(np.max(length_results["next_token_times"])),
                "next_token_memory_mean": float(np.mean(length_results["next_token_memory_deltas"])),
                "next_token_memory_std": float(np.std(length_results["next_token_memory_deltas"])),
                "next_token_memory_min": float(np.min(length_results["next_token_memory_deltas"])),
                "next_token_memory_max": float(np.max(length_results["next_token_memory_deltas"])),
                "forward_three_tokens_mean": float(
                    np.mean(length_results["forward_three_tokens_times"])
                ),
                "forward_three_tokens_std": float(
                    np.std(length_results["forward_three_tokens_times"])
                ),
                "forward_three_tokens_min": float(
                    np.min(length_results["forward_three_tokens_times"])
                ),
                "forward_three_tokens_max": float(
                    np.max(length_results["forward_three_tokens_times"])
                ),
                "forward_three_tokens_memory_mean": float(
                    np.mean(length_results["forward_three_tokens_memory_deltas"])
                ),
                "forward_three_tokens_memory_std": float(
                    np.std(length_results["forward_three_tokens_memory_deltas"])
                ),
                "forward_three_tokens_memory_min": float(
                    np.min(length_results["forward_three_tokens_memory_deltas"])
                ),
                "forward_three_tokens_memory_max": float(
                    np.max(length_results["forward_three_tokens_memory_deltas"])
                ),
                "num_runs": len(length_results["next_token_times"]),
                "oom": False,
            }

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    return results


# --------------------------------------------------------------------------- #
# Guardrail cost (full un-cached forward over the whole context)
# --------------------------------------------------------------------------- #
def benchmark_defense_model(
    model_name: str,
    token_lengths: List[int],
    num_runs: int = 10,
    device: str = "cuda",
    dtype=torch.bfloat16,
    use_flash_attention: bool = True,
    attention_chunk_size: Optional[int] = None,
) -> Dict:
    """Benchmark a guardrail's full, un-cached forward pass over the context.

    Measures the ``use_cache=False`` forward time (and memory) over the whole
    input sequence at each context length, plus the model's load-time memory
    footprint. On CUDA OOM the length is recorded with sentinel ``-1`` values.
    """
    logger.info("Benchmarking defense model (guardrail cost): %s", model_name)

    memory_before_model = get_gpu_memory_usage()
    model, tokenizer = load_timing_model(
        model_name, dtype, device, use_flash_attention, attention_chunk_size
    )
    model.eval()
    memory_after_model = get_gpu_memory_usage()
    model_memory_cost = memory_after_model - memory_before_model

    results: Dict = {
        "model_name": model_name,
        "device": device,
        "token_lengths": {},
        "model_type": "defense_model",
        "use_kv_cache": False,
        "model_memory_cost_mb": float(model_memory_cost),
    }

    for token_length in token_lengths:
        logger.info("Testing token length: %d", token_length)

        length_results: Dict[str, List[float]] = {
            "forward_times": [],
            "forward_memory_costs": [],
        }
        oom_occurred = False

        for run in tqdm(
            range(num_runs + 1),
            desc=f"Running {token_length} tokens (1 dry run + {num_runs} measured)",
        ):
            try:
                random_tokens = generate_random_tokens(tokenizer, token_length)
                input_ids = torch.tensor([random_tokens], device=device)

                with torch.inference_mode():
                    torch.cuda.synchronize()
                    gc.collect()
                    memory_before_forward = get_gpu_memory_usage()

                    start_time = time.perf_counter()
                    outputs = model(input_ids, use_cache=False)
                    torch.cuda.synchronize()
                    forward_time = time.perf_counter() - start_time

                    memory_after_forward = get_gpu_memory_usage()
                    forward_memory_cost = memory_after_forward - memory_before_forward

                    if run > 0:
                        length_results["forward_times"].append(forward_time)
                        length_results["forward_memory_costs"].append(forward_memory_cost)

                    del input_ids, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                if "out of memory" in str(err).lower():
                    logger.warning("OOM at token length %d, run %d: %s", token_length, run, err)
                    oom_occurred = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                raise

        if oom_occurred or len(length_results["forward_times"]) == 0:
            results["token_lengths"][token_length] = {
                "forward_mean": -1.0,
                "forward_std": -1.0,
                "forward_min": -1.0,
                "forward_max": -1.0,
                "forward_memory_mean": -1.0,
                "forward_memory_std": -1.0,
                "forward_memory_min": -1.0,
                "forward_memory_max": -1.0,
                "model_memory_cost": float(model_memory_cost),
                "num_runs": 0,
                "oom": True,
            }
        else:
            results["token_lengths"][token_length] = {
                "forward_mean": float(np.mean(length_results["forward_times"])),
                "forward_std": float(np.std(length_results["forward_times"])),
                "forward_min": float(np.min(length_results["forward_times"])),
                "forward_max": float(np.max(length_results["forward_times"])),
                "forward_memory_mean": float(np.mean(length_results["forward_memory_costs"])),
                "forward_memory_std": float(np.std(length_results["forward_memory_costs"])),
                "forward_memory_min": float(np.min(length_results["forward_memory_costs"])),
                "forward_memory_max": float(np.max(length_results["forward_memory_costs"])),
                "model_memory_cost": float(model_memory_cost),
                "num_runs": len(length_results["forward_times"]),
                "oom": False,
            }

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    return results


# --------------------------------------------------------------------------- #
# Reporting / persistence
# --------------------------------------------------------------------------- #
def print_results_summary(results: Dict) -> None:
    """Print a human-readable summary of one model's timing results."""
    print("\n" + "=" * 80)
    print(f"TIMING RESULTS: {results['model_name']}")
    print("=" * 80)
    print(f"Model type: {results['model_type']}")
    if "use_kv_cache" in results:
        print(f"KV-cache: {results['use_kv_cache']}")
    if "model_memory_cost_mb" in results:
        print(f"Model memory cost: {results['model_memory_cost_mb']:.2f} MB")
    print("-" * 80)

    for token_length, metrics in results["token_lengths"].items():
        print(f"\nToken length: {token_length}")

        if metrics.get("oom", False):
            print("  [OUT OF MEMORY] - No valid measurements")
            continue

        if "forward_mean" in metrics:
            print("  Forward pass:")
            print(f"    Mean: {metrics['forward_mean']:.6f}s")
            print(f"    Std:  {metrics['forward_std']:.6f}s")
            print(f"    Min:  {metrics['forward_min']:.6f}s")
            print(f"    Max:  {metrics['forward_max']:.6f}s")
            if "forward_memory_mean" in metrics:
                print("  Forward memory cost:")
                print(f"    Mean: {metrics['forward_memory_mean']:.2f} MB")
                print(f"    Std:  {metrics['forward_memory_std']:.2f} MB")
                print(f"    Min:  {metrics['forward_memory_min']:.2f} MB")
                print(f"    Max:  {metrics['forward_memory_max']:.2f} MB")
            if "model_memory_cost" in metrics:
                print(f"  Model memory cost: {metrics['model_memory_cost']:.2f} MB")

        if "next_token_mean" in metrics:
            print("  Next token generation:")
            print(f"    Mean: {metrics['next_token_mean']:.6f}s")
            print(f"    Std:  {metrics['next_token_std']:.6f}s")
            print(f"    Min:  {metrics['next_token_min']:.6f}s")
            print(f"    Max:  {metrics['next_token_max']:.6f}s")
            if "next_token_memory_mean" in metrics:
                print("  Next token memory delta:")
                print(f"    Mean: {metrics['next_token_memory_mean']:.2f} MB")
                print(f"    Std:  {metrics['next_token_memory_std']:.2f} MB")
                print(f"    Min:  {metrics['next_token_memory_min']:.2f} MB")
                print(f"    Max:  {metrics['next_token_memory_max']:.2f} MB")

        if "forward_three_tokens_mean" in metrics:
            print("  Forward 3 tokens:")
            print(f"    Mean: {metrics['forward_three_tokens_mean']:.6f}s")
            print(f"    Std:  {metrics['forward_three_tokens_std']:.6f}s")
            print(f"    Min:  {metrics['forward_three_tokens_min']:.6f}s")
            print(f"    Max:  {metrics['forward_three_tokens_max']:.6f}s")
            if "forward_three_tokens_memory_mean" in metrics:
                print("  Forward 3 tokens memory delta:")
                print(f"    Mean: {metrics['forward_three_tokens_memory_mean']:.2f} MB")
                print(f"    Std:  {metrics['forward_three_tokens_memory_std']:.2f} MB")
                print(f"    Min:  {metrics['forward_three_tokens_memory_min']:.2f} MB")
                print(f"    Max:  {metrics['forward_three_tokens_memory_max']:.2f} MB")

    print("=" * 80)


def save_results(results: Dict, output_dir: Path) -> Path:
    """Persist one model's results as ``{slug}_{model_type}_timing.json``."""
    model_slug = slugify_model(results["model_name"])
    filename = f"{model_slug}_{results['model_type']}_timing.json"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to: %s", output_path)
    return output_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="inference-cost benchmark: ADA-LP overhead vs guardrail forward pass.",
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific HF model ids to test (default: registry / guardrails)."
    )
    parser.add_argument(
        "--model-type",
        choices=["regular", "defense", "all"],
        default="all",
        help="Which benchmark(s) to run: ADA-LP models, guardrails, or both.",
    )
    parser.add_argument(
        "--token-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_TOKEN_LENGTHS,
        help="Context lengths (in tokens) to sweep.",
    )
    parser.add_argument(
        "--num-runs", type=int, default=50, help="Measured runs per length (after one warm-up)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="timing_results", help="Directory for JSON results."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="torch dtype for loading, or 'auto' (required for gpt-oss MXFP4 weights).",
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2 (default: enabled with automatic fallback).",
    )
    parser.add_argument(
        "--attention-chunk-size",
        type=int,
        default=None,
        help="Patch chunked-attention span before load (e.g. 8192 for Llama-Guard-4-12B).",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory holding guardrails.yaml (for the default guardrail set).",
    )
    return parser


def _run_set(
    model_names: List[str],
    benchmark_fn,
    args,
    device: str,
    dtype,
    use_flash_attention: bool,
    all_results: List[Dict],
    output_dir: Path,
    overrides: Optional[Dict[str, dict]] = None,
) -> None:
    overrides = overrides or {}
    for model_name in model_names:
        try:
            logger.info("%s", "=" * 60)
            logger.info("Testing %s", model_name)
            logger.info("%s", "=" * 60)
            # Per-model timing quirks (e.g. Llama-Guard-4 chunked attention / no FA2),
            # from configs/guardrails.yaml, so the default reproduction path is correct.
            ov = overrides.get(model_name, {})
            chunk = ov.get("attention_chunk_size", args.attention_chunk_size)
            fa = ov.get("use_flash_attention", use_flash_attention)
            # gpt-oss ships MXFP4 weights that must load with dtype="auto"; select it
            # per-model (matching the source) so the default sweep works without the
            # caller having to pass --dtype auto.
            model_dtype = "auto" if "gpt-oss" in model_name.lower() else dtype
            results = benchmark_fn(
                model_name=model_name,
                token_lengths=args.token_lengths,
                num_runs=args.num_runs,
                device=device,
                dtype=model_dtype,
                use_flash_attention=fa,
                attention_chunk_size=chunk,
            )
            print_results_summary(results)
            save_results(results, output_dir)
            all_results.append(results)
        except Exception as err:  # noqa: BLE001 - keep sweeping other models
            logger.error("Failed to benchmark %s: %s", model_name, err)
        finally:
            _cleanup_cuda()


def main() -> None:
    args = build_arg_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda"
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    dtype = resolve_dtype(args.dtype)
    use_flash_attention = not args.no_flash_attention

    logger.info("Starting timing benchmarks...")
    logger.info("GPU: %d | dtype: %s | flash-attn: %s", args.gpu, args.dtype, use_flash_attention)
    logger.info("Token lengths: %s", args.token_lengths)
    logger.info("Number of runs: %d", args.num_runs)
    logger.info("Output directory: %s", output_dir)

    all_results: List[Dict] = []

    if args.model_type in ("regular", "all"):
        regular_models = args.models if args.models else load_regular_model_ids()
        _run_set(
            regular_models, benchmark_regular_llm, args, device, dtype,
            use_flash_attention, all_results, output_dir,
        )

    if args.model_type in ("defense", "all"):
        defense_models = args.models if args.models else load_guardrail_model_ids(config_dir)
        _run_set(
            defense_models, benchmark_defense_model, args, device, dtype,
            use_flash_attention, all_results, output_dir,
            overrides=load_guardrail_timing_overrides(config_dir),
        )

    combined_results = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "config": {
            "token_lengths": args.token_lengths,
            "num_runs": args.num_runs,
            "gpu": args.gpu,
            "seed": args.seed,
            "model_type": args.model_type,
        },
        "results": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined_timing_results.json"
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(combined_results, fh, indent=2)

    logger.info("Combined results saved to: %s", combined_path)
    logger.info("Individual results saved in: %s", output_dir)
    logger.info("Benchmarking completed!")


if __name__ == "__main__":
    main()
