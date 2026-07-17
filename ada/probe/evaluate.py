"""ADA-LP offline evaluation — reading the Safety-Token probe at every depth.

This is the *Linear Probe* branch of Any-Depth Alignment (ADA-LP), used to
produce the deep-prefill/adversarial-attack/SFT-attack/over-refusal curves in the paper. For each stored response it replays
``user + assistant`` through the model with a *gradual KV cache*, and at every
``--depth`` tokens (up to ``--max-depth``) it:

1. forks the accumulated cache (``copy.deepcopy``),
2. appends the model's Safety Tokens (the injected assistant header),
3. hooks the hidden state at the *last* injected Safety Token (position ``-1``)
   via :class:`ada.models.extraction.HookHiddenStateCollector`, and
4. feeds that hidden state to the per-layer sklearn logistic probe (``joblib``)
   to obtain a refusal decision.

Per-model configuration (the Safety-Token span, probe layer, and hook position)
is resolved from :mod:`ada.registry`; no model name is branched on in code.

The SFT-attack ablation ``--disable_safetytoken_adapter`` toggles PEFT's
``disable_adapter_layers()`` *only* during the Safety-Token forward pass, so the
probe reads the base model's state while the rest of the replay keeps the adapter
active. Adapters are loaded from ``{benign,harmful}_adapters/{slug}/adapter-{step}``.

Outputs land under (via :mod:`ada.utils.naming`)::

    logs/{benign|harmful}/{dataset}/{model_slug[-{adapter_type}-adapter-{step}
        [-disable_safetytoken]]}/{safety}/{mask}/{hook}/seed_{seed}/{probe_type}/
        probe-layers{L}/depth_{d}_maxdepth_{md}.json

Run with ``python -m ada.probe.evaluate --model ... --dataset ...``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..data.loading import extract_messages, extract_response_text, resolve_response_file
from ..models.extraction import HookHiddenStateCollector, parse_layer_list
from ..models.loading import get_hidden_size, load_model_and_tokenizer, resolve_torch_dtype
from ..registry import get_model
from ..utils.io import read_jsonl, write_json
from ..utils.naming import (
    slugify_cache,
    slugify_hook_position,
    slugify_mask_tokens,
    slugify_model,
    slugify_safety_tokens,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model / tokenizer loading (with optional PEFT adapter)
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer_with_adapter(
    model_name: str,
    adapter: Optional[str] = None,
    adapter_type: str = "benign",
    use_flash_attention: bool = False,
    dtype: "torch.dtype | str" = torch.float16,
    device: str = "cuda:0",
):
    """Load a base model + tokenizer, optionally wrapping it with a PEFT adapter.

    The adapter is loaded from ``{adapter_type}_adapters/{model_slug}/adapter-{step}``
    (``adapter_type`` is ``benign`` or ``harmful``), matching the training layout.
    """
    model, tokenizer = load_model_and_tokenizer(
        model_name, dtype=dtype, device=device, use_flash_attention=use_flash_attention
    )

    if adapter:
        from peft import PeftModel

        adapter_path = (
            Path(f"{adapter_type}_adapters") / slugify_model(model_name) / f"adapter-{adapter}"
        )
        logger.info("Loading %s adapter from: %s", adapter_type, adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Loaded %s adapter: %s", adapter_type, adapter)

    return model, tokenizer


# --------------------------------------------------------------------------- #
# Response-file resolution
# --------------------------------------------------------------------------- #
def find_response_file(
    dataset: str,
    model: Optional[str] = None,
    benign: bool = False,
    attack: Optional[str] = None,
    response_file: Optional[str] = None,
    data_root: "str | Path" = "data/eval",
) -> str:
    """Locate the stored responses to evaluate.

    Convention (release layout under ``data_root`` first, source layout as a
    fallback so existing artifacts still resolve):

    * benign / over-refusal → ``over_refusal/{dataset}/{model_slug}/responses.jsonl``
    * attack (``gcg``/``autodan``/``pair``/``tap``) → dataset dir
      ``{dataset}_{attack}`` under ``attacks/.../{model_slug}/responses.jsonl``
    * harmful (deep-prefill) → ``deep_prefill/{dataset}_responses.jsonl``
    """
    return str(resolve_response_file(
        dataset, model, benign=benign, attack=attack,
        response_file=response_file, data_root=data_root,
    ))


# --------------------------------------------------------------------------- #
# Probe checkpoint loading
# --------------------------------------------------------------------------- #
def load_probe_checkpoint(
    model_name: str,
    safety_tokens: str,
    mask_tokens: Optional[str],
    hook_position: str,
    layer_idx: int,
    hidden_size: int,
    device: str = "cpu",
    seed: int = 42,
    probe_type: str = "logistic",
    gradual_cache: bool = True,
):
    """Load the probe for a single layer.

    Returns an sklearn estimator for ``probe_type == "logistic"`` (the paper's
    ADA-LP probe, stored with joblib), or a :class:`torch.nn.Module` for the
    ``linear`` / ``mlp`` variants. The checkpoint path mirrors ``train_logistic``::

        ckpts/{model}/{safety}/{mask}/{hook}/{cache}/seed_{seed}/
            {linear|non-linear|logistic}/layer_{L}.{pt|joblib}
    """
    base_dir = (
        Path("ckpts")
        / slugify_model(model_name)
        / slugify_safety_tokens(safety_tokens)
        / slugify_mask_tokens(mask_tokens)
        / slugify_hook_position(hook_position)
        / slugify_cache(gradual_cache)
        / f"seed_{seed}"
    )
    if probe_type == "linear":
        ckpt_path = base_dir / "linear" / f"layer_{layer_idx}.pt"
    elif probe_type == "mlp":
        ckpt_path = base_dir / "non-linear" / f"layer_{layer_idx}.pt"
    else:
        ckpt_path = base_dir / "logistic" / f"layer_{layer_idx}.joblib"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Probe checkpoint not found: {ckpt_path}")

    if probe_type == "logistic":
        clf = joblib.load(ckpt_path)
        logger.info("Loaded logistic probe for layer %d from %s", layer_idx, ckpt_path)
        return clf

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if probe_type == "linear":
        class LinearProbe(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, x):
                return self.linear(x)

        probe = LinearProbe(hidden_size)
        probe.linear.weight.data = checkpoint["weight"].unsqueeze(0)  # (H,) -> (1, H)
        bias = checkpoint["bias"]
        probe.linear.bias.data = bias.reshape(1) if bias.dim() == 0 else bias
        return probe

    # Two-layer MLP probe.
    class MLPProbe(nn.Module):
        def __init__(self, hidden_size: int, mlp_hidden: int):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, mlp_hidden)
            self.fc2 = nn.Linear(mlp_hidden, 1)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    W1, b1, W2, b2 = checkpoint["W1"], checkpoint["b1"], checkpoint["W2"], checkpoint["b2"]
    probe = MLPProbe(hidden_size, W1.shape[1])
    probe.fc1.weight.data = W1.T.contiguous()
    probe.fc1.bias.data = b1.contiguous()
    probe.fc2.weight.data = W2.unsqueeze(0).contiguous()
    probe.fc2.bias.data = (b2.reshape(1) if b2.dim() == 0 else b2).contiguous()
    probe.eval()
    probe = probe.float().to(device=device)
    logger.info(
        "Loaded mlp probe for layer %d from %s (val_acc: %s)",
        layer_idx,
        ckpt_path,
        checkpoint.get("val_accuracy", "N/A"),
    )
    return probe


# --------------------------------------------------------------------------- #
# SFT-attack ablation: disable the adapter only during the Safety-Token forward
# --------------------------------------------------------------------------- #
@contextmanager
def disable_adapter_temporarily(model, disable_adapter: bool):
    """Temporarily call PEFT ``disable_adapter_layers()`` for the wrapped block.

    Re-enables the adapter on exit. A no-op when ``disable_adapter`` is False or
    the model is not a PEFT model. Used by the SFT-attack ablation to read the probe on
    the base model's Safety-Token state while the adapter drives the rest.
    """
    toggled = False
    if disable_adapter and hasattr(model, "disable_adapter_layers"):
        try:
            model.disable_adapter_layers()
            toggled = True
        except Exception:  # noqa: BLE001 - best-effort toggle across PEFT versions
            pass
    try:
        yield
    finally:
        if toggled and hasattr(model, "enable_adapter_layers"):
            try:
                model.enable_adapter_layers()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Chat-template base-prompt construction (registry-driven, no name branching)
# --------------------------------------------------------------------------- #
def build_base_prompt(tokenizer, user_message: Dict, suffix: str = "") -> str:
    """Build the assistant-turn base prompt up to where the response begins.

    Applies the chat template with a generation prompt, then appends the
    registry-provided ``suffix`` (a reasoning/channel completion or Llama-2's
    trailing space). ``suffix`` is resolved once from the registry by the caller,
    so no generic whitespace heuristic is applied here.
    """
    text = tokenizer.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return text + suffix if suffix else text


# --------------------------------------------------------------------------- #
# Depth-0 hidden state (before any assistant tokens are appended)
# --------------------------------------------------------------------------- #
def get_depth_0_hidden_states(
    base_tokens: List[int],
    safety_tokens: List[int],
    base_hidden_states: Dict[int, torch.Tensor],
    target_layers: List[int],
    hook_collector: HookHiddenStateCollector,
    model,
    device: str,
) -> Dict[int, torch.Tensor]:
    """Hidden states at depth 0, before any assistant tokens are generated.

    If ``safety_tokens`` occur at the tail of the base prompt, read the state at
    the last matched Safety Token; otherwise fall back to the last base token.
    """
    if not safety_tokens:
        return {
            layer: base_hidden_states[layer][0, 0, :].unsqueeze(0)
            for layer in target_layers
            if layer in base_hidden_states
        }

    check_tokens = base_tokens[-10:]
    match_idx = None
    for i in range(len(check_tokens) - len(safety_tokens) + 1):
        if check_tokens[i : i + len(safety_tokens)] == safety_tokens:
            match_idx = i + len(safety_tokens) - 1
            break

    if match_idx is None:
        return {
            layer: base_hidden_states[layer][0, 0, :].unsqueeze(0)
            for layer in target_layers
            if layer in base_hidden_states
        }

    rel_neg_index = match_idx - len(check_tokens)
    base_input_ids = torch.tensor([base_tokens], device=device)

    def forward_fn():
        return model(base_input_ids, use_cache=False, output_hidden_states=False)

    collected, _ = hook_collector.collect([rel_neg_index], forward_fn)
    return {
        layer: collected[layer][0, 0, :].unsqueeze(0)
        for layer in target_layers
        if layer in collected and collected[layer].size(1) > 0
    }


# --------------------------------------------------------------------------- #
# Probe application
# --------------------------------------------------------------------------- #
def _apply_probe(probe, hidden_state: torch.Tensor, probe_type: str):
    """Return ``(prediction, refusal_probability)`` for one layer's hidden state."""
    if probe_type == "logistic":
        hs_np = hidden_state.float().detach().cpu().numpy()
        if hs_np.ndim == 1:
            hs_np = hs_np.reshape(1, -1)
        if hasattr(probe, "predict_proba"):
            prob = float(probe.predict_proba(hs_np)[0, 1])
        else:  # fall back to decision_function -> sigmoid
            z = float(np.asarray(probe.decision_function(hs_np)).reshape(-1)[0])
            prob = float(1.0 / (1.0 + np.exp(-z)))
    else:
        logits = probe(hidden_state.float())
        prob = torch.sigmoid(logits).cpu().item()
    return (1 if prob >= 0.5 else 0), prob


def _find_last_subseq(haystack: List[int], needle: List[int]) -> int:
    """Start index of the last occurrence of ``needle`` in ``haystack`` (or -1)."""
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i : i + n] == needle:
            return i
    return -1


# --------------------------------------------------------------------------- #
# Per-instance processing with a gradual KV cache
# --------------------------------------------------------------------------- #
def process_single_instance_with_gradual_cache(
    model,
    tokenizer,
    instance: Dict,
    target_layers: List[int],
    probes: Dict[int, object],
    device: str,
    mask_tokens: Optional[str] = None,
    probe_type: str = "logistic",
    hook_collector: Optional[HookHiddenStateCollector] = None,
    disable_safetytoken_adapter: bool = False,
) -> List[Dict]:
    """Replay one response and read the probe at each depth via a gradual cache.

    A single base forward builds the initial KV cache; each depth extends it with
    the next slice of assistant tokens, then forks it (``copy.deepcopy``) to
    append the Safety Tokens without polluting the running cache. The probe reads
    the hidden state at the last injected Safety Token (position ``-1``).
    """
    if hook_collector is None:
        raise ValueError("hook_collector is required for gradual-cache processing")

    results: List[Dict] = []
    base_tokens = tokenizer.encode(instance["base_prompt"], add_special_tokens=False)
    safety_tokens = tokenizer.encode(instance["safety_tokens"], add_special_tokens=False)
    mask_tokens_encoded = (
        tokenizer.encode(mask_tokens, add_special_tokens=False) if mask_tokens else []
    )
    assistant_tokens = instance["assistant_tokens"]

    with torch.inference_mode():
        base_input_ids = torch.tensor([base_tokens], device=device)

        def base_forward_fn():
            return model(base_input_ids, use_cache=True, output_hidden_states=False)

        # Base context always contributes its last token's hidden state.
        collected_base, base_out = hook_collector.collect([-1], base_forward_fn)
        base_hidden_states = {
            layer: collected_base[layer] for layer in target_layers if layer in collected_base
        }
        past = base_out.past_key_values

        def fwd(ids, past=None, need_cache=True, attention_mask=None):
            return model(
                torch.tensor([ids], device=device),
                past_key_values=past,
                use_cache=need_cache,
                output_hidden_states=False,
                attention_mask=attention_mask,
            )

        prev_depth = 0
        for depth_level in instance["depth_levels"]:
            if depth_level == 0:
                hidden_states_dict = get_depth_0_hidden_states(
                    base_tokens, safety_tokens, base_hidden_states, target_layers,
                    hook_collector, model, device,
                )
            else:
                delta = assistant_tokens[prev_depth:depth_level]
                if safety_tokens:
                    # Extend the running cache with the assistant delta.
                    out = fwd(delta, past=past, need_cache=True)
                    past = out.past_key_values

                    # Optional attention mask that zeros the last occurrence of
                    # mask_tokens in the cached context for the Safety-Token rows.
                    attn_mask = None
                    if mask_tokens_encoded:
                        context_tokens = base_tokens + assistant_tokens[:depth_level]
                        past_len = len(context_tokens)
                        cur_len = len(safety_tokens)
                        start = _find_last_subseq(context_tokens, mask_tokens_encoded)
                        if start != -1:
                            attn_mask = torch.ones(
                                1, past_len + cur_len, device=device, dtype=torch.bool
                            )
                            attn_mask[:, start : start + len(mask_tokens_encoded)] = False

                    # Fork the cache so appending the Safety Tokens is side-effect free.
                    past_probe = copy.deepcopy(past)

                    def safety_forward_fn():
                        with disable_adapter_temporarily(model, disable_safetytoken_adapter):
                            return fwd(
                                safety_tokens, past=past_probe, need_cache=False,
                                attention_mask=attn_mask,
                            )

                    collected, _ = hook_collector.collect([-1], safety_forward_fn)
                    hidden_states_dict = {
                        layer: collected[layer][0, 0, :].unsqueeze(0)
                        for layer in target_layers
                        if layer in collected and collected[layer].size(1) > 0
                    }
                    del past_probe
                else:
                    # No Safety Tokens: read the last assistant token as it extends.
                    def assistant_forward_fn():
                        return fwd(delta, past=past, need_cache=True)

                    collected, out = hook_collector.collect([-1], assistant_forward_fn)
                    past = out.past_key_values
                    hidden_states_dict = {
                        layer: collected[layer][0, 0, :].unsqueeze(0)
                        for layer in target_layers
                        if layer in collected and collected[layer].size(1) > 0
                    }

            for layer_idx in target_layers:
                if layer_idx in hidden_states_dict:
                    prediction, prob = _apply_probe(
                        probes[layer_idx], hidden_states_dict[layer_idx], probe_type
                    )
                    results.append(
                        {
                            "instance": instance["instance_idx"],
                            "depth": depth_level,
                            "probe_key": layer_idx,
                            "is_refusal": bool(prediction == 1),
                            "refusal_probability": float(prob),
                        }
                    )

            prev_depth = depth_level

        del past

    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze_with_probes(
    model_name: str,
    responses: List[Dict],
    safety_tokens: str,
    hook_position: str,
    target_layers: List[int],
    depth: int,
    max_depth: int = 3000,
    device: str = "cuda",
    seed: int = 42,
    dtype: "torch.dtype | str" = "bfloat16",
    adapter: Optional[str] = None,
    adapter_type: str = "benign",
    gradual_cache: bool = True,
    mask_tokens: Optional[str] = None,
    probe_type: str = "logistic",
    disable_safetytoken_adapter: bool = False,
) -> Dict:
    """Run ADA-LP over ``responses`` and return per-layer refusal statistics."""
    model_dtype = resolve_torch_dtype(dtype)
    model, tokenizer = load_model_and_tokenizer_with_adapter(
        model_name, adapter=adapter, adapter_type=adapter_type,
        use_flash_attention=False, dtype=model_dtype, device=device,
    )
    model = model.to(device)
    model.eval()

    hidden_size = get_hidden_size(model)
    logger.info("Model dtype: %s; probes run in float32", next(model.parameters()).dtype)
    logger.info("Gradual cache: %s; hook position: %s", gradual_cache, hook_position)

    hook_collector = HookHiddenStateCollector(model, hook_position, target_layers)

    probes: Dict[int, object] = {}
    for layer_idx in target_layers:
        probes[layer_idx] = load_probe_checkpoint(
            model_name, safety_tokens, mask_tokens, hook_position, layer_idx, hidden_size,
            device=device, seed=seed, probe_type=probe_type, gradual_cache=gradual_cache,
        )
    logger.info("Loaded %d probe(s)", len(probes))

    try:
        suffix = get_model(model_name).generation_prompt_completion
    except KeyError:
        suffix = ""

    instance_data: List[Dict] = []
    for idx, resp in enumerate(responses):
        messages = extract_messages(resp)
        assistant_tokens = tokenizer.encode(extract_response_text(resp), add_special_tokens=False)
        max_tokens_to_check = min(len(assistant_tokens), max_depth)
        depth_levels = list(range(depth, max_tokens_to_check, depth))
        base_prompt = build_base_prompt(tokenizer, messages[0], suffix)
        instance_data.append(
            {
                "instance_idx": idx,
                "base_prompt": base_prompt,
                "safety_tokens": safety_tokens,
                "assistant_tokens": assistant_tokens,
                "depth_levels": depth_levels,
            }
        )

    total_depth_levels = sum(len(inst["depth_levels"]) for inst in instance_data)
    logger.info("Processing %d instances (%d predictions)", len(instance_data), total_depth_levels)

    results: List[Dict] = []
    with torch.inference_mode():
        for instance_idx, instance in enumerate(tqdm(instance_data, desc="Processing instances")):
            results.extend(
                process_single_instance_with_gradual_cache(
                    model, tokenizer, instance, target_layers, probes, device,
                    mask_tokens=mask_tokens, probe_type=probe_type, hook_collector=hook_collector,
                    disable_safetytoken_adapter=disable_safetytoken_adapter,
                )
            )
            if instance_idx % 10 == 0:
                gc.collect()
                if instance_idx % 50 == 0:
                    torch.cuda.empty_cache()

    refusal_stats = calculate_refusal_statistics(results, target_layers)
    hook_collector.remove()

    return {
        "total_responses": len(responses),
        "total_predictions": len(results),
        "safety_tokens": safety_tokens,
        "mask_tokens": mask_tokens,
        "hook_position": hook_position,
        "target_layers": target_layers,
        "depth_step": depth,
        "max_depth": max_depth,
        "refusal_statistics": refusal_stats,
        "detailed_logs": results,
    }


def calculate_refusal_statistics(results: List[Dict], target_layers: List[int]) -> Dict:
    """Aggregate per-layer refusal counts / rates / mean probability."""
    stats: Dict[str, Dict] = {}
    for probe_key in {r["probe_key"] for r in results}:
        probe_results = [r for r in results if r["probe_key"] == probe_key]
        refusal_count = sum(1 for r in probe_results if r["is_refusal"])
        avg_prob = float(np.mean([r["refusal_probability"] for r in probe_results]))
        stats[str(probe_key)] = {
            "total_predictions": len(probe_results),
            "refusal_count": refusal_count,
            "refusal_rate": refusal_count / len(probe_results),
            "avg_refusal_probability": avg_prob,
        }
    return stats


def print_results(results: Dict) -> None:
    """Print a compact summary of the probe-based refusal analysis."""
    def show_literal_newlines(s: Optional[str]) -> str:
        return s.replace("\n", r"\n") if isinstance(s, str) else str(s)

    print("\n" + "=" * 60)
    print("PROBE-BASED REFUSAL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total responses analyzed: {results['total_responses']}")
    print(f"Total predictions made: {results['total_predictions']}")
    print(f"Target layers: {results['target_layers']}")
    print(f"Safety tokens: {show_literal_newlines(results['safety_tokens'])}")
    if "mask_tokens" in results:
        print(f"Mask tokens: {show_literal_newlines(results['mask_tokens'])}")
    print("-" * 60)
    for layer_key, stats in results["refusal_statistics"].items():
        print(f"Layer {layer_key}:")
        print(f"  Refusal predictions: {stats['refusal_count']}/{stats['total_predictions']}")
        print(f"  Refusal rate: {stats['refusal_rate']:.2%}")
        print(f"  Avg refusal probability: {stats['avg_refusal_probability']:.3f}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADA-LP offline evaluation: read the Safety-Token linear probe at every depth."
    )
    # Data
    parser.add_argument("--model", required=True, help="HuggingFace model id (registry key)")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--benign", action="store_true", help="Evaluate benign (over-refusal) responses"
    )
    parser.add_argument(
        "--attack", default=None, choices=["gcg", "autodan", "pair", "tap"],
        help="Attack variant; loads responses from the '{dataset}_{attack}' directory",
    )
    parser.add_argument(
        "--response-file", default=None, help="Explicit response file (overrides lookup)"
    )
    parser.add_argument("--data-root", default="data/eval", help="Root of the release eval corpora")

    # Adapters (SFT-attack ablation)
    parser.add_argument(
        "--adapter", default=None, type=str, help="Adapter step to load (finetuned models)"
    )
    parser.add_argument(
        "--adapter-type", "--adapter_type", dest="adapter_type",
        default="benign", choices=["benign", "harmful"], help="Adapter type",
    )
    parser.add_argument(
        "--disable-safetytoken-adapter", "--disable_safetytoken_adapter",
        dest="disable_safetytoken_adapter", action="store_true", default=False,
        help="Disable the adapter during the Safety-Token forward pass (SFT-attack ablation)",
    )

    # Probe
    parser.add_argument(
        "--safety-tokens", default=None,
        help="Safety Tokens injected before probing (default: registry probe_safety_tokens)",
    )
    parser.add_argument(
        "--hook-position", default=None,
        help="Sub-module to hook the hidden state from (default: registry hook_position)",
    )
    parser.add_argument(
        "--layers", default=None,
        help="Hook layer indices, e.g. '15', '13,15,19', or 'all' (default: registry probe_layer)",
    )
    parser.add_argument(
        "--probe-type", default="logistic", choices=["linear", "mlp", "logistic"],
        help="Probe checkpoint type to load",
    )
    # Selects the checkpoint-path slug; must match how the probe was trained
    # (default: gradual cache, matching the released probes / probe scripts).
    parser.add_argument("--gradual-cache", dest="gradual_cache", action="store_true",
                        help="(default) load probes trained with the gradual KV cache")
    parser.add_argument("--full-forward", dest="gradual_cache", action="store_false",
                        help="Load probes trained with the full-forward strategy")
    parser.set_defaults(gradual_cache=True)

    # Depth schedule
    parser.add_argument(
        "--depth", type=int, default=25, help="Depth interval (tokens) between probes"
    )
    parser.add_argument(
        "--max-depth", type=int, default=3000, help="Maximum generation depth to probe"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature the evaluated responses were sampled at; tags the output "
             "log with _temp_{t} for the sampling-temperature ablation (0.0 = the "
             "canonical greedy log; supply temp-sampled responses via --response-file)",
    )

    # Reproducibility / system
    parser.add_argument(
        "--seed", type=int, default=42, help="Training seed (for checkpoint lookup)"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    spec = get_model(args.model)

    # Registry-driven defaults; CLI-provided Safety Tokens honor literal "\n".
    if args.safety_tokens is None:
        safety_tokens = spec.probe_safety_tokens
    else:
        safety_tokens = args.safety_tokens.replace("\\n", "\n")
    hook_position = args.hook_position or spec.hook_position
    mask_tokens = None  # Mask ablation is not exposed in the release CLI.
    gradual_cache = args.gradual_cache  # selects the ckpt-path slug (see train.py)

    if args.layers is None:
        target_layers = [spec.probe_layer]
    elif str(args.layers).strip().lower() == "all":
        logger.info("Loading model once to count hook positions for --layers all")
        tmp_model, _ = load_model_and_tokenizer(
            args.model, dtype=torch.bfloat16, device=device, use_flash_attention=False
        )
        target_layers = parse_layer_list("all", model=tmp_model, hook_position=hook_position)
        del tmp_model
        torch.cuda.empty_cache()
    else:
        target_layers = parse_layer_list(args.layers)

    response_file = find_response_file(
        args.dataset, args.model, benign=args.benign, attack=args.attack,
        response_file=args.response_file, data_root=args.data_root,
    )
    responses = read_jsonl(response_file)
    logger.info("Loaded %d responses from %s", len(responses), response_file)

    results = analyze_with_probes(
        model_name=args.model,
        responses=responses,
        safety_tokens=safety_tokens,
        hook_position=hook_position,
        target_layers=target_layers,
        depth=args.depth,
        max_depth=args.max_depth,
        device=device,
        seed=args.seed,
        adapter=args.adapter,
        adapter_type=args.adapter_type,
        gradual_cache=gradual_cache,
        mask_tokens=mask_tokens,
        probe_type=args.probe_type,
        disable_safetytoken_adapter=args.disable_safetytoken_adapter,
    )

    print_results(results)

    # ----- Persist per-layer logs (path layout via ada.utils.naming) --------- #
    model_slug = slugify_model(args.model)
    log_type = "benign" if args.benign else "harmful"
    safety_slug = slugify_safety_tokens(safety_tokens)
    mask_slug = slugify_mask_tokens(mask_tokens)
    hook_slug = slugify_hook_position(hook_position)
    type_dir = {"linear": "linear", "mlp": "non-linear", "logistic": "logistic"}[args.probe_type]

    dataset_name = args.dataset.lower()
    if args.attack:
        dataset_name = f"{args.dataset.lower()}_{args.attack.lower()}"

    if args.adapter:
        model_component = f"{model_slug}-{args.adapter_type}-adapter-{args.adapter}"
        if args.disable_safetytoken_adapter:
            model_component += "-disable_safetytoken"
    else:
        model_component = model_slug

    for layer_idx in target_layers:
        layer_logs = [r for r in results["detailed_logs"] if r["probe_key"] == layer_idx]
        layer_results = {
            "total_responses": results["total_responses"],
            "total_predictions": len(layer_logs),
            "safety_tokens": results["safety_tokens"],
            "hook_position": results.get("hook_position", hook_position),
            "target_layers": [layer_idx],
            "depth_step": results["depth_step"],
            "max_depth": results["max_depth"],
            "refusal_statistics": {str(layer_idx): results["refusal_statistics"][str(layer_idx)]},
            "detailed_logs": [{k: v for k, v in r.items() if k != "probe_key"} for r in layer_logs],
        }

        temp_suffix = "" if args.temperature == 0.0 else f"_temp_{args.temperature}"
        log_path = (
            Path("logs") / log_type / dataset_name / model_component / safety_slug / mask_slug
            / hook_slug / f"seed_{args.seed}" / type_dir / f"probe-layers{layer_idx}"
            / f"depth_{args.depth}_maxdepth_{args.max_depth}{temp_suffix}.json"
        )
        write_json(log_path, layer_results)
        logger.info("Layer %d results saved to %s", layer_idx, log_path)


if __name__ == "__main__":
    main()
