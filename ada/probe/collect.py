"""ADA-LP hidden-state collection (E1).

Any-Depth Alignment's Linear-Probe variant (ADA-LP) reads the hidden state of an
injected Safety-Token span placed after a partial assistant response. This module
sweeps a fixed schedule of *generation depths* over benign and harmful response
corpora and dumps, for every requested transformer block, the hidden state at the
injected Safety Token (or, for the token-choice ablation, at every prefix of the
Safety-Token span).

For each response it re-injects the model's Safety Tokens right after the first
``d`` assistant tokens, runs a forward pass, and captures the hidden state at the
hook position (``input_layernorm`` by default). Two execution strategies produce
identical states:

* **gradual KV cache** (default; ``--gradual-cache``): prime the cache once, extend
  it with each depth delta, and fork the past via ``copy.deepcopy`` for the
  Safety-Token probe so the growing cache is never mutated. This is the convention
  the released probes, the E1 scripts, and ``ada.probe.evaluate`` all use.
* **full forward** (``--full-forward``): re-tokenize ``user + assistant[:d] + safety``
  and run one fresh forward pass per depth, optionally with a 4D additive attention
  mask that hides the previous assistant header from the Safety-Token rows
  (``--mask-tokens``). Produces states identical to the gradual cache.

Outputs land under (all path components come from :mod:`ada.utils.naming`)::

    {out}/{split}/{model}/{benign|harmful}/{safety}/{mask}/{hook}/{cache}/index_{i}/{layer}.pt

Each ``{layer}.pt`` is a ``bf16`` tensor of shape ``N x len(DEPTHS) x H`` and is
accompanied by a metadata JSON. Collection runs are shardable across GPUs via
``--index`` (which slices the corpus into eighths); periodic per-shard ``.pt``
files are written and merged under cross-process file locks so many processes can
collect concurrently for the same model+hook combination.

Run as ``python -m ada.probe.collect --model ... --harmful`` (or ``--benign``).
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import random
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..data.loading import extract_messages, load_harmful_responses
from ..models.extraction import (
    HookHiddenStateCollector,
    count_hook_positions,
    parse_layer_list,
)
from ..models.loading import gpu_memory_summary, load_model_and_tokenizer
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

# Fixed generation-depth schedule used to build the ADA-LP probe corpus. Depth 0
# reads the header at the very start of the assistant turn; the rest re-inject the
# Safety Tokens after that many assistant tokens. Do NOT change (reproducibility).
# Depths at which hidden states are sampled: every 25 tokens up to 500, matching
# the paper (§ probe corpus: "truncate to 500 tokens and sample every 25 tokens",
# yielding the reported 600k/60k train/val feature points).
DEPTHS: List[int] = list(range(0, 501, 25))

# --------------------------------------------------------------------------- #
# Cross-process file locking (concurrent collection across GPUs / processes)
# --------------------------------------------------------------------------- #
def _cleanup_stale_lock(lock_path: Path, ttl_sec: int = 900) -> bool:
    """Delete ``lock_path`` if older than ``ttl_sec``; return True if cleaned."""
    try:
        st = os.stat(lock_path)
        if time.time() - st.st_mtime > ttl_sec:
            os.unlink(lock_path)
            return True
    except FileNotFoundError:
        pass
    return False


@contextmanager
def file_lock(lock_path: Path, poll: float = 0.1, ttl_sec: int = 900):
    """Simple cross-process lock using ``O_EXCL`` with stale-lock cleanup."""
    fd = None
    max_retries = 200  # high-contention scenarios (8 GPUs writing shards)
    retry_count = 0
    base_sleep = poll

    while retry_count < max_retries:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {socket.gethostname()} {int(time.time())}\n".encode())
            break
        except FileExistsError:
            if _cleanup_stale_lock(lock_path, ttl_sec):
                continue
            sleep_time = min(base_sleep * (1.1 ** retry_count), 5.0) + random.random() * 0.5
            time.sleep(sleep_time)
            retry_count += 1
        except OSError as err:
            if err.errno == 16:  # device or resource busy
                if retry_count % 20 == 0:
                    logger.warning("Lock busy (errno 16), retry %d/%d: %s",
                                   retry_count, max_retries, lock_path)
                sleep_time = min(base_sleep * (1.1 ** retry_count), 5.0) + random.random() * 0.5
                time.sleep(sleep_time)
                retry_count += 1
            else:
                raise

    if fd is None:
        raise RuntimeError(f"Failed to acquire lock after {max_retries} retries: {lock_path}")

    try:
        yield
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(str(lock_path))
        except FileNotFoundError:
            pass


def get_global_lock_path(
    base_output_dir: Path, model_name: str, hook_position: str, gpu_index: Optional[int] = None
) -> Path:
    """Lock path covering all data types / splits / operations for a model+hook."""
    model_slug = slugify_model(model_name)
    hook_slug = slugify_hook_position(hook_position)
    if gpu_index is not None:
        return base_output_dir / f".collection_lock_{model_slug}_{hook_slug}_gpu{gpu_index}"
    return base_output_dir / f".global_collection_lock_{model_slug}_{hook_slug}"


@contextmanager
def global_collection_lock(
    base_output_dir: Path,
    model_name: str,
    hook_position: str,
    operation: str = "",
    gpu_index: Optional[int] = None,
):
    """Prevent conflicts across concurrent collection scenarios.

    Serializes save/merge operations across processes for the same model+hook
    combination. With ``gpu_index`` set, uses a per-GPU lock to reduce contention.
    """
    lock_path = get_global_lock_path(base_output_dir, model_name, hook_position, gpu_index)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    op_info = f" [{operation}]" if operation else ""
    gpu_info = f" [GPU {gpu_index}]" if gpu_index is not None else " [GLOBAL]"
    logger.debug("Acquiring collection lock%s%s: %s", gpu_info, op_info, lock_path)
    with file_lock(lock_path, poll=0.5, ttl_sec=1800):
        logger.debug("Acquired collection lock%s%s: %s", gpu_info, op_info, lock_path)
        yield
        logger.debug("Released collection lock%s%s: %s", gpu_info, op_info, lock_path)


def atomic_torch_save(tensor: torch.Tensor, target_path: Path) -> None:
    """Save a tensor atomically via a temporary file and rename."""
    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(tensor, tmp, _use_new_zipfile_serialization=False, pickle_protocol=5)
        os.replace(tmp, target_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Safety-Token prefixes (token-choice ablation) and small helpers
# --------------------------------------------------------------------------- #
def get_safety_token_prefixes(safety_tokens: str, tokenizer) -> List[str]:
    """Return every prefix of the Safety-Token span (powers ``--collect-all-tokens``).

    The i-th prefix decodes the first ``i`` Safety-Token ids; index 0 is the
    special ``"empty"`` bucket (the assistant token that precedes the span). These
    strings become the per-prefix output sub-directories for the token ablation.
    """
    if not safety_tokens:
        return ["empty"]
    encoded = tokenizer.encode(safety_tokens, add_special_tokens=False)
    prefixes = []
    for i in range(len(encoded) + 1):
        prefixes.append("empty" if i == 0 else tokenizer.decode(encoded[:i], skip_special_tokens=False))
    return prefixes


def find_last_subseq(haystack: List[int], needle: List[int]) -> int:
    """Return the start index of the last occurrence of ``needle``, else ``-1``."""
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def set_random_seeds(seed: int) -> None:
    """Seed ``random`` / ``numpy`` / ``torch`` for reproducible collection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Per-sample hidden-state collection
# --------------------------------------------------------------------------- #
def _store_states(
    all_hidden_states,
    collected,
    target_layers: List[int],
    collect_all_tokens: bool,
    safety_prefixes: Optional[List[str]],
    token_positions: List[int],
) -> None:
    """Append hidden states from one forward pass into ``all_hidden_states``.

    For ``collect_all_tokens`` the collected token positions map 1:1 onto the
    Safety-Token prefixes; otherwise only the last (probe) token is stored.
    """
    for layer_idx in target_layers:
        if layer_idx not in collected:
            continue
        states = collected[layer_idx]
        if collect_all_tokens:
            for i, prefix in enumerate(safety_prefixes[:len(token_positions)]):
                if i < states.size(1):
                    all_hidden_states[prefix][layer_idx].append(states[0, i, :])
        elif states.size(1) > 0:
            all_hidden_states[layer_idx].append(states[0, -1, :])


def _collect_gradual(
    model,
    hook_collector: HookHiddenStateCollector,
    target_layers: List[int],
    device,
    *,
    user_tokens: List[int],
    assistant_tokens: List[int],
    safety_tokens_encoded: List[int],
    mask_tokens_encoded: List[int],
    depths: List[int],
    collect_all_tokens: bool,
    safety_prefixes: Optional[List[str]],
    all_hidden_states,
) -> None:
    """Gradual-KV-cache strategy: extend the cache per depth, fork it for probing."""
    with torch.inference_mode():
        def fwd(ids, past=None, need_cache=True, attention_mask=None):
            input_ids = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
            return model(
                input_ids,
                past_key_values=past,
                use_cache=need_cache,
                output_hidden_states=False,  # hooks capture the states instead
                attention_mask=attention_mask,
            )

        past, prev_depth = None, 0
        # Carried last-assistant-token state, reused when a depth delta is empty
        # (short responses). Initialised so the reuse path can never NameError.
        prev_last_assistant_states: Dict[int, torch.Tensor] = {}

        # Depth 0: read the header at the start of the assistant turn.
        if 0 in depths:
            if collect_all_tokens:
                token_positions = list(range(-(len(safety_tokens_encoded) + 1), 0))
            else:
                token_positions = [-1]
            collected, out = hook_collector.collect(
                token_positions, lambda: fwd(user_tokens, past=None, need_cache=True)
            )
            past = out.past_key_values
            _store_states(all_hidden_states, collected, target_layers,
                          collect_all_tokens, safety_prefixes, token_positions)
            del collected

        for depth in depths:
            if depth == 0:
                continue

            delta = assistant_tokens[prev_depth:depth]
            out = None
            last_assistant_states: Dict[int, torch.Tensor] = {}

            if collect_all_tokens and delta:
                # Capture the last assistant token so the combined span reads
                # [last_assistant, safety_token_1, ...] across two forward passes.
                assistant_collected, out = hook_collector.collect(
                    [-1], lambda: fwd(delta, past=past, need_cache=True)
                )
                for layer_idx in target_layers:
                    if layer_idx in assistant_collected:
                        last_assistant_states[layer_idx] = assistant_collected[layer_idx][0, 0, :]
                del assistant_collected
            elif delta:
                out = fwd(delta, past=past, need_cache=True)
            else:
                # Depth delta empty (response shorter than this depth): reuse the
                # previous last-assistant state and leave the cache unchanged.
                last_assistant_states = prev_last_assistant_states

            if out is not None:
                past = out.past_key_values

            if safety_tokens_encoded:
                # 2D key-padding mask zeroing the previous header in the cache.
                attn_mask = None
                if mask_tokens_encoded:
                    context_tokens = user_tokens + assistant_tokens[:depth]
                    past_len = len(context_tokens)
                    cur_len = len(safety_tokens_encoded)
                    start = find_last_subseq(context_tokens, mask_tokens_encoded)
                    if start != -1:
                        attn_mask = torch.ones(1, past_len + cur_len, device=device, dtype=torch.long)
                        attn_mask[:, start:start + len(mask_tokens_encoded)] = 0

                # Fork the growing cache so the probe pass never mutates it.
                past_probe = copy.deepcopy(past)

                if collect_all_tokens:
                    token_positions = list(range(-len(safety_tokens_encoded), 0))
                else:
                    token_positions = [-1]

                collected, _ = hook_collector.collect(
                    token_positions,
                    lambda: fwd(safety_tokens_encoded, past=past_probe,
                                need_cache=False, attention_mask=attn_mask),
                )

                for layer_idx in target_layers:
                    if layer_idx not in collected:
                        continue
                    if collect_all_tokens:
                        combined = []
                        if layer_idx in last_assistant_states:
                            combined.append(last_assistant_states[layer_idx])
                        for i in range(collected[layer_idx].size(1)):
                            combined.append(collected[layer_idx][0, i, :])
                        for i, prefix in enumerate(safety_prefixes[:len(combined)]):
                            all_hidden_states[prefix][layer_idx].append(combined[i])
                    elif collected[layer_idx].size(1) > 0:
                        all_hidden_states[layer_idx].append(collected[layer_idx][0, -1, :])
                del collected, past_probe
            else:
                # No Safety Tokens: fall back to the captured last-assistant state.
                for layer_idx in target_layers:
                    if layer_idx in last_assistant_states:
                        if collect_all_tokens:
                            first_prefix = safety_prefixes[0] if safety_prefixes else "empty"
                            all_hidden_states[first_prefix][layer_idx].append(
                                last_assistant_states[layer_idx])
                        else:
                            all_hidden_states[layer_idx].append(last_assistant_states[layer_idx])

            prev_last_assistant_states = last_assistant_states
            prev_depth = depth

        del past


def _collect_full_forward(
    model,
    hook_collector: HookHiddenStateCollector,
    target_layers: List[int],
    device,
    *,
    user_tokens: List[int],
    assistant_tokens: List[int],
    safety_tokens_encoded: List[int],
    mask_tokens_encoded: List[int],
    depths: List[int],
    collect_all_tokens: bool,
    safety_prefixes: Optional[List[str]],
    all_hidden_states,
) -> None:
    """Full-forward strategy: one fresh forward pass per depth (optionally 4D mask)."""
    with torch.inference_mode():
        def fwd(ids, need_cache=False, attention_mask=None):
            return model(
                torch.tensor([ids], dtype=torch.long, device=device),
                use_cache=need_cache,
                output_hidden_states=False,  # hooks capture the states instead
                attention_mask=attention_mask,
            )

        n_safe = len(safety_tokens_encoded)
        for depth in depths:
            if depth == 0:
                # No Safety Tokens inserted; read the last n_safe+1 user positions.
                if collect_all_tokens:
                    token_positions = list(range(-(n_safe + 1), 0))
                else:
                    token_positions = [-1]
                collected, _ = hook_collector.collect(
                    token_positions, lambda: fwd(user_tokens, need_cache=False)
                )
                _store_states(all_hidden_states, collected, target_layers,
                              collect_all_tokens, safety_prefixes, token_positions)
                del collected
                continue

            context_ids = user_tokens + assistant_tokens[:depth]
            input_ids = (context_ids + safety_tokens_encoded) if safety_tokens_encoded else context_ids

            # 4D additive mask: causal base plus hiding the previous header from
            # ONLY the Safety-Token query rows, preserving causal structure.
            attn_mask = None
            if safety_tokens_encoded and mask_tokens_encoded:
                past_len = len(context_ids)
                q_len = len(input_ids)
                start = find_last_subseq(context_ids, mask_tokens_encoded)
                if start != -1:
                    model_dtype = next(model.parameters()).dtype
                    mask_val = torch.finfo(model_dtype).min
                    causal = torch.triu(
                        torch.full((q_len, q_len), mask_val, device=device, dtype=model_dtype),
                        diagonal=1,
                    )
                    attn_mask = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, q_len, kv_len)
                    attn_mask[:, :, past_len:q_len, start:start + len(mask_tokens_encoded)] = mask_val

            if collect_all_tokens and safety_tokens_encoded:
                token_positions = list(range(-(n_safe + 1), 0))
            else:
                token_positions = [-1]

            collected, _ = hook_collector.collect(
                token_positions, lambda: fwd(input_ids, need_cache=False, attention_mask=attn_mask)
            )

            for layer_idx in target_layers:
                if layer_idx not in collected:
                    continue
                states = collected[layer_idx]
                if collect_all_tokens:
                    if safety_tokens_encoded:
                        for i, prefix in enumerate(safety_prefixes[:len(token_positions)]):
                            if i < states.size(1):
                                all_hidden_states[prefix][layer_idx].append(states[0, i, :])
                    elif states.size(1) > 0:
                        last_state = states[0, -1, :]
                        for prefix in safety_prefixes:
                            all_hidden_states[prefix][layer_idx].append(last_state)
                elif states.size(1) > 0:
                    all_hidden_states[layer_idx].append(states[0, -1, :])
            del collected


def collect_hidden_states_single(
    model,
    tokenizer,
    sample: Dict,
    target_layers: List[int],
    *,
    collect_all_tokens: bool = False,
    max_tokens: int = 500,
    safety_tokens: str = "",
    gradual_cache: bool = False,
    device=None,
    prompt_completion: str = "",
    mask_tokens: Optional[str] = None,
    hook_position: str = "input_layernorm",
    hook_collector: Optional[HookHiddenStateCollector] = None,
) -> Union[Dict[int, torch.Tensor], Dict[str, Dict[int, torch.Tensor]]]:
    """Collect Safety-Token hidden states for one response across all depths.

    Returns ``{layer: (len(DEPTHS), H)}`` or, with ``collect_all_tokens``,
    ``{prefix: {layer: (len(DEPTHS), H)}}``.
    """
    if device is None:
        device = next(model.parameters()).device
    if hook_collector is None:
        hook_collector = HookHiddenStateCollector(model, hook_position, target_layers)

    safety_prefixes = get_safety_token_prefixes(safety_tokens, tokenizer) if collect_all_tokens else None
    if collect_all_tokens:
        all_hidden_states = {p: {l: [] for l in target_layers} for p in safety_prefixes}
    else:
        all_hidden_states = {l: [] for l in target_layers}

    messages = extract_messages(sample)
    user_message = messages[0]
    assistant_message = messages[1]["content"]

    # Build the user turn + generation prompt, then finish the assistant header so
    # the response continues correctly. ``prompt_completion`` carries the
    # per-model completion (reasoning/channel suffix, or Llama-2's trailing space);
    # it is resolved once from the registry, so no generic heuristic is applied.
    conversation_text = tokenizer.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if prompt_completion:
        conversation_text += prompt_completion

    user_tokens = tokenizer.encode(conversation_text, add_special_tokens=False)
    assistant_tokens = tokenizer.encode(assistant_message, add_special_tokens=False)
    if len(assistant_tokens) > max_tokens:
        assistant_tokens = assistant_tokens[:max_tokens]

    safety_tokens_encoded = (
        tokenizer.encode(safety_tokens, add_special_tokens=False) if safety_tokens else []
    )
    mask_tokens_encoded = (
        tokenizer.encode(mask_tokens, add_special_tokens=False) if mask_tokens else []
    )

    strategy = _collect_gradual if gradual_cache else _collect_full_forward
    strategy(
        model, hook_collector, target_layers, device,
        user_tokens=user_tokens,
        assistant_tokens=assistant_tokens,
        safety_tokens_encoded=safety_tokens_encoded,
        mask_tokens_encoded=mask_tokens_encoded,
        depths=DEPTHS,
        collect_all_tokens=collect_all_tokens,
        safety_prefixes=safety_prefixes,
        all_hidden_states=all_hidden_states,
    )

    # Stack per-depth states (already on CPU) into (len(DEPTHS), H) tensors.
    if collect_all_tokens:
        final = {}
        for prefix in all_hidden_states:
            final[prefix] = {
                layer: torch.stack(all_hidden_states[prefix][layer])
                for layer in target_layers
                if all_hidden_states[prefix][layer]
            }
        return final
    return {
        layer: torch.stack(all_hidden_states[layer])
        for layer in target_layers
        if all_hidden_states[layer]
    }


# --------------------------------------------------------------------------- #
# Sharded save / merge
# --------------------------------------------------------------------------- #
def save_hidden_states_batch(
    all_hidden_states,
    base_parent_dir: Path,
    target_layers: List[int],
    model_dtype,
    collect_all_tokens: bool,
    safety_tokens: str,
    mask_tokens: Optional[str],
    tokenizer,
    batch_start: int,
    batch_end: int,
    index: int,
    gradual_cache: bool,
    no_empty: bool,
    hook_position: str,
    model_name: str,
) -> None:
    """Write one batch of collected states as intermediate ``*_batch_*.pt`` shards."""
    logger.info("Saving hidden states batch %d-%d...", batch_start, batch_end)
    base_output_dir = base_parent_dir.parent.parent  # {out}/{split}
    mask_slug = slugify_mask_tokens(mask_tokens)
    hook_slug = slugify_hook_position(hook_position)
    cache_slug = slugify_cache(gradual_cache)

    with global_collection_lock(base_output_dir, model_name, hook_position,
                                f"save_batch_{batch_start}_{batch_end}", gpu_index=index):
        if collect_all_tokens:
            if no_empty:
                logger.info("Skipping 'empty' prefix due to --no-empty flag")
            for prefix in get_safety_token_prefixes(safety_tokens, tokenizer):
                if no_empty and prefix == "empty":
                    continue
                if not (prefix in all_hidden_states and all_hidden_states[prefix]):
                    continue
                prefix_dir = (base_parent_dir / slugify_safety_tokens(prefix)
                              / mask_slug / hook_slug / cache_slug / f"index_{index}")
                for layer_idx in target_layers:
                    buf = all_hidden_states[prefix].get(layer_idx)
                    if not buf:
                        continue
                    _save_shard(buf, model_dtype,
                                prefix_dir / f"{layer_idx}_batch_{batch_start}_{batch_end}.pt")
        else:
            base_dir = (base_parent_dir / slugify_safety_tokens(safety_tokens)
                        / mask_slug / hook_slug / cache_slug / f"index_{index}")
            for layer_idx in target_layers:
                buf = all_hidden_states.get(layer_idx)
                if not buf:
                    continue
                _save_shard(buf, model_dtype,
                            base_dir / f"{layer_idx}_batch_{batch_start}_{batch_end}.pt")


def _save_shard(buffer: List[torch.Tensor], model_dtype, layer_path: Path) -> None:
    """Stack buffered per-sample tensors, cast to ``model_dtype`` and save atomically."""
    layer_data = torch.stack(buffer)
    if model_dtype is not None and hasattr(layer_data, "dtype") and layer_data.dtype != model_dtype:
        layer_data = layer_data.to(dtype=model_dtype)
    lock = layer_path.with_suffix(layer_path.suffix + ".lock")
    with file_lock(lock):
        t0 = time.perf_counter()
        atomic_torch_save(layer_data, layer_path)
        logger.info("Saved %s in %.2fs", layer_path, time.perf_counter() - t0)


def merge_batch_files(
    base_parent_dir: Path,
    target_layers: List[int],
    collect_all_tokens: bool,
    safety_tokens: str,
    mask_tokens: Optional[str],
    tokenizer,
    index: int,
    gradual_cache: bool,
    no_empty: bool,
    hook_position: str,
    model_name: str,
) -> None:
    """Concatenate all ``*_batch_*.pt`` shards into final ``{layer}.pt`` files."""
    logger.info("Merging batch files into final layer files...")
    base_output_dir = base_parent_dir.parent.parent  # {out}/{split}
    mask_slug = slugify_mask_tokens(mask_tokens)
    hook_slug = slugify_hook_position(hook_position)
    cache_slug = slugify_cache(gradual_cache)

    with global_collection_lock(base_output_dir, model_name, hook_position,
                                "merge_batch_files", gpu_index=index):
        if collect_all_tokens:
            if no_empty:
                logger.info("Merging: skipping 'empty' prefix due to --no-empty flag")
            for prefix in get_safety_token_prefixes(safety_tokens, tokenizer):
                if no_empty and prefix == "empty":
                    continue
                prefix_dir = (base_parent_dir / slugify_safety_tokens(prefix)
                              / mask_slug / hook_slug / cache_slug / f"index_{index}")
                if not prefix_dir.exists():
                    continue
                for layer_idx in target_layers:
                    _merge_layer(prefix_dir, layer_idx, label=f"Prefix: {prefix}, Layer: {layer_idx}")
        else:
            base_dir = (base_parent_dir / slugify_safety_tokens(safety_tokens)
                        / mask_slug / hook_slug / cache_slug / f"index_{index}")
            for layer_idx in target_layers:
                _merge_layer(base_dir, layer_idx, label=f"Layer: {layer_idx}")


def _merge_layer(directory: Path, layer_idx: int, label: str) -> None:
    """Concatenate a layer's shards into ``{layer}.pt`` and delete the shards."""
    batch_files = sorted(directory.glob(f"{layer_idx}_batch_*.pt"))
    if not batch_files:
        return
    tensors = [torch.load(bf, map_location="cpu", weights_only=False) for bf in batch_files]
    if not tensors:
        return
    final_tensor = torch.cat(tensors, dim=0)
    logger.info("Final save - %s, Shape: %s", label, tuple(final_tensor.shape))
    final_path = directory / f"{layer_idx}.pt"
    lock = final_path.with_suffix(final_path.suffix + ".lock")
    with file_lock(lock):
        atomic_torch_save(final_tensor, final_path)
    for bf in batch_files:
        bf.unlink()


def save_metadata(args, out_dir: Path, data_type: str, index: int,
                  safety_tokens: str, mask_tokens: Optional[str], hook_position: str) -> None:
    """Write a metadata JSON recording the collection hyperparameters."""
    metadata = {
        "hyperparameters": {
            "model": args.model,
            "safety_tokens": safety_tokens,
            "mask_tokens": mask_tokens,
            "hook_position": hook_position,
            "depths": DEPTHS,
            "index": args.index,
            "gradual_cache": args.gradual_cache,
            "collect_all_tokens": args.collect_all_tokens,
            "dtype": args.dtype,
            "max_tokens": args.max_tokens,
            "layers": args.layers,
            "save_interval": args.save_interval,
        },
        "data_type": data_type,
        "index": index,
    }
    write_json(out_dir / f"{data_type}_metadata_index_{index}.json", metadata)


# --------------------------------------------------------------------------- #
# Corpus processing
# --------------------------------------------------------------------------- #
def process_responses(
    model,
    tokenizer,
    responses: List[Dict],
    target_layers: List[int],
    args,
    output_dir: Path,
    data_type: str,
    device,
    model_dtype,
    *,
    safety_tokens: str,
    mask_tokens: Optional[str],
    hook_position: str,
    prompt_completion: str,
) -> int:
    """Collect and shard hidden states for a whole corpus, then merge the shards."""
    if not responses:
        logger.warning("No %s samples available", data_type)
        return 0

    model_slug = slugify_model(args.model)
    mask_slug = slugify_mask_tokens(mask_tokens)
    hook_slug = slugify_hook_position(hook_position)
    cache_slug = slugify_cache(args.gradual_cache)
    base_parent_dir = output_dir / model_slug / data_type
    base_parent_dir.mkdir(parents=True, exist_ok=True)

    # Create output directories + per-directory metadata.
    if args.collect_all_tokens:
        safety_prefixes = get_safety_token_prefixes(safety_tokens, tokenizer)
        logger.info("safety_prefixes: %s", safety_prefixes)
        for prefix in safety_prefixes:
            prefix_dir = (base_parent_dir / slugify_safety_tokens(prefix)
                          / mask_slug / hook_slug / cache_slug / f"index_{args.index}")
            prefix_dir.mkdir(parents=True, exist_ok=True)
            save_metadata(args, prefix_dir, data_type, args.index,
                          safety_tokens, mask_tokens, hook_position)
        all_hidden_states = {p: {l: [] for l in target_layers} for p in safety_prefixes}
    else:
        base_dir = (base_parent_dir / slugify_safety_tokens(safety_tokens)
                    / mask_slug / hook_slug / cache_slug / f"index_{args.index}")
        base_dir.mkdir(parents=True, exist_ok=True)
        save_metadata(args, base_dir, data_type, args.index,
                      safety_tokens, mask_tokens, hook_position)
        all_hidden_states = {l: [] for l in target_layers}

    logger.info("Initializing hook collector for %s at %d positions...",
                hook_position, len(target_layers))
    hook_collector = HookHiddenStateCollector(model, hook_position, target_layers)

    processed_count = 0
    pbar = tqdm(total=len(responses), desc=f"Processing {data_type} samples", unit="sample")
    for sample_idx, sample in enumerate(responses):
        hidden_states = collect_hidden_states_single(
            model, tokenizer, sample, target_layers,
            collect_all_tokens=args.collect_all_tokens,
            max_tokens=args.max_tokens,
            safety_tokens=safety_tokens,
            gradual_cache=args.gradual_cache,
            device=device,
            prompt_completion=prompt_completion,
            mask_tokens=mask_tokens,
            hook_position=hook_position,
            hook_collector=hook_collector,
        )

        collected_any = False
        if args.collect_all_tokens:
            for prefix, prefix_states in hidden_states.items():
                for layer_idx in target_layers:
                    if layer_idx in prefix_states:
                        all_hidden_states[prefix][layer_idx].append(prefix_states[layer_idx])
                        collected_any = True
        else:
            for layer_idx in target_layers:
                if layer_idx in hidden_states:
                    all_hidden_states[layer_idx].append(hidden_states[layer_idx])
                    collected_any = True

        if collected_any:
            processed_count += 1
        else:
            logger.warning("No hidden states collected for sample %d", sample_idx)

        del hidden_states
        if sample_idx % 10 == 0:
            gc.collect()
            if sample_idx % 50 == 0:
                torch.cuda.empty_cache()

        # Periodic flush: move buffers out (no aliasing) and write a shard.
        if processed_count and processed_count % args.save_interval == 0:
            batch_start = processed_count - args.save_interval
            batch_end = processed_count - 1
            states_to_save: Dict = {}
            if args.collect_all_tokens:
                for prefix in all_hidden_states:
                    states_to_save[prefix] = {}
                    for layer_idx in target_layers:
                        buf = all_hidden_states[prefix][layer_idx]
                        if buf:
                            states_to_save[prefix][layer_idx] = buf
                            all_hidden_states[prefix][layer_idx] = []
            else:
                for layer_idx in target_layers:
                    buf = all_hidden_states[layer_idx]
                    if buf:
                        states_to_save[layer_idx] = buf
                        all_hidden_states[layer_idx] = []
            save_hidden_states_batch(
                states_to_save, base_parent_dir, target_layers, model_dtype,
                args.collect_all_tokens, safety_tokens, mask_tokens, tokenizer,
                batch_start, batch_end, args.index, args.gradual_cache,
                args.no_empty, hook_position, args.model,
            )

        pbar.update(1)
    pbar.close()

    # Flush any remaining (partial final batch).
    if processed_count > 0 and processed_count % args.save_interval != 0:
        if any(all_hidden_states[key] for key in all_hidden_states):
            batch_start = (processed_count // args.save_interval) * args.save_interval
            batch_end = processed_count - 1
            save_hidden_states_batch(
                all_hidden_states, base_parent_dir, target_layers, model_dtype,
                args.collect_all_tokens, safety_tokens, mask_tokens, tokenizer,
                batch_start, batch_end, args.index, args.gradual_cache,
                args.no_empty, hook_position, args.model,
            )

    merge_batch_files(
        base_parent_dir, target_layers, args.collect_all_tokens, safety_tokens,
        mask_tokens, tokenizer, args.index, args.gradual_cache, args.no_empty,
        hook_position, args.model,
    )
    del hook_collector
    logger.info("Collected hidden states for %d %s samples with %d depths each",
                processed_count, data_type, len(DEPTHS))
    return processed_count


def _select_eighth(responses: List[Dict], index: int) -> List[Dict]:
    """Return the ``index``-th eighth of ``responses`` (last eighth takes remainder)."""
    eighth = len(responses) // 8
    start = index * eighth
    end = start + eighth if index < 7 else len(responses)
    logger.info("Using eighth %d: samples %d-%d (%d samples)", index, start, end, end - start)
    return responses[start:end]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _unescape(value: Optional[str]) -> Optional[str]:
    """Turn a literal ``\\n`` in a CLI-provided string into a real newline."""
    if value is None:
        return None
    return value.replace("\\n", "\n")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect ADA-LP Safety-Token hidden states from benign/harmful responses"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id (must be in the registry)")
    parser.add_argument("--safety-tokens", default=None,
                        help="Safety Tokens to inject (default: registry probe_safety_tokens). "
                             "Escape sequences like \\n are interpreted as newlines.")
    parser.add_argument("--mask-tokens", default=None,
                        help="String whose most recent occurrence in the existing context is "
                             "attention-masked for Safety-Token rows (default: none/no masking). "
                             "Escape sequences like \\n are interpreted as newlines.")
    parser.add_argument("--hook-position", default=None,
                        help="Sub-module within each block to read (default: registry hook_position)")
    parser.add_argument("--layers", default="all",
                        help="Hook indices to collect, 0-based (e.g. 'all', '15', '13,15,19')")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Data split")
    parser.add_argument("--benign", action="store_true", help="Collect from benign responses")
    parser.add_argument("--harmful", action="store_true", help="Collect from harmful responses")
    parser.add_argument("--index", type=int, choices=list(range(8)), default=0,
                        help="Corpus eighth (0-7) for sharding across GPUs")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum assistant response length per instance")
    parser.add_argument("--depths", default=None,
                        help="Generation depths to sample, comma/space separated "
                             "(default: the paper schedule 0,25,...,500). Overriding this "
                             "changes the probe corpus; keep the default to reproduce the paper.")
    # Gradual KV cache is the default (matches the released probes, the E1 scripts,
    # and ada.probe.evaluate, so bare collect->train->evaluate paths line up).
    parser.add_argument("--gradual-cache", dest="gradual_cache", action="store_true",
                        help="(default) gradual KV-cache strategy: prime once, fork per depth")
    parser.add_argument("--full-forward", dest="gradual_cache", action="store_false",
                        help="Use one fresh forward pass per depth instead (identical states, slower)")
    parser.set_defaults(gradual_cache=True)
    parser.add_argument("--collect-all-tokens", action="store_true",
                        help="Collect states for every Safety-Token prefix (token-choice ablation)")
    parser.add_argument("--save-interval", type=int, default=1250,
                        help="Write a shard every N processed samples")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "float32", "bfloat16", "auto"], help="Model precision")
    parser.add_argument("--no-empty", action="store_true",
                        help="Skip the 'empty' prefix during save/merge (collect-all-tokens)")
    parser.add_argument("--output-dir", default="hidden_states",
                        help="Base directory for collected hidden states")
    parser.add_argument("--harmful-file", default=None,
                        help="Harmful responses JSONL "
                             "(default: data/train/probe/harmful/wildjailbreak/{split}_responses.jsonl)")
    parser.add_argument("--benign-files", nargs="+", default=None,
                        help="Benign responses JSONL file(s), concatenated "
                             "(default: data/train/probe/benign/{wildchat1m,wildjailbreak}/"
                             "{split}_responses.jsonl)")
    return parser


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]


def main() -> None:
    args = create_argument_parser().parse_args()

    if not args.benign and not args.harmful:
        logger.error("Must specify either --benign or --harmful (or both)")
        return

    # Optional override of the (paper-fixed) depth schedule for custom experiments.
    if args.depths:
        global DEPTHS
        DEPTHS = [int(tok) for tok in args.depths.replace(",", " ").split()]
        logger.info("Overriding depth schedule with %d custom depths: %s", len(DEPTHS), DEPTHS)

    # Per-model configuration from the registry (single source of truth).
    spec = get_model(args.model)
    safety_tokens = _unescape(args.safety_tokens)
    if safety_tokens is None:
        safety_tokens = spec.probe_safety_tokens
    mask_tokens = _unescape(args.mask_tokens)  # None -> no masking (paper default)
    hook_position = args.hook_position or spec.hook_position
    prompt_completion = spec.generation_prompt_completion

    # CUDA setup before any CUDA op; CUDA_VISIBLE_DEVICES pins the chosen GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    set_random_seeds(args.seed)
    logger.info("Set random seed to %d", args.seed)

    model_dtype = _resolve_dtype(args.dtype)
    logger.info("Loading model: %s with %s precision", args.model, args.dtype)
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=model_dtype, device="cuda:0", use_flash_attention=False
    )
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    target_layers = parse_layer_list(args.layers, model=model, hook_position=hook_position)

    logger.info("Collecting hidden states for hook indices: %s (0-based)", target_layers)
    logger.info("Hook position: %s", hook_position)
    logger.info("Safety tokens: %r", safety_tokens)
    logger.info("Mask tokens: %r", mask_tokens)
    logger.info("Memory strategy: %s", "Gradual cache" if args.gradual_cache else "Full forward pass")
    logger.info("Max tokens: %d | depths: %s", args.max_tokens, DEPTHS)
    logger.info("Collect all tokens: %s | dtype: %s | device: %s",
                args.collect_all_tokens, model_dtype, device)
    logger.info("Save interval: %d samples", args.save_interval)
    if args.no_empty:
        logger.info("No-empty mode: skipping 'empty' prefix in save/merge")
    logger.info(gpu_memory_summary())

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(safety_tokens=safety_tokens, mask_tokens=mask_tokens,
                  hook_position=hook_position, prompt_completion=prompt_completion)

    total_harmful = total_benign = 0
    if args.harmful:
        harmful_file = args.harmful_file or (
            f"data/train/probe/harmful/wildjailbreak/{args.split}_responses.jsonl"
        )
        logger.info("Loading harmful responses from %s", harmful_file)
        harmful = load_harmful_responses(harmful_file)
        logger.info("Loaded %d harmful responses", len(harmful))
        harmful = _select_eighth(harmful, args.index)
        total_harmful = process_responses(
            model, tokenizer, harmful, target_layers, args, output_dir,
            "harmful", device, model_dtype, **common,
        )

    if args.benign:
        benign_files = args.benign_files or [
            f"data/train/probe/benign/wildchat1m/{args.split}_responses.jsonl",
            f"data/train/probe/benign/wildjailbreak/{args.split}_responses.jsonl",
        ]
        benign: List[Dict] = []
        for path in benign_files:
            records = read_jsonl(path)
            logger.info("Loaded %d benign responses from %s", len(records), path)
            benign.extend(records)
        logger.info("Concatenated %d total benign responses", len(benign))
        benign = _select_eighth(benign, args.index)
        total_benign = process_responses(
            model, tokenizer, benign, target_layers, args, output_dir,
            "benign", device, model_dtype, **common,
        )

    logger.info("=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("Model: %s | hook: %s | strategy: %s", args.model, hook_position,
                "Gradual cache" if args.gradual_cache else "Full forward pass")
    logger.info("Hook indices: %d | max tokens: %d | dtype: %s",
                len(target_layers), args.max_tokens, model_dtype)
    logger.info("Data index: %d (eighth %d/8) | output: %s", args.index, args.index + 1, output_dir)
    if args.harmful:
        logger.info("Total harmful samples: %d", total_harmful)
    if args.benign:
        logger.info("Total benign samples: %d", total_benign)
    logger.info("Final memory usage: %s", gpu_memory_summary())
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
