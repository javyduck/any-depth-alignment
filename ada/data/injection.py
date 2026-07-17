"""Safety-Token injection — the core mechanism of Any-Depth Alignment.

At the heart of ADA is a single construction: take a partial assistant response
of some *generation depth* ``d`` and re-inject the assistant header (the "Safety
Tokens") right after it. Reading the model's state at that injected header (ADA-LP)
or letting it generate a short lookahead from there (ADA-RK) re-triggers the
model's innate shallow-refusal alignment at *any* depth.

This module builds those injected prompts at the token level, avoiding a
decode→encode round trip so the exact token boundaries are preserved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ProbeSample:
    """One depth-truncated, header-injected example for the probe corpus."""

    input_ids: torch.Tensor  # user_prefix + assistant[:depth] + safety_tokens
    label: int  # 1 = harmful, 0 = benign
    depth: int
    dataset: str


class ChatTemplateCache:
    """Caches tokenized user prefixes and the Safety-Token span.

    Applying a chat template is not free; when sweeping thousands of continuations
    over many depths, the user prefix and Safety-Token encodings are recomputed
    constantly. This cache eliminates that overhead.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._user_prefix: Dict[str, torch.Tensor] = {}
        self._safety_tokens: Optional[torch.Tensor] = None

    def user_prefix_tokens(self, user_prompt: str) -> torch.Tensor:
        """Return the tokenized, generation-prompt-terminated user turn."""
        if user_prompt not in self._user_prefix:
            messages = [{"role": "user", "content": user_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            # Some templates do not end in whitespace; a trailing space keeps the
            # assistant continuation from being merged into the final header token.
            if not (text.endswith("\n") or text.endswith(" ")):
                text += " "
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            self._user_prefix[user_prompt] = torch.tensor(ids)
        return self._user_prefix[user_prompt]

    def safety_tokens(self, safety_tokens: str) -> torch.Tensor:
        if self._safety_tokens is None:
            ids = self.tokenizer.encode(safety_tokens, add_special_tokens=False)
            self._safety_tokens = torch.tensor(ids)
        return self._safety_tokens


def build_injected_prompt(
    cache: ChatTemplateCache,
    user_prompt: str,
    assistant_tokens: torch.Tensor,
    depth: int,
    safety_tokens: str,
) -> torch.Tensor:
    """Compose ``user_prefix + assistant[:depth] + safety_tokens`` at token level."""
    return torch.cat(
        [
            cache.user_prefix_tokens(user_prompt),
            assistant_tokens[:depth],
            cache.safety_tokens(safety_tokens),
        ]
    )


def sample_balanced_batch(
    harmful: List[Dict],
    benign: Dict[str, List[Dict]],
    batch_size: int,
) -> List[Tuple[Dict, int, str]]:
    """Sample ``batch_size`` records, half harmful and half spread over benign sets."""
    half = batch_size // 2
    out: List[Tuple[Dict, int, str]] = [
        (r, 1, "harmful") for r in random.sample(harmful, min(half, len(harmful)))
    ]

    names = list(benign)
    if names:
        per, extra = divmod(half, len(names))
        for i, name in enumerate(names):
            pool = benign[name]
            if not pool:
                continue
            k = per + (1 if i < extra else 0)
            out.extend((r, 0, name) for r in random.sample(pool, min(k, len(pool))))

    random.shuffle(out)
    return out


def collate(batch: List[ProbeSample]) -> Dict[str, torch.Tensor]:
    """Right-pad a batch of :class:`ProbeSample` into model inputs."""
    input_ids = [s.input_ids for s in batch]
    max_len = max(len(ids) for ids in input_ids)
    padded = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    mask = torch.zeros(len(input_ids), max_len, dtype=torch.bool)
    for i, ids in enumerate(input_ids):
        padded[i, : len(ids)] = ids
        mask[i, : len(ids)] = True
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "labels": torch.tensor([s.label for s in batch]),
        "depths": [s.depth for s in batch],
    }
