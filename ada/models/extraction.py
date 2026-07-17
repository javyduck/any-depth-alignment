"""Forward-hook hidden-state extraction at configurable positions.

:class:`HookHiddenStateCollector` taps a chosen sub-module of every transformer
block (``input_layernorm`` by default, but also ``self_attn``, ``mlp``, or the
various layernorms for the hook-position ablation) and reads the hidden state at
specified token positions during a single forward pass. ADA reads the position of
the injected Safety Token (the last token of the injected header).

Hook indexing is 0-based over blocks, so "layer 15" means the 15th block's hook,
matching the ``probe_layer`` values in the model registry.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


class HookHiddenStateCollector:
    """Collect hidden states at a given hook position for chosen token positions."""

    def __init__(self, model, hook_position: str, target_layers: List[int]):
        self.model = model
        self.hook_position = hook_position
        self.target_layers = set(target_layers)
        self.collected: Dict[int, torch.Tensor] = {}
        self.token_positions: "List[int] | None" = None
        self._handles: list = []

    # -- module discovery ---------------------------------------------------- #
    def _base_model(self):
        base = self.model
        if hasattr(base, "base_model"):
            base = base.base_model
        while hasattr(base, "model"):
            base = base.model
        if hasattr(base, "language_model"):
            base = base.language_model
        return base

    def _find_module_in_layer(self, layer):
        pos = self.hook_position
        for name, module in layer.named_modules():
            if name == pos:
                return module
        for name, module in layer.named_modules():
            if name.endswith(pos):
                return module
        candidates = [(n, m) for n, m in layer.named_modules() if pos in n]
        if candidates:
            candidates.sort(key=lambda x: (len(x[0]), x[0]))
            if len(candidates) > 1:
                logger.warning("Multiple matches for hook '%s'; using '%s'", pos, candidates[0][0])
            return candidates[0][1]
        raise ValueError(f"No module found for hook position '{self.hook_position}' in a block")

    def _target_modules(self) -> List[Tuple[int, object]]:
        layers = list(self._base_model().layers)
        return [(i, self._find_module_in_layer(layers[i])) for i in sorted(self.target_layers) if i < len(layers)]

    # -- hooks --------------------------------------------------------------- #
    def _hook_fn(self, layer_idx: int) -> Callable:
        def hook(_module, _inp, output):
            if self.token_positions is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() != 3:
                return
            seq_len = hidden.size(1)
            picked = [hidden[:, p, :] for p in self.token_positions if -seq_len <= p < 0]
            if picked:
                self.collected[layer_idx] = torch.stack(picked, dim=1).detach().cpu()

        return hook

    def register(self):
        for layer_idx, module in self._target_modules():
            self._handles.append(module.register_forward_hook(self._hook_fn(layer_idx)))

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def collect(self, token_positions: List[int], forward_fn: Callable):
        """Run ``forward_fn`` while capturing hidden states at ``token_positions``.

        ``token_positions`` are negative indices (e.g. ``[-1]`` for the last,
        injected Safety Token). Returns ``(states_by_layer, forward_result)``.
        """
        self.token_positions = token_positions
        self.collected.clear()
        self.register()
        try:
            result = forward_fn()
        finally:
            self.remove()
            self.token_positions = None
        return dict(self.collected), result


def count_hook_positions(model, hook_position: str) -> int:
    """Number of transformer blocks that expose ``hook_position``."""
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model
    while hasattr(base, "model"):
        base = base.model
    if hasattr(base, "language_model"):
        base = base.language_model
    return sum(
        any(hook_position in name for name, _ in layer.named_modules())
        for layer in base.layers
    )


def parse_layer_list(layers: str, model=None, hook_position: str = "input_layernorm") -> List[int]:
    """Parse a ``--layers`` argument.

    ``"all"`` expands to every available hook index (requires ``model``); a
    comma/space separated string like ``"13,15,19"`` is parsed literally.
    """
    layers = str(layers).strip()
    if layers.lower() == "all":
        if model is None:
            raise ValueError("layers='all' requires a model to count hook positions")
        return list(range(count_hook_positions(model, hook_position)))
    return [int(tok) for tok in layers.replace(",", " ").split()]
