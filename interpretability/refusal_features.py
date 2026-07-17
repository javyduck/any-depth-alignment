"""Refusal-linked cross-layer transcoder (CLT) features on Gemma-2-2b-it.

These ten GemmaScope transcoder features (identified from top activations on
directly-harmful, refusal-eliciting prompts) are the "super-neurons" whose
activations the Appendix-C analysis tracks. They spike on the injected assistant
header — especially the `model` token and the following newline — and are near
zero across a harmful continuation, which is the mechanistic signature of a
Safety Token re-triggering refusal.

Each entry is ``(layer, feature_index)`` into the GemmaScope
``gemma-scope-2b-pt-transcoders`` (width_16k, lowest-L0) transcoder for that layer,
loaded through the ``"gemma"`` preset of circuit-tracer's ``ReplacementModel``.
"""

from __future__ import annotations

from typing import List, Tuple

# (layer, feature_index)
REFUSAL_FEATURES: List[Tuple[int, int]] = [
    (18, 12640),  # activates on apology phrases ("sorry"/"apologize")
    (19, 9694),
    (20, 5315),
    (21, 16351),
    (22, 5394),   # activates on first-person "I" (refusal templates: "I cannot"/"I am sorry")
    (23, 13675),
    (24, 7179),
    (25, 6986),
    (25, 8178),
    (25, 11109),
]

# Convenience aliases used in the two highlighted per-feature panels (Fig. clt_activation).
APOLOGY_FEATURE = (18, 12640)
FIRST_PERSON_FEATURE = (22, 5394)
