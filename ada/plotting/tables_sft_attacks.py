"""Appendix table: adversarial-attack ASR under SFT, ADA-LP Enable vs Disable.

Reproduces the ``sft_asr_ablation`` table (Appendix): for each model, SFT regime
(benign / adversarial), and checkpoint step, the attack success rate of ADA-LP on
GCG / AutoDAN / PAIR / TAP, comparing the LoRA adapter **Enabled** vs **Disabled**
on the Safety-Token forward pass.

Reads the per-checkpoint ADA-LP attack logs produced by
``scripts/sft_eval.sh`` (``ada.probe.evaluate --attack ... --adapter ...``):

    logs/harmful/{dataset}_{attack}/{model_slug}-{adapter_type}-adapter-{step}[-disable_safetytoken]/
        {safety}/mask_token_none/hook_input_layernorm/seed_42/logistic/probe-layers{L}/depth_25_maxdepth_3000.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ..registry import get_model
from ..utils.naming import slugify_model, slugify_safety_tokens
from ._common import (
    DEFAULT_DEPTH_STEP as DEPTH,
    DEFAULT_MAX_DEPTH as MAX_DEPTH,
    asr_from_generation_log,
    attack_set_total,
)

ATTACKS = ["gcg", "autodan", "pair", "tap"]


def _ada_lp_adapter_log(
    dataset: str, attack: str, model: str, adapter_type: str, step: int, disable: bool
) -> Path:
    spec = get_model(model)
    suffix = f"-{adapter_type}-adapter-{step}" + ("-disable_safetytoken" if disable else "")
    return (
        Path("logs") / "harmful" / f"{dataset.lower()}_{attack.lower()}"
        / f"{slugify_model(model)}{suffix}"
        / slugify_safety_tokens(spec.probe_safety_tokens)
        / "mask_token_none" / "hook_input_layernorm" / "seed_42" / "logistic"
        / f"probe-layers{spec.probe_layer}"
        / f"depth_{DEPTH}_maxdepth_{MAX_DEPTH}.json"
    )


def _asr(dataset: str, attack: str, model: str, adapter_type: str, step: int, disable: bool) -> Optional[float]:
    path = _ada_lp_adapter_log(dataset, attack, model, adapter_type, step, disable)
    if not path.exists():
        return None
    return 100.0 * asr_from_generation_log(path, attack_set_total(dataset))


def print_table(models: List[str], regimes: List[str], steps: List[int], dataset: str) -> None:
    header = f"{'Regime':<12}{'Step':>6}{'Model':>28}   " + "".join(
        f"{a.upper():>8}" for a in ATTACKS
    )
    for label, disable in [("ADA-LP Enable", False), ("ADA-LP Disable", True)]:
        print(f"\n=== {label} | dataset={dataset} ===")
        print(header)
        for regime in regimes:
            for step in steps:
                for model in models:
                    cells = []
                    for atk in ATTACKS:
                        v = _asr(dataset, atk, model, regime, step, disable)
                        cells.append(f"{v:>7.1f}%" if v is not None else f"{'--':>8}")
                    print(f"{regime:<12}{step:>6}{model.split('/')[-1]:>28}   " + "".join(cells))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+",
                   default=["meta-llama/Llama-2-7b-chat-hf", "google/gemma-2-9b-it"])
    p.add_argument("--regimes", nargs="+", default=["benign", "harmful"])
    p.add_argument("--steps", nargs="+", type=int, default=[100, 200, 500, 1000])
    p.add_argument("--datasets", nargs="+", default=["advbench", "jailbreakbench"])
    args = p.parse_args()
    for dataset in args.datasets:
        print_table(args.models, args.regimes, args.steps, dataset)


if __name__ == "__main__":
    main()
