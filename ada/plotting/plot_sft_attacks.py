"""Refusal rate vs. supervised fine-tuning (SFT) step.

Reconstructs ``figures/sft_all_harmful_datasets_{model}.pdf`` from
``final-sft.ipynb``. The experiment fine-tunes a model with a LoRA adapter for a
growing number of steps and asks: does the defense survive continued training?
Two fine-tuning regimes are compared side by side:

* **Benign SFT**   - adapter trained on benign data (a proxy for accidental
  catastrophic forgetting of safety).
* **Adversarial SFT** - adapter trained on harmful data (an explicit attack).

For each regime we plot the refusal rate (higher is better) averaged over four
harmful datasets (AdvBench, JailbreakBench, HExPHI, StrongReject) at two
generation depths (100 solid, 1000 dashed), for the methods:

* **Base Model**       - undefended generation (``mode_empty``).
* **Deep Alignment**   - the Qi et al. deep-alignment checkpoint baseline.
* **ADA (RK)**         - Rethinking (``mode_add_safetytoken``).
* **ADA (LP) Enable**  - the linear probe with the SFT adapter's Safety-Token
  behaviour left *enabled* (the plain adapter logs).
* **ADA (LP) Disable** - the linear probe with the adapter's Safety-Token
  behaviour *disabled* (the ``-disable_safetytoken`` logs); this isolates the
  probe's own robustness from any residual adapter alignment.

Per-file refusal curves and all on-disk paths come from
:mod:`ada.plotting._common`; per-model details (probe layer, Safety-Token span,
slug, and the matching deep-alignment checkpoint) come from :mod:`ada.registry` /
``configs/models.yaml`` rather than being hard-coded.

Run as::

    python -m ada.plotting.plot_sft_attacks --model llama --output-dir figures
    python -m ada.plotting.plot_sft_attacks --model gemma --data-root /path/to/logs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from .. import registry
from ..utils.io import read_json
from ._common import (
    ensure_output_dir,
    find_generation_log,
    find_probe_log,
    probe_layer,
    probe_safety_tokens,
)

# --------------------------------------------------------------------------- #
# Experiment constants (from final-sft.ipynb)
# --------------------------------------------------------------------------- #
# ``--model`` maps to the base checkpoint that was fine-tuned; the registry then
# supplies its probe layer / Safety-Token span, and configs/models.yaml supplies
# the matching deep-alignment baseline.
MODEL_CHOICES = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "gemma": "google/gemma-2-9b-it",
}

DEFAULT_DATASETS = ["advbench", "jailbreakbench", "hexphi", "strongreject"]
SFT_STEPS = [0, 10, 20, 50, 100, 200, 500, 1000]
TARGET_DEPTHS = [100, 1000]

# The two SFT regimes -> the {adapter_type} path component -> subplot title.
ADAPTER_TYPES = [("benign", "Benign SFT"), ("harmful", "Adversarial SFT")]

# Compact = the main-text figure (single ADA-LP); full = the appendix figure with
# the LoRA-adapter Enable/Disable ablation on the Safety-Token forward.
METHOD_CONFIGS_COMPACT = {
    "Base Model": {"color": "#4C78A8", "marker": "o"},
    "Deep Alignment": {"color": "#17AEAF", "marker": "s"},
    "ADA (RK)": {"color": "#F26B21", "marker": "*"},
    "ADA (LP)": {"color": "#D62F2F", "marker": "v"},
}
METHOD_CONFIGS_FULL = {
    "Base Model": {"color": "#4C78A8", "marker": "o"},
    "Deep Alignment": {"color": "#17AEAF", "marker": "s"},
    "ADA (RK)": {"color": "#F26B21", "marker": "*"},
    "ADA (LP) Enable": {"color": "#E75480", "marker": "h"},
    "ADA (LP) Disable": {"color": "#D62F2F", "marker": "v"},
}


def _deep_alignment_hf_id(base_hf_id: str) -> Optional[str]:
    """HF id of the deep-alignment checkpoint fine-tuned from ``base_hf_id``.

    Read from the ``deep_alignment_baselines`` block of configs/models.yaml so no
    checkpoint name is hard-coded here.
    """
    yaml_path = Path(registry.__file__).resolve().parent.parent / "configs" / "models.yaml"
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for entry in raw.get("deep_alignment_baselines", []):
        if entry.get("base") == base_hf_id:
            return entry["hf_id"]
    return None


def sft_log_path(
    method: str,
    dataset: str,
    hf_id: str,
    deep_hf_id: Optional[str],
    adapter_type: str,
    step: int,
    data_root: Path,
) -> Optional[Path]:
    """Locate the per-depth log for ``(method, dataset, adapter_type, step)``.

    Step 0 = the un-fine-tuned base model (no ``-adapter-`` component and, for
    ADA-LP, no ``-disable_safetytoken`` suffix). Returns ``None`` when a method
    is not applicable (e.g. no deep-alignment checkpoint for this base).
    """
    gen_base = data_root / "vllm_generation_logs"
    probe_base = data_root / "logs"
    # Adapter kwargs are only added for step > 0.
    adapter_kw = {} if step == 0 else {"adapter_type": adapter_type, "step": step}

    if method == "Base Model":
        return find_generation_log("harmful", dataset, hf_id, "empty",
                                   base_dir=gen_base, **adapter_kw)
    if method == "Deep Alignment":
        if deep_hf_id is None:
            return None
        return find_generation_log("harmful", dataset, deep_hf_id, "empty",
                                   base_dir=gen_base, **adapter_kw)
    if method == "ADA (RK)":
        return find_generation_log("harmful", dataset, hf_id, "add_safetytoken",
                                   base_dir=gen_base, **adapter_kw)
    if method in ("ADA (LP) Enable", "ADA (LP)"):
        return find_probe_log("harmful", dataset, hf_id, probe_safety_tokens(hf_id),
                              probe_layer(hf_id), base_dir=probe_base, **adapter_kw)
    if method == "ADA (LP) Disable":
        # The "disabled" ablation only exists once an adapter is present.
        disable_kw = ({} if step == 0
                      else {"adapter_type": adapter_type, "step": step,
                            "disable_safetytoken": True})
        return find_probe_log("harmful", dataset, hf_id, probe_safety_tokens(hf_id),
                              probe_layer(hf_id), base_dir=probe_base, **disable_kw)
    raise ValueError(f"Unknown method: {method}")


def _refusal_rate_at_depth(path: Path, depth: int, min_count: int = 10) -> Optional[float]:
    """Refusal rate at a fixed depth using the SFT-attack notebook's denominator.

    Matches ``final-sft.ipynb`` ``parse_log_at_depth``: refusals-at-``depth`` over
    the log's global ``total_responses``. Deep depths that fewer than ``min_count``
    responses ever reach are returned as ``None`` (dropped from the averaged curve),
    exactly as the source guard did, so a couple of long outliers can't dominate.
    """
    data = read_json(path)
    total = data.get("total_responses") or 0
    if not total:
        return None
    def _flag(v):  # is_refusal may be a bool or the string "True"/"False"
        return v is True or (isinstance(v, str) and v.strip().lower() == "true")
    at_depth = [e for e in data.get("detailed_logs", []) if int(e.get("depth", -1)) == depth]
    if len(at_depth) < min_count:
        return None
    n = sum(1 for e in at_depth if _flag(e.get("is_refusal", False)))
    return n / total


def get_sft_data(
    hf_id: str,
    deep_hf_id: Optional[str],
    adapter_type: str,
    target_depth: int,
    datasets: List[str],
    data_root: Path,
    methods: Dict[str, dict],
) -> Dict[str, Dict[int, float]]:
    """Refusal rate per method per SFT step, averaged (unweighted) over datasets.

    For each (method, step) we read the refusal curve of every dataset, take its
    rate at ``target_depth``, and average across the datasets that produced data
    (mirroring the notebook's simple unweighted mean; a step is skipped when no
    dataset has a value at that depth).
    """
    methods_data: Dict[str, Dict[int, float]] = {}
    for method in methods:
        step_rates: Dict[int, float] = {}
        for step in SFT_STEPS:
            per_dataset: List[float] = []
            for dataset in datasets:
                path = sft_log_path(method, dataset, hf_id, deep_hf_id,
                                    adapter_type, step, data_root)
                if path is None or not path.exists():
                    continue
                rate = _refusal_rate_at_depth(path, target_depth)
                if rate is not None:
                    per_dataset.append(rate)
            if per_dataset:
                step_rates[step] = float(np.mean(per_dataset))
        if step_rates:
            methods_data[method] = step_rates
    return methods_data


def plot_regime(
    hf_id: str,
    deep_hf_id: Optional[str],
    adapter_type: str,
    title: str,
    ax,
    datasets: List[str],
    data_root: Path,
    show_ylabel: bool,
    methods: Dict[str, dict],
) -> None:
    x_positions = np.arange(len(SFT_STEPS))
    for target_depth in TARGET_DEPTHS:
        linestyle = "-" if target_depth == 100 else "--"
        methods_data = get_sft_data(hf_id, deep_hf_id, adapter_type,
                                    target_depth, datasets, data_root, methods)
        for method, step_rates in methods_data.items():
            idx = [i for i, s in enumerate(SFT_STEPS) if s in step_rates]
            ys = [step_rates[SFT_STEPS[i]] for i in idx]
            if not ys:
                continue
            mconf = methods[method]
            ax.plot(
                [x_positions[i] for i in idx], ys,
                color=mconf["color"], marker=mconf["marker"], linestyle=linestyle,
                linewidth=2.5, markersize=10, alpha=0.9, label=method,
            )

    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_xlabel("SFT Steps", fontsize=22, weight="bold")
    if show_ylabel:
        ax.set_ylabel(r"Refusal Rate (%, $\uparrow$ is better)", fontsize=24, weight="bold")
    ax.set_title(title, fontsize=24, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(s) for s in SFT_STEPS])
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))


def make_figure(model_key: str, datasets: List[str], data_root: Path, output_dir: Path,
                methods: Dict[str, dict], suffix: str = "") -> None:
    hf_id = MODEL_CHOICES[model_key]
    deep_hf_id = _deep_alignment_hf_id(hf_id)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    for ax, (adapter_type, title) in zip(axes, ADAPTER_TYPES):
        plot_regime(hf_id, deep_hf_id, adapter_type, title, ax, datasets, data_root,
                    show_ylabel=(adapter_type == "benign"), methods=methods)

    # Combined legend: one entry per method (colour) + two line-style entries
    # explaining the depth encoding.
    method_handles, method_labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in method_labels:
                method_handles.append(handle)
                method_labels.append(label)
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2),
    ]
    axes[1].legend(
        method_handles + style_handles,
        method_labels + ["Depth 100", "Depth 1000"],
        fontsize=16, loc="lower right", bbox_to_anchor=(0.98, 0.05), ncol=1,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    out_dir = ensure_output_dir(output_dir)
    out_path = out_dir / f"sft_all_harmful_datasets_{model_key}{suffix}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--model", choices=sorted(MODEL_CHOICES), default="llama",
        help="Base model that was fine-tuned (selects probe layer / Safety Tokens "
        "and the deep-alignment baseline from the registry).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="Harmful datasets to average the refusal rate over.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("."),
        help="Directory containing vllm_generation_logs/ and logs/.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--full-only", action="store_true",
                        help="Emit only the _full (Enable/Disable) appendix figure.")
    parser.add_argument("--compact-only", action="store_true",
                        help="Emit only the compact main-text figure.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.full_only:  # main-text figure: single ADA-LP (4 methods)
        make_figure(args.model, args.datasets, args.data_root, args.output_dir,
                    methods=METHOD_CONFIGS_COMPACT, suffix="")
    if not args.compact_only:  # appendix figure: Enable vs Disable ablation (5 methods)
        make_figure(args.model, args.datasets, args.data_root, args.output_dir,
                    methods=METHOD_CONFIGS_FULL, suffix="_full")


if __name__ == "__main__":
    main()
