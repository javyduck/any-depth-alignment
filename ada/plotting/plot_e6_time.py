"""E6 - Inference-cost figure: latency vs. context length.

Reconstructs ``figures/time.png`` from ``final-time.ipynb``. The notebook
hard-coded the per-length latencies; here we instead *read* them from the
timing table produced by :mod:`ada.timing.benchmark` +
:mod:`ada.timing.make_table` (``-> timing_results_table.csv``), so the figure
stays in sync with the measured numbers.

Three curves are drawn, all as a function of the prompt/context length:

* **Generate Next Token w/ KV cache** - the ``next_token`` row of a regular LLM
  (the marginal cost of one more decoded token; ADA-RK/base generation cost).
* **Forward Three Safety Tokens w/ KV cache** - the ``forward_three_tokens`` row
  of the same LLM (ADA-LP: one KV-cached forward over the injected Safety-Token
  span before reading a hidden state).
* **Guardrail Forward** - the ``forward`` row of an external guardrail, which must
  re-encode the *entire* context (no KV cache), hence the linear blow-up.

The timing-table cells are formatted ``"<mean_ms> ± <std_ms> (<mem_mb>)"`` (or
``"OOM"``); we parse mean/std and shade ±1 std.

Run as::

    python -m ada.plotting.plot_e6_time \
        --csv timing_results_table.csv --output-dir figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The regular LLM whose next-token / three-token forward costs stand in for
# "base generation" and "ADA-LP", and the guardrail whose full forward is the
# comparison baseline. These are plain HF ids matched against the CSV "Model"
# column; override on the CLI to plot a different pair.
DEFAULT_REGULAR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_GUARDRAIL_MODEL = "ibm-granite/granite-guardian-3.3-8b"

# Context lengths retained for the figure. The notebook dropped the 500-token
# column (index 1) to spread the points; we keep the same selection.
SELECTED_CONTEXT_LENGTHS = [100, 1000, 2500, 5000, 10000]

_CELL_RE = re.compile(r"([-\d.]+)\s*[±+]-?\s*([-\d.]+)")


def _parse_cell(cell: object) -> Tuple[float, float]:
    """Parse a ``"mean ± std (mem)"`` timing-table cell into ``(mean, std)``.

    Returns ``(nan, nan)`` for missing / ``OOM`` / ``N/A`` cells so they drop
    out of the plotted curve.
    """
    if not isinstance(cell, str):
        return (float("nan"), float("nan"))
    match = _CELL_RE.search(cell)
    if not match:
        return (float("nan"), float("nan"))
    return (float(match.group(1)), float(match.group(2)))


def load_timing_table(csv_path: Path) -> pd.DataFrame:
    """Load the timing CSV and forward-fill the sparse ``Model`` column.

    Regular-model results occupy two rows (``next_token`` then
    ``forward_three_tokens``); only the first carries the model name, so we
    forward-fill to recover it on the second.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df["Model"] = df["Model"].ffill()
    return df


def _length_columns(df: pd.DataFrame) -> List[int]:
    """Return the integer context-length column headers, in order."""
    lengths: List[int] = []
    for col in df.columns:
        try:
            lengths.append(int(col))
        except ValueError:
            continue
    return lengths


def _series(df: pd.DataFrame, model: str, row_type: str, lengths: List[int]):
    """Return (means, stds) arrays for one (model, type) row over ``lengths``."""
    rows = df[(df["Model"] == model) & (df["Type"] == row_type)]
    if rows.empty:
        raise KeyError(
            f"No row for model={model!r} type={row_type!r} in the timing table. "
            f"Available models: {sorted(df['Model'].dropna().unique())}"
        )
    row = rows.iloc[0]
    means, stds = [], []
    for length in lengths:
        mean, std = _parse_cell(row.get(str(length)))
        means.append(mean)
        stds.append(std)
    return np.array(means), np.array(stds)


def _plot_curve(ax, x, mean, std, label: str) -> None:
    """Plot one mean curve with a ±1 std shaded band (skipping NaN points)."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    valid = ~np.isnan(mean)
    xv = np.asarray(x, dtype=float)[valid]
    mv = mean[valid]
    sv = np.nan_to_num(std[valid], nan=0.0)
    (line,) = ax.plot(xv, mv, marker="o", label=label)
    ax.fill_between(xv, mv - sv, mv + sv, alpha=0.2, color=line.get_color())


def make_figure(
    df: pd.DataFrame,
    regular_model: str,
    guardrail_model: str,
    output_path: Path,
) -> None:
    all_lengths = _length_columns(df)
    # Restrict to the lengths we want to show, preserving CSV order.
    lengths = [n for n in all_lengths if n in SELECTED_CONTEXT_LENGTHS]

    next_mean, next_std = _series(df, regular_model, "next_token", lengths)
    three_mean, three_std = _series(df, regular_model, "forward_three_tokens", lengths)
    guard_mean, guard_std = _series(df, guardrail_model, "forward", lengths)

    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_curve(ax, lengths, next_mean, next_std, "Generate Next Token w/ KV cache")
    _plot_curve(ax, lengths, three_mean, three_std, "Forward Three Safety Tokens w/ KV cache")
    _plot_curve(ax, lengths, guard_mean, guard_std, "Guardrail Forward")

    ax.set_xlabel("Context Tokens", fontsize=24)
    ax.set_ylabel("Inference Time (ms)", fontsize=24)
    ax.legend(fontsize=18)
    ax.tick_params(axis="both", labelsize=22)
    ax.set_xticks(lengths)

    # Match the notebook's y-axis: ensure a 25 ms tick and a fixed range.
    yticks = list(ax.get_yticks())
    if 25 not in yticks:
        yticks = sorted(yticks + [25])
    ax.set_yticks(yticks)
    ax.set_ylim(-5, 510)
    ax.grid(True)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("timing_results_table.csv"),
        help="Timing table produced by ada.timing.make_table.",
    )
    parser.add_argument(
        "--regular-model",
        default=DEFAULT_REGULAR_MODEL,
        help="HF id (as in the CSV 'Model' column) for the next-token / "
        "three-token-forward curves.",
    )
    parser.add_argument(
        "--guardrail-model",
        default=DEFAULT_GUARDRAIL_MODEL,
        help="HF id (as in the CSV 'Model' column) for the guardrail-forward curve.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = load_timing_table(args.csv)
    make_figure(
        df,
        regular_model=args.regular_model,
        guardrail_model=args.guardrail_model,
        output_path=args.output_dir / "time.png",
    )


if __name__ == "__main__":
    main()
