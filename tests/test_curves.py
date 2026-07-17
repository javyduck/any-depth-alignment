"""Refusal-curve accounting and release-layout path resolution."""

from __future__ import annotations

from pathlib import Path

from ada.plotting._common import cumulative_refusal_curve, parse_refusal_curve
from ada.plotting.plot_e3_attacks import base_response_path, deep_alignment_response_path
from ada.plotting.plot_e5_benign import _baseline_benign_response_path


def _rows(n_instances, refusal_depths):
    """n instances checked at depths {25,50,75}; refusal_depths[inst] = set of refusing depths."""
    rows = []
    for inst in range(n_instances):
        for d in (25, 50, 75):
            rows.append({"instance": inst, "depth": d,
                         "is_refusal": d in refusal_depths.get(inst, set())})
    return rows


def test_parse_refusal_curve_independent_per_depth(make_log):
    # 20 instances; at depth 25 half refuse, at depth 50 none, at depth 75 all.
    refusals = {i: ({25} if i < 10 else set()) | ({75}) for i in range(20)}
    log = make_log(_rows(20, refusals), total=20)
    curve = parse_refusal_curve(log)
    assert curve[25] == 0.5
    assert curve[50] == 0.0
    assert curve[75] == 1.0


def test_parse_refusal_curve_drops_sparse_depths(make_log):
    # Depth 200 seen by only 3 instances (< MIN_COUNT=10) -> dropped.
    rows = _rows(15, {})
    rows += [{"instance": i, "depth": 200, "is_refusal": True} for i in range(3)]
    curve = parse_refusal_curve(make_log(rows, total=15))
    assert 25 in curve and 200 not in curve


def test_cumulative_refusal_curve_monotonic(make_log):
    # inst0 refuses first at 50, inst1 at 25, inst2 never.
    refusals = {0: {50}, 1: {25}}
    log = make_log(_rows(3, refusals), total=3)
    cum = cumulative_refusal_curve(log)
    assert cum[25] == 1 / 3          # only inst1 by depth 25
    assert cum[50] == 2 / 3          # inst0 + inst1 by depth 50
    assert cum[75] == 2 / 3          # inst2 never refuses
    depths = sorted(cum)
    assert all(cum[a] <= cum[b] for a, b in zip(depths, depths[1:]))  # non-decreasing


def test_e3_base_path_prefers_release_layout():
    p = base_response_path("advbench", "gcg", "meta-llama/Llama-2-7b-chat-hf")
    # default (nothing on disk) resolves to the release location, not the source tree
    assert Path("data/eval/attacks") in p.parents or str(p).startswith("data/eval/attacks")


def test_e3_deep_alignment_path_none_when_no_checkpoint():
    # A model without a deep-alignment baseline yields None.
    assert deep_alignment_response_path("advbench", "gcg", "openai/gpt-oss-120b") is None


def test_e5_baseline_prefers_over_refusal():
    p = _baseline_benign_response_path(Path("."), "mmlu", "google_gemma-2-9b-it")
    assert "data/eval/over_refusal" in str(p)
