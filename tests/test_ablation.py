"""Appendix ablations: checkpoint-frequency subsampling + temperature over-refusal."""

from __future__ import annotations

from ada.plotting.tables_ablation import (
    _checked,
    _method_log_path,
    asr_at_frequency,
    over_refusal_at_temperature,
)


def test_method_log_path_harmful_ada_lp():
    p = str(_method_log_path("ada_lp", "advbench", "google/gemma-2-9b-it", "gcg"))
    assert p.startswith("logs/harmful/advbench_gcg/google_gemma-2-9b-it/")
    assert "probe-layers23" in p and p.endswith("depth_25_maxdepth_3000.json")


def test_method_log_path_benign_and_temperature():
    # benign over-refusal (attack=None) -> logs/benign/<dataset> (no attack suffix)
    p = str(_method_log_path("ada_lp", "mmlu", "google/gemma-2-9b-it", None, temperature=0.5))
    assert p.startswith("logs/benign/mmlu/")
    assert p.endswith("depth_25_maxdepth_3000_temp_0.5.json")


def test_method_log_path_generation_methods():
    rk = str(_method_log_path("ada_rk", "advbench", "google/gemma-2-9b-it", "gcg"))
    assert "vllm_generation_logs/harmful/advbench_gcg" in rk and "mode_add_safetytoken" in rk
    base = str(_method_log_path("base", "mmlu", "google/gemma-2-9b-it", None))
    assert base.startswith("vllm_generation_logs/benign/mmlu/") and "mode_empty" in base


def test_checked_fixed_interval():
    assert _checked(25, 25) is True
    assert _checked(50, 25) is True
    assert _checked(30, 25) is False
    assert _checked(0, 25) is False       # depth 0 is never a checkpoint
    assert _checked(100, 100) is True
    assert _checked(50, 100) is False


def test_checked_adaptive_schedule():
    # dense (every 25) up to 100, then sparse (every 100).
    assert _checked(25, None) is True
    assert _checked(75, None) is True
    assert _checked(100, None) is True
    assert _checked(150, None) is False   # >100 and not a multiple of 100
    assert _checked(200, None) is True


def test_asr_at_frequency_subsampling(make_log):
    # 10 instances; one refuses ONLY at depth 50.
    rows = []
    for i in range(10):
        for d in (25, 50, 75, 100):
            rows.append({"instance": i, "depth": d, "is_refusal": (i == 0 and d == 50)})
    log = make_log(rows, total=10)
    # interval 25 checks depth 50 -> inst0 caught -> ASR = 9/10
    assert asr_at_frequency(log, 10, 25) == 0.9
    # interval 100 checks only depth 100 -> inst0 NOT caught -> ASR = 10/10
    assert asr_at_frequency(log, 10, 100) == 1.0


def test_asr_missing_file_returns_none(tmp_path):
    assert asr_at_frequency(tmp_path / "nope.json", 10, 25) is None


def test_over_refusal_at_temperature(make_log):
    # 4 benign instances, 1 flagged at some depth -> 25% over-refusal.
    rows = [{"instance": i, "depth": 25, "is_refusal": (i == 0)} for i in range(4)]
    log = make_log(rows, total=4)
    assert over_refusal_at_temperature(log) == 0.25
