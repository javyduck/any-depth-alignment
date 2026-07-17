"""Smoke: every ``python -m ada.*`` entrypoint imports and wires argparse cleanly.

Marked ``smoke`` because each spawns a subprocess with heavy imports (torch/vLLM).
Run with:  ``pytest -m smoke tests/test_cli_help.py``
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

ENTRYPOINTS = [
    "ada.probe.collect", "ada.probe.train", "ada.probe.evaluate",
    "ada.rethink.generate", "ada.rethink.claude",
    "ada.guardrails.evaluate",
    "ada.attacks.sft", "ada.attacks.extract",
    "ada.timing.benchmark", "ada.timing.make_table",
    "ada.datagen.hexphi_reference", "ada.datagen.gen_harmful_gpt",
    "ada.datagen.gen_wildchat1m", "ada.datagen.gen_benign_wildjailbreak",
    "ada.datagen.gen_benign_responses", "ada.datagen.continue_wildjailbreak",
    "ada.datagen.merge_benign_corpora",
    "ada.plotting.plot_e1_probe", "ada.plotting.plot_e1_tsne", "ada.plotting.tables_e1",
    "ada.plotting.plot_e2_prefill",
    "ada.plotting.plot_e3_attacks", "ada.plotting.tables_e3",
    "ada.plotting.plot_e4_sft", "ada.plotting.tables_e4",
    "ada.plotting.plot_e5_benign", "ada.plotting.plot_e6_time",
    "ada.plotting.tables_ablation",
    "ada.serving.server",
]


@pytest.mark.smoke
@pytest.mark.parametrize("module", ENTRYPOINTS)
def test_entrypoint_help(module):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONWARNINGS": "ignore"}
    r = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 0, f"{module} --help failed:\n{r.stderr[-800:]}"
    assert "usage" in (r.stdout + r.stderr).lower()
