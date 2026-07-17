"""Evaluation benchmark loaders (test-time prompts, pulled live from HuggingFace).

These are the *prompt* sources for evaluation. Model responses to them are
generated once and cached under ``data/eval/`` — the benchmark prompts themselves
are always loaded fresh from the Hub so no benchmark text is redistributed.

Two groups:

* :func:`load_harmful_prompts` — harmful behaviours (AdvBench, StrongREJECT,
  JailbreakBench, HEx-PHI, WildJailbreak, SafeDecoding attackers).
* :func:`load_benign_prompts` — benign/over-refusal benchmarks (GSM8K, MATH,
  MMLU, BBH, HumanEval, SimpleQA, GPQA Diamond, XSTest, AlpacaEval).

HEx-PHI is gated (``LLM-Tuning-Safety/HEx-PHI``); loading it requires accepting
its license and an ``HF_TOKEN``. Its prompts are never shipped with this release.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

from datasets import load_dataset
from huggingface_hub import hf_hub_download


# --------------------------------------------------------------------------- #
# Harmful behaviour benchmarks
# --------------------------------------------------------------------------- #
def load_advbench() -> List[str]:
    ds = load_dataset("walledai/AdvBench", split="train")
    return [ex["prompt"] for ex in ds]


def load_strongreject() -> List[str]:
    ds = load_dataset("walledai/StrongREJECT", split="train")
    return [ex["prompt"] for ex in ds]


def load_jailbreakbench() -> List[str]:
    ds = load_dataset("JailbreakBench/JBB-Behaviors", name="behaviors", split="harmful")
    return [ex["Goal"] for ex in ds]


def load_hexphi() -> List[str]:
    """Load HEx-PHI (categories 1, 3-11; category 2 = CSAM is excluded).

    Gated dataset — requires an accepted license and ``HF_TOKEN``.
    """
    repo, cats, prompts = "LLM-Tuning-Safety/HEx-PHI", [1, 3, 4, 5, 6, 7, 8, 9, 10, 11], []
    for c in cats:
        local_csv = hf_hub_download(repo_id=repo, filename=f"category_{c}.csv", repo_type="dataset")
        ds = load_dataset(
            "csv", data_files={"train": local_csv}, column_names=["prompt"], header=None, split="train"
        )
        prompts.extend(ds["prompt"])
    return prompts


def load_wildjailbreak(seed: int = 42, n_per_type: int = 5000) -> List[str]:
    """Sample vanilla + adversarial harmful prompts from WildJailbreak."""
    random.seed(seed)
    ds = load_dataset(
        "allenai/wildjailbreak", "train", split="train", delimiter="\t", keep_default_na=False
    )
    vanilla, adversarial = [], []
    for ex in ds:
        if ex["data_type"] == "vanilla_harmful":
            vanilla.append(ex["vanilla"])
        elif ex["data_type"] == "adversarial_harmful":
            adversarial.append(ex["adversarial"] or ex["vanilla"])
    prompts = random.sample(vanilla, min(n_per_type, len(vanilla)))
    prompts += random.sample(adversarial, min(n_per_type, len(adversarial)))
    random.shuffle(prompts)
    return prompts


def load_safedecoding_attackers(limit: int = 35) -> List[str]:
    ds = load_dataset("UWNSL/SafeDecoding-Attackers", split="train")
    return [ex["prompt"] for ex in ds if ex.get("prompt")][:limit]


# --------------------------------------------------------------------------- #
# Benign / over-refusal benchmarks
# --------------------------------------------------------------------------- #
def load_gsm8k() -> List[str]:
    ds = load_dataset("openai/gsm8k", name="main", split="test")
    return [ex["question"] for ex in ds]


def load_math() -> List[str]:
    from datasets import get_dataset_config_names

    prompts: List[str] = []
    for subset in get_dataset_config_names("EleutherAI/hendrycks_math"):
        ds = load_dataset("EleutherAI/hendrycks_math", name=subset, split="test")
        prompts.extend(ex["problem"] for ex in ds)
    return prompts


def load_mmlu() -> List[str]:
    ds = load_dataset("cais/mmlu", name="all", split="test")
    prompts, seen = [], set()
    for ex in ds:
        key = f"{ex['question']}_{'_'.join(ex['choices'])}"
        if key in seen:
            continue
        seen.add(key)
        lines = [f"Question: {ex['question']}"]
        lines += [f"{chr(65 + i)}. {c}" for i, c in enumerate(ex["choices"])]
        lines.append("Answer:")
        prompts.append("\n".join(lines))
    return prompts


def load_bbh() -> List[str]:
    from datasets import get_dataset_config_names

    prompts: List[str] = []
    for subset in get_dataset_config_names("lukaemon/bbh"):
        ds = load_dataset("lukaemon/bbh", name=subset, split="test")
        prompts.extend(ex["input"] for ex in ds)
    return list(dict.fromkeys(prompts))  # de-dup, preserve order


def load_humaneval() -> List[str]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [ex["prompt"] for ex in ds]


def load_alpaca_eval() -> List[str]:
    ds = load_dataset("tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True)
    return [ex["instruction"] for ex in ds]


def load_simpleqa() -> List[str]:
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    return [ex["problem"] for ex in ds]


def load_gpqa() -> List[str]:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    return [ex["Question"] for ex in ds]


def load_xstest(label: str = "safe") -> List[str]:
    ds = load_dataset("walledai/XSTest", split="test")
    return [ex["prompt"] for ex in ds if ex.get("label") == label]


_HARMFUL: Dict[str, Callable[[], List[str]]] = {
    "advbench": load_advbench,
    "strongreject": load_strongreject,
    "jailbreakbench": load_jailbreakbench,
    "hexphi": load_hexphi,
    "wildjailbreak": load_wildjailbreak,
    "safedecoding": load_safedecoding_attackers,
}

_BENIGN: Dict[str, Callable[[], List[str]]] = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "mmlu": load_mmlu,
    "bbh": load_bbh,
    "humaneval": load_humaneval,
    "alpaca_eval": load_alpaca_eval,
    "simpleqa": load_simpleqa,
    "gpqa": load_gpqa,
    "xstest": load_xstest,
    "xstest_unsafe": lambda: load_xstest(label="unsafe"),
}


def load_prompts_from_file(path: "str | Path") -> List[str]:
    """Load a custom prompt set from a local file — test a NEW dataset with no code edit.

    Supported formats (by extension):
      * ``.txt``   — one prompt per line (blank lines skipped)
      * ``.csv``   — a ``prompt`` column if present, else the first column
      * ``.jsonl`` — ``{"prompt": ...}`` per line, or a chat record
                     ``{"messages": [{"role": "user", ...}, ...]}``
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if suffix == ".jsonl":
        prompts: List[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "prompt" in rec:
                prompts.append(rec["prompt"])
            elif "messages" in rec:
                prompts.append(next(m["content"] for m in rec["messages"] if m["role"] == "user"))
            else:
                raise ValueError(f"{p}: JSONL record needs a 'prompt' or 'messages' field")
        return prompts
    if suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return []
        header = [c.strip().lower() for c in rows[0]]
        if "prompt" in header:
            i = header.index("prompt")
            return [r[i] for r in rows[1:] if r and r[i].strip()]
        return [r[0] for r in rows if r and r[0].strip()]  # headerless: first column
    raise ValueError(f"Unsupported prompt file '{p}' (use .txt / .csv / .jsonl)")


def load_harmful_prompts(name: str) -> List[str]:
    """Harmful prompts by benchmark name, or from a local file (see load_prompts_from_file)."""
    if Path(name).is_file():
        return load_prompts_from_file(name)
    key = name.lower()
    if key not in _HARMFUL:
        raise ValueError(
            f"Unknown harmful benchmark '{name}'. Known: {sorted(_HARMFUL)}, "
            "or pass a path to a .txt/.csv/.jsonl prompt file."
        )
    return _HARMFUL[key]()


def load_benign_prompts(name: str) -> List[str]:
    """Benign prompts by benchmark name, or from a local file (see load_prompts_from_file)."""
    if Path(name).is_file():
        return load_prompts_from_file(name)
    key = name.lower()
    if key not in _BENIGN:
        raise ValueError(
            f"Unknown benign benchmark '{name}'. Known: {sorted(_BENIGN)}, "
            "or pass a path to a .txt/.csv/.jsonl prompt file."
        )
    return _BENIGN[key]()
