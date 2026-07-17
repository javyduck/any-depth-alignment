"""Extend WildJailbreak into the harmful half of the ADA-LP probe corpus.

The probe corpus pairs deep *harmful* assistant prefills with deep *benign*
prefills. This script builds the harmful half from WildJailbreak by topping the
successful-jailbreak pool up to a balanced target of 11,000 examples
(5,500 vanilla-harmful + 5,500 adversarial-harmful), generated with a jailbroken
generator and screened by a ``gpt-4o`` safety judge (the same Batch-API
machinery as :mod:`ada.datagen.gen_harmful_gpt`).

Flow:

    1. Load a prior WildJailbreak generation log and count how many prompts of
       each type already yield a harmful completion.
    2. Sample additional unused prompts (with a 2x buffer for judge attrition).
    3. Generate + judge completions for them, merge with the prior log.
    4. Keep the longest harmful completion per prompt, then split into
       train/val (500 vanilla + 500 adversarial held out for val; train capped
       at 5,000 per type).

Outputs
-------
* ``{results_dir}/wildjailbreak_{gen_model}_s{new_samples}_eval.jsonl`` — merged
  per-prompt log (completions + judge labels).
* ``{out_dir}/{train,val,responses}_responses.jsonl`` where ``out_dir`` defaults
  to ``data/train/probe/harmful/wildjailbreak`` — the harmful probe corpus.

Environment
-----------
``OPENAI_API_KEY`` must be set.

Run with::

    python -m ada.datagen.continue_wildjailbreak --samples 3 --new-samples 3
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ada.datagen.gen_harmful_gpt import (
    DEFAULT_GEN_MODEL,
    DEFAULT_JUDGE_MODEL,
    GEN_SYS_PROMPT_FRESH,
    apply_min_token_override,
    build_corpus,
    build_eval_records,
    build_generation_records,
    count_tokens,
    log_corpus_stats,
    results_path,
    run_eval_batch,
    run_generation_batch,
    slugify_gen_model,
)
from ada.utils.io import read_jsonl, write_jsonl
from openai import OpenAI

logger = logging.getLogger(__name__)

MIN_TOKENS_FOR_EVAL = 100
TARGET_VANILLA = 5500
TARGET_ADVERSARIAL = 5500
VAL_PER_TYPE = 500
TRAIN_CAP_PER_TYPE = 5000


# --------------------------------------------------------------------------- #
# WildJailbreak prompt loading (full split by data_type)
# --------------------------------------------------------------------------- #
def load_wildjailbreak_full(seed: int = 42) -> Tuple[List[str], List[str]]:
    """Return (vanilla_harmful, adversarial_harmful) prompt lists (no sampling)."""
    random.seed(seed)
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/wildjailbreak", "train", split="train", delimiter="\t", keep_default_na=False
    )
    vanilla, adversarial = [], []
    for ex in ds:
        if ex["data_type"] == "vanilla_harmful":
            vanilla.append(ex["vanilla"])
        elif ex["data_type"] == "adversarial_harmful":
            adversarial.append(ex["adversarial"] if ex["adversarial"] else ex["vanilla"])
    return vanilla, adversarial


def analyze_current_results(
    records: List[dict], vanilla_prompts: Set[str], adversarial_prompts: Set[str]
) -> Tuple[Set[str], int, int]:
    """Count used prompts and successful jailbreaks by type in a prior log."""
    used_prompts: Set[str] = set()
    successful_vanilla = successful_adversarial = 0
    for rec in records:
        used_prompts.add(rec["prompt"])
        has_harmful = False
        for resp in rec.get("responses", []):
            label = resp.get("label", "")
            if count_tokens(resp.get("text", "")) < MIN_TOKENS_FOR_EVAL:
                label = "harmless"
            if label == "harmful":
                has_harmful = True
                break
        if has_harmful:
            if rec["prompt"] in adversarial_prompts:
                successful_adversarial += 1
            elif rec["prompt"] in vanilla_prompts:
                successful_vanilla += 1
    logger.info(
        "Prior log: %d records, %d vanilla + %d adversarial successful",
        len(records),
        successful_vanilla,
        successful_adversarial,
    )
    return used_prompts, successful_vanilla, successful_adversarial


def sample_additional_prompts(
    used_prompts: Set[str],
    vanilla_harmful: List[str],
    adversarial_harmful: List[str],
    current_vanilla: int,
    current_adversarial: int,
) -> Tuple[List[str], List[str]]:
    """Sample unused prompts to close the gap to target (2x attrition buffer)."""
    # The source re-seeded (via a fresh full-dataset load) immediately before this
    # sampling; mirror that so the chosen prompts are reproducible.
    random.seed(42)
    available_vanilla = [p for p in vanilla_harmful if p not in used_prompts]
    available_adversarial = [p for p in adversarial_harmful if p not in used_prompts]

    need_vanilla = max(0, TARGET_VANILLA - current_vanilla)
    need_adversarial = max(0, TARGET_ADVERSARIAL - current_adversarial)
    sample_vanilla = int(need_vanilla * 2)
    sample_adversarial = int(need_adversarial * 2)

    additional_vanilla = (
        random.sample(available_vanilla, min(sample_vanilla, len(available_vanilla)))
        if sample_vanilla > 0
        else []
    )
    additional_adversarial = (
        random.sample(available_adversarial, min(sample_adversarial, len(available_adversarial)))
        if sample_adversarial > 0
        else []
    )
    logger.info(
        "Sampling %d vanilla + %d adversarial additional prompts",
        len(additional_vanilla),
        len(additional_adversarial),
    )
    return additional_vanilla, additional_adversarial


# --------------------------------------------------------------------------- #
# Train / val split
# --------------------------------------------------------------------------- #
def split_train_val(
    filtered: List[dict], vanilla_prompts: Set[str], adversarial_prompts: Set[str]
) -> Tuple[List[dict], List[dict]]:
    """Hold out VAL_PER_TYPE of each type; cap train at TRAIN_CAP_PER_TYPE each."""
    # The source re-seeded (via a fresh full-dataset load) immediately before the
    # split; mirror that so the train/val partition is reproducible.
    random.seed(42)
    vanilla_responses, adversarial_responses = [], []
    for rec in filtered:
        prompt = rec["messages"][0]["content"]
        if prompt in adversarial_prompts:
            adversarial_responses.append(rec)
        elif prompt in vanilla_prompts:
            vanilla_responses.append(rec)

    val_vanilla = random.sample(vanilla_responses, min(VAL_PER_TYPE, len(vanilla_responses)))
    val_adversarial = random.sample(
        adversarial_responses, min(VAL_PER_TYPE, len(adversarial_responses))
    )
    train_vanilla = [r for r in vanilla_responses if r not in val_vanilla]
    train_adversarial = [r for r in adversarial_responses if r not in val_adversarial]
    if len(train_vanilla) > TRAIN_CAP_PER_TYPE:
        train_vanilla = random.sample(train_vanilla, TRAIN_CAP_PER_TYPE)
    if len(train_adversarial) > TRAIN_CAP_PER_TYPE:
        train_adversarial = random.sample(train_adversarial, TRAIN_CAP_PER_TYPE)

    val_responses = val_vanilla + val_adversarial
    train_responses = train_vanilla + train_adversarial
    random.shuffle(val_responses)
    random.shuffle(train_responses)
    logger.info(
        "Split: train=%d (%d vanilla + %d adversarial), val=%d (%d vanilla + %d adversarial)",
        len(train_responses),
        len(train_vanilla),
        len(train_adversarial),
        len(val_responses),
        len(val_vanilla),
        len(val_adversarial),
    )
    return train_responses, val_responses


def collect(args: argparse.Namespace, client: OpenAI) -> None:
    vanilla_harmful, adversarial_harmful = load_wildjailbreak_full()
    vanilla_prompts = set(vanilla_harmful)
    adversarial_prompts = set(adversarial_harmful)

    prev_path = results_path(Path(args.results_dir), "wildjailbreak", args.gen_model, args.samples)
    if not prev_path.exists():
        raise FileNotFoundError(f"Current results file not found: {prev_path}")
    records = read_jsonl(prev_path)
    used_prompts, current_vanilla, current_adversarial = analyze_current_results(
        records, vanilla_prompts, adversarial_prompts
    )

    if current_vanilla >= TARGET_VANILLA and current_adversarial >= TARGET_ADVERSARIAL:
        logger.info("Already have enough successful examples; nothing to do.")
        return

    additional_vanilla, additional_adversarial = sample_additional_prompts(
        used_prompts, vanilla_harmful, adversarial_harmful, current_vanilla, current_adversarial
    )
    all_additional = additional_vanilla + additional_adversarial
    if not all_additional:
        logger.info("No additional prompts to process.")
        return

    if args.statistic:
        logger.info(
            "[statistic] would process %d additional prompts (%d vanilla + %d adversarial)",
            len(all_additional),
            len(additional_vanilla),
            len(additional_adversarial),
        )
        return

    run_id = f"continue_wildjailbreak_{int(time.time())}_{random.randint(0, 9999):04d}"
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    gen_records = build_generation_records(
        all_additional,
        args.gen_model,
        args.temperature,
        args.new_samples,
        "wildjailbreak",
        GEN_SYS_PROMPT_FRESH,
    )
    gen_map_raw = run_generation_batch(client, gen_records, tmp_dir, run_id, args.poll_interval)
    gen_map: Dict[int, List[dict]] = {i: resp for i, resp in enumerate(gen_map_raw.values())}

    eval_records = build_eval_records(
        gen_map, all_additional, args.judge_model, "wildjailbreak", MIN_TOKENS_FOR_EVAL
    )
    eval_map = run_eval_batch(client, eval_records, tmp_dir, run_id, args.poll_interval)

    new_records = []
    for i, prompt in enumerate(all_additional):
        responses = []
        for j, resp in enumerate(gen_map.get(i, [])):
            txt = resp["text"]
            if count_tokens(txt) < MIN_TOKENS_FOR_EVAL:
                label = "harmless"
            else:
                label = eval_map.get((i, j), "error:missing_eval")
            responses.append({"text": txt, "label": label})
        new_records.append({"prompt": prompt, "responses": responses})

    all_records = records + new_records
    apply_min_token_override(all_records, MIN_TOKENS_FOR_EVAL)

    out_log = results_path(Path(args.results_dir), "wildjailbreak", args.gen_model, args.new_samples)
    write_jsonl(out_log, all_records)
    logger.info("Saved merged log (%d records) -> %s", len(all_records), out_log)

    filtered, stats = build_corpus(all_records, args.filter)
    log_corpus_stats(stats)

    train_responses, val_responses = split_train_val(
        filtered, vanilla_prompts, adversarial_prompts
    )
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train_responses.jsonl", train_responses)
    write_jsonl(out_dir / "val_responses.jsonl", val_responses)
    all_responses = train_responses + val_responses
    random.shuffle(all_responses)
    write_jsonl(out_dir / "responses.jsonl", all_responses)
    logger.info(
        "Wrote %d train / %d val / %d total -> %s",
        len(train_responses),
        len(val_responses),
        len(all_responses),
        out_dir,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="Generator chat model")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Safety-judge chat model")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=3, help="Sample count that names the prior log")
    ap.add_argument("--new-samples", type=int, default=3, help="Completions per prompt on retry")
    ap.add_argument("--statistic", action="store_true", help="Report only; call no models")
    ap.add_argument("--filter", action="store_true", help="Only keep harmful completions")
    ap.add_argument("--results-dir", default="results", help="Directory holding the eval logs")
    ap.add_argument(
        "--out-dir",
        default="data/train/probe/harmful/wildjailbreak",
        help="Output directory for the harmful probe corpus",
    )
    ap.add_argument("--tmp-dir", default="tmp", help="Scratch dir for batch input files")
    ap.add_argument("--poll-interval", type=int, default=10, help="Batch poll interval (seconds)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    collect(args, OpenAI())


if __name__ == "__main__":
    main()
