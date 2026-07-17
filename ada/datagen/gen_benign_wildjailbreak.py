"""Build the WildJailbreak "benign" (safe-response) half of the ADA-LP probe corpus.

The probe that ADA-LP reads off the injected Safety-Token hidden state is trained
to separate deep *harmful* prefills from deep *benign* (safe) prefills. This
script supplies the safe class from WildJailbreak's own ``completion`` field,
which is a safe response for *every* data type: a helpful answer for benign
prompts and a refusal for harmful prompts. All four WildJailbreak data types
(vanilla/adversarial x benign/harmful) therefore contribute safe prefills, and
all four are sampled here — the "benign" label refers to the safe class, not to a
benign-only subset.

Completions shorter than 25 words are discarded; 10,000 examples per data type
are sampled for train and 1,000 per data type for val, written as
``{"messages": [user, assistant]}`` records.

Outputs
-------
``{out_dir}/{train,val}_responses.jsonl`` where ``out_dir`` defaults to
``data/train/probe/benign/wildjailbreak``.

Note
----
``sample_benign_wildjailbreak.py`` in the source tree was a near-duplicate of
this script (2,500/250 per data type, >=100-word filter) and is intentionally
dropped from the release; its behaviour is reachable here via the
``--train-per-category`` / ``--val-per-category`` / ``--min-words`` flags.

Run with::

    python -m ada.datagen.gen_benign_wildjailbreak
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

from ada.utils.io import write_jsonl

logger = logging.getLogger(__name__)

# WildJailbreak data types, in the order the source concatenated them.
DATA_TYPES = ("vanilla_harmful", "vanilla_benign", "adversarial_harmful", "adversarial_benign")


def load_wildjailbreak_completions(
    min_words: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Load WildJailbreak completions (>= ``min_words``) grouped by data type.

    Returns four ``[{"prompt", "response"}]`` lists in :data:`DATA_TYPES` order:
    vanilla_harmful, vanilla_benign, adversarial_harmful, adversarial_benign.
    The ``response`` is the dataset's safe ``completion`` for every data type;
    for the adversarial types the adversarial framing is used as the prompt when
    present (else the vanilla prompt).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/wildjailbreak", "train", split="train", delimiter="\t", keep_default_na=False
    )
    buckets: dict[str, List[dict]] = {dt: [] for dt in DATA_TYPES}
    for ex in ds:
        response = ex["completion"]
        if not response or not response.strip():
            continue
        if len(response.split()) < min_words:
            continue
        data_type = ex["data_type"]
        if data_type not in buckets:
            continue
        if data_type.startswith("adversarial"):
            prompt = ex["adversarial"] if ex["adversarial"] else ex["vanilla"]
        else:
            prompt = ex["vanilla"]
        buckets[data_type].append({"prompt": prompt, "response": response})

    for dt in DATA_TYPES:
        logger.info("WildJailbreak %s (>=%d words): %d", dt, min_words, len(buckets[dt]))
    return tuple(buckets[dt] for dt in DATA_TYPES)  # type: ignore[return-value]


def sample_data(
    categories: List[List[dict]],
    train_per_category: int,
    val_per_category: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """Sample train/val disjointly per category, then concatenate in order.

    The RNG draws happen in the source's order — *all* train sets first, then all
    val sets — so the same seed reproduces the exact split.
    """
    random.seed(seed)
    trains = [random.sample(items, min(train_per_category, len(items))) for items in categories]
    remainings = [
        [x for x in items if x not in train] for items, train in zip(categories, trains)
    ]
    vals = [random.sample(rem, min(val_per_category, len(rem))) for rem in remainings]
    train_data = [x for train in trains for x in train]
    val_data = [x for val in vals for x in val]
    return train_data, val_data


def format_responses(data: List[dict]) -> List[dict]:
    """Wrap ``{"prompt", "response"}`` pairs as ``{"messages": [user, assistant]}``."""
    return [
        {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]
        }
        for item in data
    ]


def collect(args: argparse.Namespace) -> None:
    categories = list(load_wildjailbreak_completions(args.min_words))
    if not any(categories):
        logger.warning("No data found; exiting.")
        return

    train_data, val_data = sample_data(
        categories,
        train_per_category=args.train_per_category,
        val_per_category=args.val_per_category,
        seed=args.seed,
    )
    train_responses = format_responses(train_data)
    val_responses = format_responses(val_data)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train_responses.jsonl", train_responses)
    write_jsonl(out_dir / "val_responses.jsonl", val_responses)
    logger.info(
        "Wrote %d train / %d val safe responses -> %s",
        len(train_responses),
        len(val_responses),
        out_dir,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--min-words", type=int, default=25, help="Minimum completion length (words)")
    # 2500 x 4 categories = 10k train, 250 x 4 = 1k val — the shipped 11k WildJailbreak
    # benign split (matches the source sampler sample_benign_wildjailbreak.py).
    ap.add_argument("--train-per-category", type=int, default=2500)
    ap.add_argument("--val-per-category", type=int, default=250)
    ap.add_argument(
        "--out-dir",
        default="data/train/probe/benign/wildjailbreak",
        help="Output directory for the benign probe corpus",
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    collect(parse_args(argv))


if __name__ == "__main__":
    main()
