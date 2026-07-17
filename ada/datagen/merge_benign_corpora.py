"""Merge the two benign probe corpora into a single training/validation set.

ADA-LP's probe is trained on benign prefills drawn from two complementary
sources — WildChat-1M (in-distribution, model-generated) and WildJailbreak-benign
(adversarially-framed but benign). This step concatenates their train and val
splits into one ``merged`` corpus consumed by the probe-training stage.

Records are concatenated in a fixed order (WildChat-1M first, then
WildJailbreak) and *not* reshuffled, matching the source pipeline so the merged
corpus is byte-reproducible.

Inputs
------
``{benign_root}/{wildchat1m,wildjailbreak}/{train,val}_responses.jsonl``

Output
------
``{benign_root}/merged/{train,val}_responses.jsonl``

Run with::

    python -m ada.datagen.merge_benign_corpora
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from ada.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def merge_split(sources: List[Path], split: str) -> List[Dict]:
    """Concatenate ``{source}/{split}_responses.jsonl`` across sources, in order."""
    merged: List[Dict] = []
    for source in sources:
        path = source / f"{split}_responses.jsonl"
        records = read_jsonl(path)
        logger.info("  %s %s: %d records", source.name, split, len(records))
        merged.extend(records)
    return merged


def collect(args: argparse.Namespace) -> None:
    benign_root = Path(args.benign_root)
    sources = [benign_root / name for name in args.sources]
    out_dir = benign_root / args.merged_name

    for split in ("train", "val"):
        logger.info("Merging %s split ...", split)
        records = merge_split(sources, split)
        out_path = out_dir / f"{split}_responses.jsonl"
        write_jsonl(out_path, records)
        logger.info("Wrote %d %s records -> %s", len(records), split, out_path)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--benign-root",
        default="data/train/probe/benign",
        help="Root directory containing the per-source benign corpora",
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["wildchat1m", "wildjailbreak"],
        help="Sub-directories to merge, in concatenation order",
    )
    ap.add_argument("--merged-name", default="merged", help="Output sub-directory name")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    collect(parse_args(argv))


if __name__ == "__main__":
    main()
