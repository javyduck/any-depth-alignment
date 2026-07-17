#!/usr/bin/env python3
"""Publish the Any-Depth Alignment datasets and ADA-LP probes to the Hub.

Creates two repositories:
  * a GATED dataset repo  (harmful safety-research content; HEx-PHI excluded)
  * a PUBLIC model repo    (the ADA-LP logistic probes)

Usage:
    python scripts/upload_to_hf.py --namespace javyduck
    python scripts/upload_to_hf.py --namespace javyduck --skip-dataset   # probes only

Requires a logged-in HuggingFace account (`huggingface-cli login`) with write
access to the namespace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

# HEx-PHI PROMPT-BEARING files are gated (LLM-Tuning-Safety) and never uploaded.
# The prompt-free reference file (hexphi_references.jsonl — SHA-256 hashes + our
# own continuations) IS uploaded so licensed users can reconstruct the split.
DATASET_IGNORE = [
    "*hexphi_responses.jsonl",
    "*hexphi_judgments.jsonl",
    "*hexphi_full_judgments.csv",
    "*hexphi_benign_indexes.txt",
    "*hex-phi*", "*hex_phi*",
    # Withheld: the SFT data that jailbreaks the generator model (the "recipe").
    # We release the harmful *continuations* for defense eval, never the recipe.
    "train/openai_ft/*", "train/openai_ft/**", "*jailbroken*", "*insecure*",
    "README.md", "generated/*", "*.gitkeep",
    "**/__pycache__/*", "*.pyc",
]
PROBES_IGNORE = ["**/__pycache__/*", "*.pyc"]


def upload_dataset(api: HfApi, namespace: str, name: str) -> None:
    repo_id = f"{namespace}/{name}"

    # Generate the compliant HEx-PHI reference file (hashes + our continuations,
    # no gated prompt text) so licensed users can reconstruct the split.
    responses = ROOT / "data/eval/deep_prefill/hexphi_responses.jsonl"
    references = ROOT / "data/eval/deep_prefill/hexphi_references.jsonl"
    if responses.exists() and not references.exists():
        print("[dataset] generating HEx-PHI references (prompt-free)")
        from ada.datagen.hexphi_reference import export_references
        export_references(responses, references)

    print(f"[dataset] creating gated repo {repo_id}")
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    # Gated = access-request (author approves each request).
    api.update_repo_settings(repo_id, repo_type="dataset", gated="manual")

    print(f"[dataset] uploading data/ (HEx-PHI excluded) -> {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(ROOT / "data"),
        path_in_repo="",
        ignore_patterns=DATASET_IGNORE,
        commit_message="Add ADA train/eval datasets (HEx-PHI excluded)",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(ROOT / "scripts/hf_cards/dataset_card.md"),
        path_in_repo="README.md",
        commit_message="Add dataset card",
    )
    print(f"[dataset] done: https://huggingface.co/datasets/{repo_id}")


def upload_results(api: HfApi, namespace: str, name: str, results_dir: Path) -> None:
    """Upload the slim, text-stripped example logs + figures to the gated dataset repo.

    Placed under ``example_results/`` so `make_all_figures.sh` can regenerate the
    main figures without re-running any inference. See scripts/build_example_results.py.
    """
    repo_id = f"{namespace}/{name}"
    print(f"[results] uploading {results_dir} -> {repo_id}/example_results")
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(results_dir),
        path_in_repo="example_results",
        ignore_patterns=["**/__pycache__/*", "*.pyc"],
        commit_message="Add slim re-plottable example results (harmful text stripped)",
    )
    print(f"[results] done: https://huggingface.co/datasets/{repo_id}/tree/main/example_results")


def upload_probes(api: HfApi, namespace: str, name: str) -> None:
    repo_id = f"{namespace}/{name}"
    print(f"[probes] creating public repo {repo_id}")
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

    print(f"[probes] uploading ckpts/ -> {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(ROOT / "ckpts"),
        path_in_repo="ckpts",
        ignore_patterns=PROBES_IGNORE,
        commit_message="Add ADA-LP logistic probes",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(ROOT / "scripts/hf_cards/probes_card.md"),
        path_in_repo="README.md",
        commit_message="Add model card",
    )
    print(f"[probes] done: https://huggingface.co/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", default="javyduck", help="HF user or org")
    p.add_argument("--dataset-name", default="any-depth-alignment")
    p.add_argument("--probes-name", default="any-depth-alignment-probes")
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--skip-probes", action="store_true")
    p.add_argument("--results-dir", default=None,
                   help="If set, upload this staged example-results dir (from "
                        "build_example_results.py) to the dataset repo's example_results/.")
    args = p.parse_args()

    api = HfApi()
    print("[hf] authenticated as", api.whoami()["name"])
    if not args.skip_probes:
        upload_probes(api, args.namespace, args.probes_name)
    if not args.skip_dataset:
        upload_dataset(api, args.namespace, args.dataset_name)
    if args.results_dir:
        upload_results(api, args.namespace, args.dataset_name, Path(args.results_dir))


if __name__ == "__main__":
    main()
