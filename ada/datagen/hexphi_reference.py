"""Compliant sharing of the HEx-PHI deep-prefill split.

HEx-PHI is gated under the LLM-Tuning-Safety license and may not be
redistributed, so we cannot publish ``hexphi_responses.jsonl`` (it embeds the
gated prompts). But the harmful *continuations* we generate are our own model
outputs, not HEx-PHI content — the only HEx-PHI-derived thing needed to pair them
back up is the prompt itself, which each user already has once they accept the
HEx-PHI license.

This module bridges that gap without ever moving HEx-PHI text:

* ``export``      strips the prompt from each record and keys the continuation by
                  a **one-way SHA-256** of the prompt (irreversible), producing a
                  redistributable ``hexphi_references.jsonl``.
* ``reconstruct`` re-joins those references against a *local* HEx-PHI copy
                  (loaded via ``ada.data.benchmarks.load_hexphi`` with the user's
                  own accepted license) to rebuild the full ``hexphi_responses.jsonl``.

The reference file contains no prompt text — only hashes + our continuations — so
it is safe to publish (gated), while the full split is only ever reconstructed on
a machine that already holds a licensed HEx-PHI copy.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from ..data.benchmarks import load_hexphi
from ..data.loading import extract_messages
from ..utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

DEFAULT_RESPONSES = Path("data/eval/deep_prefill/hexphi_responses.jsonl")
DEFAULT_REFERENCES = Path("data/eval/deep_prefill/hexphi_references.jsonl")


def _prompt_hash(prompt: str) -> str:
    """Irreversible identifier for a HEx-PHI prompt (not the prompt itself)."""
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def export_references(responses: Path, references: Path) -> None:
    """Strip prompts → write ``{hexphi_sha256, hexphi_index, response}`` records.

    Requires the local, licensed ``hexphi_responses.jsonl`` and HEx-PHI access
    (to map each prompt to a stable index). No prompt text is written out.
    """
    prompts = load_hexphi()
    index_of = {_prompt_hash(p): i for i, p in enumerate(prompts)}

    out, unmatched = [], 0
    for rec in read_jsonl(responses):
        msgs = extract_messages(rec)
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
        h = _prompt_hash(user)
        if h not in index_of:
            unmatched += 1
            continue
        out.append({"hexphi_sha256": h, "hexphi_index": index_of[h], "response": assistant})

    write_jsonl(references, out)
    logger.info("Wrote %d references to %s (%d unmatched, dropped)", len(out), references, unmatched)
    if unmatched:
        logger.warning("%d records did not match a HEx-PHI prompt and were skipped", unmatched)


def reconstruct_responses(references: Path, responses: Path) -> None:
    """Re-join references with a local HEx-PHI copy → full ``hexphi_responses.jsonl``.

    The user must have accepted the HEx-PHI license (``load_hexphi`` pulls it with
    their ``HF_TOKEN``). No HEx-PHI text ever left the original source.
    """
    prompts = load_hexphi()
    prompt_of = {_prompt_hash(p): p for p in prompts}

    out, missing = [], 0
    for ref in read_jsonl(references):
        prompt = prompt_of.get(ref.get("hexphi_sha256"))
        if prompt is None:  # fall back to positional index if hashes shifted
            idx = ref.get("hexphi_index")
            prompt = prompts[idx] if isinstance(idx, int) and idx < len(prompts) else None
        if prompt is None:
            missing += 1
            continue
        out.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ref["response"]},
        ]})

    write_jsonl(responses, out)
    logger.info("Reconstructed %d records to %s (%d unresolved)", len(out), responses, missing)
    if missing:
        logger.warning(
            "%d references could not be matched to your HEx-PHI copy — check that "
            "you accepted the current HEx-PHI license/version", missing
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="prompts -> hashes (redistributable references)")
    e.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    e.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)

    r = sub.add_parser("reconstruct", help="references + local HEx-PHI -> full responses")
    r.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    r.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)

    args = p.parse_args()
    if args.cmd == "export":
        export_references(args.responses, args.references)
    else:
        reconstruct_responses(args.references, args.responses)


if __name__ == "__main__":
    main()
