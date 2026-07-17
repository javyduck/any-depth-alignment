"""Generate the WildChat-1M benign half of the ADA-LP probe corpus (local vLLM).

Complements the WildJailbreak-benign split with in-distribution, genuinely benign
assistant responses drawn from real WildChat-1M user turns. The target model
itself generates the completions (via vLLM), so the benign prefills the probe
learns from match the model whose Safety-Token hidden state ADA-LP will read.

Only single-turn English dialogues are used (first user turn only, <= 500 words,
non-empty, not a "As a prompt generator" meta-prompt). Prompts are shuffled with
a fixed seed, the model generates at temperature 0.7, and completions shorter
than 50 words are discarded. Collection stops at 11,000 accepted responses,
split into 10,000 train + 1,000 val.

Outputs
-------
``{out_dir}/{train,val}_responses.jsonl`` where ``out_dir`` defaults to
``data/train/probe/benign/wildchat1m`` — ``{"messages": [user, assistant]}``.

Run with::

    python -m ada.datagen.gen_wildchat1m --model google/gemma-2-9b-it
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import List

from ada.utils.io import write_jsonl

logger = logging.getLogger(__name__)

TARGET_RESPONSES = 11000
N_TRAIN = 10000
MIN_RESPONSE_WORDS = 50
MAX_PROMPT_WORDS = 500


def count_words(text: str) -> int:
    return len(text.strip().split())


def build_vllm_engine(model_name: str):
    """Build a vLLM engine + tokenizer for high-throughput batch generation."""
    from vllm import LLM

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",  # vLLM supports bfloat16 more reliably than float16
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=32,
    )
    return llm, llm.get_tokenizer()


def prepare_prompts(prompts: List[str], tokenizer) -> List[str]:
    """Apply the chat template with the generation prompt appended."""
    formatted = []
    for prompt in prompts:
        formatted.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return formatted


def is_single_turn_valid(example: dict) -> bool:
    """Keep valid single-turn English dialogues (first user turn only)."""
    if example["language"] != "English":
        return False
    conversation = example["conversation"]
    if len(conversation) < 2:
        return False
    if conversation[0]["role"] != "user" or conversation[1]["role"] != "assistant":
        return False
    user_message = conversation[0]["content"].strip()
    if user_message == "":
        return False
    if count_words(user_message) > MAX_PROMPT_WORDS:
        return False
    if user_message.startswith("As a prompt generator"):
        return False
    return True


def load_wildchat1m_prompts(seed: int = 42) -> List[str]:
    """Load, filter, and shuffle first-turn prompts from WildChat-1M."""
    from datasets import load_dataset

    logger.info("Loading WildChat-1M dataset ...")
    ds = load_dataset("allenai/WildChat-1M", split="train")
    total = len(ds)
    prompts, start = [], time.time()
    for idx, example in enumerate(ds, 1):
        if is_single_turn_valid(example):
            prompts.append(example["conversation"][0]["content"].strip())
        if idx % 10000 == 0:
            logger.info(
                "  processed %d/%d | kept %d (%.1fs)", idx, total, len(prompts), time.time() - start
            )
    logger.info("Loaded %d prompts from WildChat-1M", len(prompts))
    random.seed(seed)
    random.shuffle(prompts)
    return prompts


def generate_responses(args: argparse.Namespace) -> List[dict]:
    """Generate until TARGET_RESPONSES accepted (>= MIN_RESPONSE_WORDS words)."""
    from vllm import SamplingParams

    prompts = load_wildchat1m_prompts(args.seed)
    logger.info("Loading model '%s' ...", args.model)
    llm, tokenizer = build_vllm_engine(args.model)

    sampling_params = SamplingParams(
        temperature=0.7, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=args.max_tokens
    )

    responses: List[dict] = []
    total_generated = 0
    prompt_idx = 0
    while len(responses) < TARGET_RESPONSES and prompt_idx < len(prompts):
        batch = prompts[prompt_idx : prompt_idx + args.batch_size]
        prompt_idx += len(batch)
        if not batch:
            break
        outputs = llm.generate(prepare_prompts(batch, tokenizer), sampling_params, use_tqdm=False)
        batch_responses = [o.outputs[0].text.strip() for o in outputs]
        total_generated += len(batch_responses)
        for prompt, response in zip(batch, batch_responses):
            if not response or count_words(response) < MIN_RESPONSE_WORDS:
                continue
            responses.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
            )
            if len(responses) >= TARGET_RESPONSES:
                break
        logger.info("generated=%d accepted=%d/%d", total_generated, len(responses), TARGET_RESPONSES)

    if len(responses) < TARGET_RESPONSES:
        logger.warning("Collected only %d responses (target %d)", len(responses), TARGET_RESPONSES)
    return responses


def collect(args: argparse.Namespace) -> None:
    responses = generate_responses(args)
    train_responses = responses[:N_TRAIN]
    val_responses = responses[N_TRAIN:TARGET_RESPONSES]

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train_responses.jsonl", train_responses)
    write_jsonl(out_dir / "val_responses.jsonl", val_responses)
    logger.info(
        "Wrote %d train / %d val responses -> %s",
        len(train_responses),
        len(val_responses),
        out_dir,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", required=True, help="HF model id to generate benign responses with")
    ap.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens per response")
    ap.add_argument("--batch-size", type=int, default=64, help="vLLM generation batch size")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument("--seed", type=int, default=42, help="Prompt shuffle seed")
    ap.add_argument(
        "--out-dir",
        default="data/train/probe/benign/wildchat1m",
        help="Output directory for the benign probe corpus",
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    collect(args)


if __name__ == "__main__":
    main()
