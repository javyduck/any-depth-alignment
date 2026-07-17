"""Generate local (vLLM) benign responses for over-refusal evaluation (over-refusal).

For each benign/over-refusal benchmark, the target model answers every prompt so
downstream evaluation can measure how often ADA needlessly refuses a benign
request. This is the benign counterpart to the harmful-prefill corpus and is used
by the over-refusal experiments.

Two behaviours are folded into one script:

* Benign benchmarks (GSM8K, MATH, MMLU, BBH, HumanEval, SimpleQA, GPQA, XSTest,
  AlpacaEval): sample at temperature 0.7. For the multiple-choice / short-answer
  sets (MATH, MMLU, BBH, SimpleQA) prompts are shuffled and the first 1,000
  completions longer than 25 tokens are kept.
* SafeDecoding attacker prompts: run ``--dataset safedecoding --greedy`` to
  reproduce the SafeDecoding baseline (greedy decoding, no length filtering).

If a name is not a benign benchmark it falls back to the harmful loader, so
``--dataset safedecoding`` resolves to the SafeDecoding attacker prompts.

Per-model chat-template quirks (borrowed templates, family-specific generation
suffixes for reasoning / harmony / [/INST] formatting) are resolved from
``ada.registry`` where possible.

Output
------
``{out_root}/{dataset}/{model_slug}/responses.jsonl`` — ``{"messages": [...]}``.

Run with::

    python -m ada.datagen.gen_benign_responses --dataset xstest --model google/gemma-2-9b-it
    python -m ada.datagen.gen_benign_responses --dataset safedecoding --greedy \
        --model meta-llama/Llama-2-7b-chat-hf
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import List

from ada.data.benchmarks import load_benign_prompts, load_harmful_prompts
from ada.registry import chat_template_source, get_model
from ada.utils.io import write_jsonl
from ada.utils.naming import slugify_model

logger = logging.getLogger(__name__)

# Short-answer / multiple-choice sets: shuffle, then keep first 1,000 answers
# longer than the token floor below.
FILTER_DATASETS = {"math", "mmlu", "bbh", "simpleqa"}
FILTER_TARGET = 1000
FILTER_MIN_TOKENS = 25

# Exact chat models that require a trailing space after the [/INST] header.
_SPACE_SUFFIX_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
}


def load_prompts(dataset: str) -> List[str]:
    """Load a benign benchmark, falling back to the harmful loader by name."""
    try:
        return load_benign_prompts(dataset)
    except ValueError:
        return load_harmful_prompts(dataset)


def build_model_and_tokenizer(model_name: str):
    """Build a vLLM engine + tokenizer, borrowing a chat template if needed."""
    from transformers import AutoTokenizer
    from vllm import LLM

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",  # vLLM supports bfloat16 more reliably than float16
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=32,
    )
    tokenizer = llm.get_tokenizer()

    template_src = chat_template_source(model_name)
    if template_src and not getattr(tokenizer, "chat_template", None):
        logger.info("Borrowing chat template for %s from %s", model_name, template_src)
        tokenizer.chat_template = AutoTokenizer.from_pretrained(
            template_src, trust_remote_code=True
        ).chat_template
    return llm, tokenizer


def generation_suffix(model_name: str) -> str:
    """String appended after the generation prompt to reach the answer channel.

    Registry-first: for a registered model it is exactly
    ``ModelSpec.generation_prompt_completion`` (the single source of truth). The
    name-based heuristic below is only a fallback for *unregistered* extra
    checkpoints that over-refusal generation may also cover (e.g. Mistral-7B-v0.3,
    gpt-oss-20b, Qwen3), which have no registry entry.
    """
    try:
        return get_model(model_name).generation_prompt_completion
    except KeyError:
        pass
    lower = model_name.lower()
    if "gpt-oss" in lower:
        return "<|channel|>analysis<|message|>\n\n<|end|><|start|>assistant<|channel|>final<|message|>"
    if "deepseek-r1-distill" in lower:
        return "\n</think>\n\n"
    if model_name in _SPACE_SUFFIX_MODELS:
        return " "
    return ""


def prepare_prompts(prompts: List[str], tokenizer, model_name: str) -> List[str]:
    """Apply the chat template plus any family-specific generation suffix."""
    suffix = generation_suffix(model_name)
    formatted = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted.append(text + suffix)
    return formatted


def collect(args: argparse.Namespace) -> None:
    logger.info("Loading dataset '%s' ...", args.dataset)
    prompts = load_prompts(args.dataset)
    logger.info("Loaded %d prompts", len(prompts))

    if args.max_prompts and args.max_prompts < len(prompts):
        rng = random.Random(args.seed)
        rng.shuffle(prompts)
        prompts = prompts[: args.max_prompts]
        logger.info("Sampled %d prompts", len(prompts))

    should_filter = args.dataset.lower() in FILTER_DATASETS
    if should_filter:
        random.Random(args.seed).shuffle(prompts)
        logger.info("Filter dataset: keeping first %d answers > %d tokens", FILTER_TARGET, FILTER_MIN_TOKENS)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0 if args.greedy else 0.7,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
    )

    logger.info("Loading model '%s' ...", args.model)
    llm, tokenizer = build_model_and_tokenizer(args.model)

    responses: List[dict] = []
    total_generated = 0
    prompt_idx = 0
    while prompt_idx < len(prompts):
        if should_filter and len(responses) >= FILTER_TARGET:
            break
        batch = prompts[prompt_idx : prompt_idx + args.batch_size]
        prompt_idx += len(batch)
        if not batch:
            break
        outputs = llm.generate(
            prepare_prompts(batch, tokenizer, args.model), sampling_params, use_tqdm=False
        )
        batch_responses = [o.outputs[0].text.strip() for o in outputs]
        total_generated += len(batch_responses)
        for prompt, response in zip(batch, batch_responses):
            if not response:
                continue
            if should_filter:
                if len(tokenizer.encode(response, add_special_tokens=False)) <= FILTER_MIN_TOKENS:
                    continue
            responses.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
            )
            if should_filter and len(responses) >= FILTER_TARGET:
                break
        logger.info("generated=%d accepted=%d", total_generated, len(responses))

    if should_filter and len(responses) > FILTER_TARGET:
        responses = responses[:FILTER_TARGET]

    logger.info("Final: generated %d, collected %d", total_generated, len(responses))

    out_path = Path(args.out_root) / args.dataset.lower() / slugify_model(args.model) / "responses.jsonl"
    write_jsonl(out_path, responses)
    logger.info("Results saved -> %s", out_path)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dataset", required=True, help="Benign benchmark, or 'safedecoding'")
    ap.add_argument("--model", required=True, help="HF model id to generate responses with")
    ap.add_argument("--greedy", action="store_true", help="Greedy decoding (temperature 0)")
    ap.add_argument("--max-prompts", type=int, default=None, help="Optional cap on prompt count")
    ap.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens per response")
    ap.add_argument("--batch-size", type=int, default=32, help="vLLM generation batch size")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/sampling")
    ap.add_argument("--out-root", default="benign_responses", help="Output root directory")
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
