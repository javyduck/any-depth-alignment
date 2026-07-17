"""SFT fine-tuning attack against Any-Depth Alignment (ADA).

This module implements the supervised fine-tuning (SFT) *attack* used in the
paper's robustness study (experiment SFT-attack). A LoRA adapter is trained on a small
set of harmful instruction/response conversations to see whether a few gradient
steps can erode a model's innate shallow-refusal alignment (the alignment ADA
re-triggers at depth). Only the assistant tokens contribute to the loss; every
prompt/header token is masked to ``-100``.

Checkpoint-sweep design
-----------------------
The attack is evaluated as a function of fine-tuning *depth* (number of update
steps). Rather than emitting intermediate checkpoints from a single run, each
point on the sweep is produced by an **independent** training run with its own
``--max_steps`` and its own ``--output_dir`` / ``--final_adapter_name``.
Automatic checkpointing is therefore disabled (``save_strategy="no"``); the run
saves only the final adapter to ``<output_dir>/<final_adapter_name>``. An
external driver launches one run per step count to build the full sweep.

Training uses LoRA on the attention and MLP projection matrices together with
DeepSpeed ZeRO-3 (optimizer/parameter CPU offload) for memory efficiency. The
tokenizer's chat template is resolved through :mod:`ada.registry` (which knows,
for example, that the augmented Llama-2 deep-alignment baseline ships without a
template and must borrow one), so this file contains no model-name conditionals.

Run with, e.g.::

    deepspeed -m ada.attacks.sft \\
        --model_name_or_path meta-llama/Llama-2-7b-chat-hf \\
        --data_path data/train/sft/harmful_sft.jsonl \\
        --output_dir runs/e4_sft \\
        --final_adapter_name adapter-60 \\
        --max_steps 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from ..models.loading import load_tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# configs/ lives next to the ada/ package at the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEEPSPEED_CONFIG = str(_REPO_ROOT / "configs" / "deepspeed_zero3.json")

# The LoRA target modules used throughout the paper: all attention and MLP
# projection matrices.
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs and enable deterministic cuDNN for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SupervisedDataset(Dataset):
    """SFT dataset built from the tokenizer's chat template.

    Each record is a list of chat messages. Conversations are re-tokenized
    incrementally so that only assistant-response tokens carry a training label;
    all prompt, system, and header tokens are masked to ``-100``. Sequences are
    truncated to ``max_seq_length``.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 2048,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define `eos_token`.")

        # Right padding is required for causal LM training.
        tokenizer.padding_side = "right"

        if getattr(tokenizer, "chat_template", None) is None:
            logger.warning("Tokenizer has no chat_template. This may cause issues.")

        self.data = self._load_jsonl(data_path)
        logger.info("Loaded %d examples from %s", len(self.data), data_path)

    @staticmethod
    def _load_jsonl(path: str) -> List[List[Dict]]:
        """Read a JSONL file of chat conversations into lists of messages."""
        examples: List[List[Dict]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        examples.append(obj["messages"])
                    elif obj.get("body", {}).get("messages"):
                        examples.append(obj["body"]["messages"])
                    else:
                        logger.warning("Line %d: Skipping item without `messages`.", line_num)
                except json.JSONDecodeError:
                    logger.warning("Line %d: Skipping malformed JSON.", line_num)
        return examples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """Build ``(input_ids, attention_mask, labels)`` for one conversation.

        The conversation is rebuilt incrementally: for every assistant turn we
        re-tokenize the prefix (with the generation prompt / assistant header)
        and the prefix-plus-response, so the response tokens are exactly their
        difference. Only those response tokens receive labels; everything else,
        including the assistant header, is masked to ``-100``.
        """
        msgs = self.data[idx]

        full_input_ids: List[int] = []
        full_labels: List[int] = []

        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant":
                # Conversation up to (but excluding) this assistant message.
                conv_so_far = msgs[:i]
                full_conv = conv_so_far + [msg]

                # Tokens through the assistant header (generation prompt).
                conv_tokens = self.tokenizer.apply_chat_template(
                    conv_so_far,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,
                )

                # Tokens for the full conversation including the response.
                full_tokens = self.tokenizer.apply_chat_template(
                    full_conv,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_tensors=None,
                )

                assistant_response_tokens = full_tokens[len(conv_tokens):]

                prev_len = len(full_input_ids)
                full_input_ids = full_tokens

                # -100 for everything except the current assistant response.
                new_labels = [-100] * len(conv_tokens) + assistant_response_tokens
                full_labels = [-100] * prev_len + new_labels[prev_len:]

                # Keep labels aligned with input_ids.
                full_labels = full_labels[: len(full_input_ids)]
                if len(full_labels) < len(full_input_ids):
                    full_labels.extend([-100] * (len(full_input_ids) - len(full_labels)))

            if len(full_input_ids) >= self.max_seq_length:
                full_input_ids = full_input_ids[: self.max_seq_length]
                full_labels = full_labels[: self.max_seq_length]
                break

        # No assistant turns: tokenize everything with all labels masked.
        if not full_input_ids:
            full_input_ids = self.tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors=None,
            )
            full_labels = [-100] * len(full_input_ids)

        if len(full_input_ids) > self.max_seq_length:
            full_input_ids = full_input_ids[: self.max_seq_length]
            full_labels = full_labels[: self.max_seq_length]

        attention_mask = [1] * len(full_input_ids)
        return {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "labels": full_labels,
        }


class CustomDataCollator:
    """Right-pad variable-length sequences into a training batch.

    ``input_ids`` are padded with the tokenizer pad id, ``attention_mask`` with
    0, and ``labels`` with ``-100`` so padding never contributes to the loss.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(f["input_ids"]) for f in features)

        input_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            padding_length = max_length - len(input_ids)

            input_ids_batch.append(input_ids + [self.tokenizer.pad_token_id] * padding_length)
            attention_mask_batch.append(attention_mask + [0] * padding_length)
            labels_batch.append(labels + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


def build_model(model_name_or_path: str, use_flash_attention: bool = True):
    """Load the base causal LM for LoRA training under DeepSpeed ZeRO-3.

    ``device_map`` is left as ``None`` so DeepSpeed handles device placement, and
    the KV cache is disabled for training. Flash Attention 2 is attempted first
    and falls back to the default kernel if unavailable.
    """
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "token": os.environ.get("HF_TOKEN"),
        "use_cache": False,  # required for gradient checkpointing / training
        "device_map": None,  # DeepSpeed handles device placement
    }

    if use_flash_attention:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, attn_implementation="flash_attention_2", **model_kwargs
            )
            logger.info("Using Flash Attention 2 for memory efficiency")
        except Exception as err:  # noqa: BLE001 - fall back on any FA2 failure
            logger.warning("Flash Attention 2 not available: %s. Using standard attention.", err)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    logger.info("Enabling gradient checkpointing for memory optimization")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model


def build_lora_config(
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    target_modules=DEFAULT_LORA_TARGET_MODULES,
) -> LoraConfig:
    """Build the causal-LM LoRA config used by the SFT attack."""
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    """Assemble the (mostly fixed) TrainingArguments for the attack.

    Automatic checkpointing is disabled (``save_strategy="no"``): each depth in
    the checkpoint sweep is an independent ``max_steps`` run that saves only its
    final adapter.
    """
    return TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_strategy="no",  # only the final adapter is saved
        bf16=True,
        optim="adamw_torch",
        deepspeed=args.deepspeed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        disable_tqdm=False,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
    )


def run_training(args: argparse.Namespace) -> str:
    """Run one SFT-attack training and save the final LoRA adapter.

    Returns the directory the final adapter was written to.
    """
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading tokenizer for %s", args.model_name_or_path)
    tokenizer = load_tokenizer(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model %s", args.model_name_or_path)
    model = build_model(args.model_name_or_path)

    lora_config = build_lora_config(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SupervisedDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    if args.max_train_samples is not None:
        train_dataset.data = train_dataset.data[: args.max_train_samples]
        logger.info("Limited training data to %d samples", args.max_train_samples)

    data_collator = CustomDataCollator(tokenizer)
    training_args = build_training_arguments(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    final_output_dir = os.path.join(args.output_dir, args.final_adapter_name)
    logger.info("Saving final adapter to %s", final_output_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    logger.info("Training completed!")
    return final_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT-attack training (LoRA + DeepSpeed ZeRO-3) against ADA."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or HuggingFace model identifier")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data (JSONL of chat conversations)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for this training run")
    parser.add_argument("--final_adapter_name", type=str, default="final_adapter",
                        help="Sub-directory name for the saved final adapter")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Optional cap on the number of training samples")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Number of training steps for this run (one sweep point)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log training metrics every N steps")
    parser.add_argument("--deepspeed", type=str, default=DEFAULT_DEEPSPEED_CONFIG,
                        help="DeepSpeed configuration JSON (ZeRO-3)")

    # LoRA arguments.
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")

    # Added by the DeepSpeed launcher; accepted so argparse does not error.
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by DeepSpeed)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
