"""Live-streaming ADA-LP demo server (Any-Depth Alignment, ICLR 2026).

This FastAPI server is a runnable demonstration of the paper's *real-time*
streaming defense. A single base model acts as its own guardrail: while it streams
a response token by token, ADA-LP periodically re-injects the model's assistant
header (the "Safety Tokens") and reads the hidden state at the injected probe
token to decide, at *any* generation depth, whether the response has turned
harmful.

Mechanism (identical to the offline ADA-LP evaluation, run online here):

* Hold one model + tokenizer + one scikit-learn logistic probe in memory.
* Prefill the prompt, then decode one token at a time, streaming each token.
* Every ``--check-interval`` generated tokens (default 25), **fork** the KV cache
  (a deep copy so generation is untouched), append the model's
  ``probe_safety_tokens`` from :mod:`ada.registry`, and run a *single* forward
  pass. Read the hidden state of the last injected token (the probe token) at the
  model's ``probe_layer`` / ``hook_position`` via
  :class:`ada.models.extraction.HookHiddenStateCollector`. Because the hook taps
  the block's ``input_layernorm`` output, the state is already normalized — no
  model-specific layernorm fix-up is needed (unlike the stale prototype).
* Apply the logistic probe (``predict_proba``). If the harmful probability crosses
  ``--threshold``, halt generation and substitute a refusal string.

The overhead is *constant*: one extra forward over a handful of header tokens per
check window, independent of how deep the response has grown.

Every per-model setting — the probe's Safety-Token span, the probe layer, and the
hook position — is resolved from :mod:`ada.registry`; nothing is hard-coded per
model. The probe checkpoint is loaded from the standard on-disk layout::

    ckpts/{model}/{safety}/{mask}/{hook}/{cache}/seed_{seed}/logistic/layer_{L}.joblib

resolved through :mod:`ada.utils.naming` so it stays compatible with the training
pipeline (``ada`` logistic-probe trainer).

Run::

    python -m ada.serving.server --model google/gemma-2-9b-it \\
        --check-interval 25 --threshold 0.5 --host 0.0.0.0 --port 8000

Endpoints:
    POST /generate   Stream (NDJSON) or return a guarded completion.
    GET  /healthz    Liveness + resolved configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

from ..models.extraction import HookHiddenStateCollector
from ..models.loading import load_model_and_tokenizer
from ..registry import get_model
from ..utils.naming import (
    slugify_cache,
    slugify_hook_position,
    slugify_mask_tokens,
    slugify_model,
    slugify_safety_tokens,
)

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Default canned refusal substituted when the probe flags the stream as harmful.
DEFAULT_REFUSAL = "I'm sorry, but I can't help with that."


# --------------------------------------------------------------------------- #
# Probe loading
# --------------------------------------------------------------------------- #
def build_probe_path(
    model_name: str,
    probe_safety_tokens: str,
    probe_layer: int,
    hook_position: str,
    *,
    ckpt_dir: "str | Path" = "ckpts",
    mask_tokens: Optional[str] = None,
    gradual_cache: bool = True,
    seed: int = 42,
) -> Path:
    """Resolve the joblib probe path for ``model_name`` using the naming scheme.

    Mirrors the layout written by the logistic-probe trainer so a probe trained
    offline is found unchanged here.
    """
    return (
        Path(ckpt_dir)
        / slugify_model(model_name)
        / slugify_safety_tokens(probe_safety_tokens)
        / slugify_mask_tokens(mask_tokens)
        / slugify_hook_position(hook_position)
        / slugify_cache(gradual_cache)
        / f"seed_{seed}"
        / "logistic"
        / f"layer_{probe_layer}.joblib"
    )


def load_probe(path: "str | Path"):
    """Load the scikit-learn logistic-regression probe saved with joblib."""
    import joblib  # local import: heavy dependency, only needed at startup

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Probe checkpoint not found: {path}\n"
            "Train it first (ada logistic-probe trainer) or point --ckpt-dir at "
            "the directory holding the released checkpoints."
        )
    clf = joblib.load(path)
    logger.info("Loaded logistic probe from %s", path)
    return clf


def probe_harmful_prob(clf, hidden: np.ndarray) -> float:
    """Probability that ``hidden`` is harmful under the logistic probe.

    Uses ``predict_proba`` when available, falling back to a sigmoid over the
    decision function — matching the offline evaluation exactly.
    """
    x = hidden.reshape(1, -1)
    if hasattr(clf, "predict_proba"):
        return float(clf.predict_proba(x)[0, 1])
    z = float(np.asarray(clf.decision_function(x)).reshape(-1)[0])
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------------- #
# Server state
# --------------------------------------------------------------------------- #
@dataclass
class DefenseConfig:
    """Defaults for the ADA-LP streaming check (overridable per request)."""

    threshold: float = 0.5
    check_interval: int = 25
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    refusal_string: str = DEFAULT_REFUSAL


class ServerState:
    """Holds the single in-memory model, tokenizer, and logistic probe."""

    def __init__(
        self,
        model_name: str,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        use_flash_attention: bool = False,
        ckpt_dir: "str | Path" = "ckpts",
        mask_tokens: Optional[str] = None,
        gradual_cache: bool = True,
        seed: int = 42,
        defaults: Optional[DefenseConfig] = None,
    ):
        import threading

        self.model_name = model_name
        self.spec = get_model(model_name)
        self.defaults = defaults or DefenseConfig()

        torch_dtype = _DTYPES[dtype]
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name, dtype=torch_dtype, device=device,
            use_flash_attention=use_flash_attention,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Per-model ADA-LP settings — all from the registry, never hard-coded.
        self.probe_layer = self.spec.probe_layer
        self.hook_position = self.spec.hook_position
        self.probe_safety_tokens = self.spec.probe_safety_tokens
        safety_ids = self.tokenizer.encode(
            self.probe_safety_tokens, add_special_tokens=False
        )
        self.safety_ids = torch.tensor([safety_ids], device=self.device)

        probe_path = build_probe_path(
            model_name,
            self.probe_safety_tokens,
            self.probe_layer,
            self.hook_position,
            ckpt_dir=ckpt_dir,
            mask_tokens=mask_tokens,
            gradual_cache=gradual_cache,
            seed=seed,
        )
        self.probe = load_probe(probe_path)

        # One collector, reused under the model lock (single active generation).
        self.collector = HookHiddenStateCollector(
            self.model, self.hook_position, [self.probe_layer]
        )
        # Serialize forwards so concurrent requests never interleave on the model.
        self.model_lock = threading.Lock()

        logger.info(
            "ADA-LP ready: model=%s probe_layer=%d hook=%s safety_tokens=%r",
            model_name, self.probe_layer, self.hook_position, self.probe_safety_tokens,
        )


# --------------------------------------------------------------------------- #
# Sampling + probe check
# --------------------------------------------------------------------------- #
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Greedy (temperature <= 0) or nucleus (top-p) sampling of one token."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cdf > top_p
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, choice).squeeze(-1)


def run_probe_check(state: ServerState, past) -> float:
    """Fork the KV cache, inject Safety Tokens, and return the harmful probability.

    A deep copy of ``past`` is used because the injected forward would otherwise
    append its keys/values to the live generation cache.
    """
    probe_past = copy.deepcopy(past)

    def forward_fn():
        return state.model(
            input_ids=state.safety_ids,
            past_key_values=probe_past,
            use_cache=False,
            output_hidden_states=False,
        )

    with torch.inference_mode():
        states_by_layer, _ = state.collector.collect([-1], forward_fn)

    hidden = states_by_layer[state.probe_layer][0, 0, :]  # last injected token
    hidden_np = hidden.float().cpu().numpy()
    return probe_harmful_prob(state.probe, hidden_np)


# --------------------------------------------------------------------------- #
# Guarded streaming generation
# --------------------------------------------------------------------------- #
def stream_events(
    state: ServerState,
    messages: List[Dict[str, Any]],
    cfg: DefenseConfig,
) -> Iterator[Dict[str, Any]]:
    """Yield NDJSON-ready events for one guarded generation.

    Event kinds: ``start``, ``token``, ``check``, ``refusal``, ``done``.
    """
    tok = state.tokenizer
    model = state.model
    eos_id = tok.eos_token_id

    with state.model_lock:
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=state.device)

        with torch.inference_mode():
            out = model(input_ids, use_cache=True, output_hidden_states=False)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

        yield {
            "event": "start",
            "model": state.model_name,
            "probe_layer": state.probe_layer,
            "hook_position": state.hook_position,
            "check_interval": cfg.check_interval,
            "threshold": cfg.threshold,
        }

        generated_ids: List[int] = []
        check_depths: List[int] = []
        check_probs: List[float] = []
        triggered = False
        trigger_depth: Optional[int] = None
        trigger_prob: Optional[float] = None

        for _ in range(cfg.max_new_tokens):
            next_id = sample_next_token(logits, cfg.temperature, cfg.top_p).view(1, 1)
            tid = int(next_id.item())
            if tid == eos_id:
                break

            generated_ids.append(tid)
            depth = len(generated_ids)
            yield {
                "event": "token",
                "text": tok.decode([tid], skip_special_tokens=False),
                "depth": depth,
            }

            # Advance the live cache by the committed token.
            with torch.inference_mode():
                out = model(next_id, past_key_values=past, use_cache=True,
                            output_hidden_states=False)
                past = out.past_key_values
                logits = out.logits[:, -1, :]

            # ADA-LP check at the window boundary.
            if cfg.check_interval > 0 and depth % cfg.check_interval == 0:
                prob = run_probe_check(state, past)
                check_depths.append(depth)
                check_probs.append(prob)
                flagged = prob >= cfg.threshold
                yield {"event": "check", "depth": depth,
                       "refusal_prob": prob, "flagged": flagged}
                if flagged:
                    triggered = True
                    trigger_depth = depth
                    trigger_prob = prob
                    yield {"event": "refusal", "depth": depth,
                           "refusal_prob": prob, "response": cfg.refusal_string}
                    break

        if triggered:
            response = cfg.refusal_string
        else:
            response = tok.decode(generated_ids, skip_special_tokens=True)

        yield {
            "event": "done",
            "triggered": triggered,
            "trigger_depth": trigger_depth,
            "refusal_prob": trigger_prob,
            "response": response,
            "check_depths": check_depths,
            "check_probabilities": check_probs,
        }


def run_generation(
    state: ServerState,
    messages: List[Dict[str, Any]],
    cfg: DefenseConfig,
) -> Dict[str, Any]:
    """Run a guarded generation and return the final (non-streaming) summary."""
    final: Dict[str, Any] = {}
    for event in stream_events(state, messages, cfg):
        if event["event"] == "done":
            final = event
    return final


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
def create_app(state: ServerState):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    class GenerateRequest(BaseModel):
        messages: List[Dict[str, Any]]
        max_new_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        threshold: Optional[float] = None
        check_interval: Optional[int] = None
        refusal_string: Optional[str] = None
        stream: bool = True

    def _resolve(req: "GenerateRequest") -> DefenseConfig:
        d = state.defaults
        return DefenseConfig(
            threshold=req.threshold if req.threshold is not None else d.threshold,
            check_interval=req.check_interval if req.check_interval is not None
            else d.check_interval,
            max_new_tokens=req.max_new_tokens if req.max_new_tokens is not None
            else d.max_new_tokens,
            temperature=req.temperature if req.temperature is not None else d.temperature,
            top_p=req.top_p if req.top_p is not None else d.top_p,
            refusal_string=req.refusal_string if req.refusal_string is not None
            else d.refusal_string,
        )

    app = FastAPI(title="Any-Depth Alignment — ADA-LP streaming demo")

    @app.post("/generate")
    def generate(req: GenerateRequest):
        cfg = _resolve(req)
        if req.stream:
            def ndjson() -> Iterator[str]:
                for event in stream_events(state, req.messages, cfg):
                    yield json.dumps(event, ensure_ascii=False) + "\n"

            return StreamingResponse(ndjson(), media_type="application/x-ndjson")
        return JSONResponse(run_generation(state, req.messages, cfg))

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "model": state.model_name,
            "probe_layer": state.probe_layer,
            "hook_position": state.hook_position,
            "check_interval": state.defaults.check_interval,
            "threshold": state.defaults.threshold,
        }

    return app


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="google/gemma-2-9b-it",
                        help="HF id; must be registered in configs/models.yaml.")
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES),
                        help="Model compute dtype.")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for the model (e.g. cuda:0, cpu).")
    parser.add_argument("--flash-attention", action=argparse.BooleanOptionalAction,
                        default=False, help="Use Flash Attention 2 if available.")

    # Probe checkpoint resolution (must match the training-time layout).
    parser.add_argument("--ckpt-dir", default="ckpts",
                        help="Root directory holding logistic probe checkpoints.")
    parser.add_argument("--mask-tokens", default=None,
                        help="Mask tokens used at collection time (path component).")
    parser.add_argument("--gradual-cache", action=argparse.BooleanOptionalAction,
                        default=True, help="Gradual-cache path component of the probe.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed used in the probe checkpoint path.")

    # Defense defaults (overridable per request).
    parser.add_argument("--check-interval", type=int, default=25,
                        help="Run the ADA-LP probe every N generated tokens.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Harmful-probability threshold to halt and refuse.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy; >0 enables nucleus sampling.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--refusal-string", default=DEFAULT_REFUSAL,
                        help="Text substituted when the probe flags the stream.")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    defaults = DefenseConfig(
        threshold=args.threshold,
        check_interval=args.check_interval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        refusal_string=args.refusal_string,
    )

    state = ServerState(
        args.model,
        dtype=args.dtype,
        device=args.device,
        use_flash_attention=args.flash_attention,
        ckpt_dir=args.ckpt_dir,
        mask_tokens=args.mask_tokens,
        gradual_cache=args.gradual_cache,
        seed=args.seed,
        defaults=defaults,
    )

    import uvicorn

    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
