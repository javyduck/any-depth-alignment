# Vendored adversarial-attack engines

The adversarial-attack experiments (§ Robustness Under Adversarial Prompt Attacks) run four
jailbreak attacks against each base model. Their search engines are third-party
research code, vendored here **unmodified** except for the first-party driver
scripts (`run_*.sh`, `run_*_attack.py` / `attack_*.py`) written for this paper and
the removal of hardcoded API keys (now read from `$OPENAI_API_KEY`).

| Directory | Attack | Upstream | License |
|-----------|--------|----------|---------|
| `nanogcg/` | GCG (gradient suffix optimization) | Gray Swan AI — nanoGCG | MIT |
| `autodan/` | AutoDAN (genetic prompt search) | Xiaogeng Liu — AutoDAN | MIT |
| `pair/`    | PAIR (iterative LLM refinement)  | PAIR Team | MIT |
| `tap/`     | TAP (tree-of-attacks w/ pruning) | Robust Intelligence | MIT |

Each subdirectory retains its original `LICENSE`. All four are MIT-licensed;
attribution and copyright notices are preserved. See each upstream project for
citation details.

## Reproducing adversarial-attack

1. Generate raw attacks: `scripts/adversarial_generate.sh` (drives the four engines
   over `data/eval/attack_prompts/{advbench,jailbreakbench}.csv`).
2. Judge + extract harmful pairs: `python -m ada.attacks.extract` (GPT-4o judge)
   → `data/eval/attacks/{dataset}_{attack}/<model>/responses.jsonl`.
3. Evaluate ADA + baselines on the attacked prompts: `scripts/adversarial_eval.sh`.

Requires `OPENAI_API_KEY` (AutoDAN/PAIR/TAP use OpenAI attacker/judge models).
