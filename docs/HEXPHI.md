# Getting the HEx-PHI deep-prefill split (license-compliant)

HEx-PHI is used in the deep-prefill experiment (E2) but is gated under the
[LLM-Tuning-Safety](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI)
license, which does not permit redistribution — so neither this repo nor the
published dataset ships the HEx-PHI prompts or the `hexphi_responses.jsonl` file
that embeds them.

You can still obtain **our exact continuations**, because the gated part (the
prompt) is something you provide from your own licensed HEx-PHI copy, and the
valuable part (our generated harmful continuation) is our own model output. We
bridge the two with a prompt-free **reference file**.

## Recommended: reconstruct our data from the references

The published dataset includes `eval/deep_prefill/hexphi_references.jsonl` —
records of `{hexphi_sha256, hexphi_index, response}`. Each holds a **one-way
SHA-256 of the prompt** (not the prompt) plus our continuation. Join it against
your own HEx-PHI copy to rebuild the full split:

```bash
# 1. Accept the HEx-PHI license, then authenticate.
huggingface-cli login                       # or: export HF_TOKEN=hf_...

# 2. Get the references (from the gated dataset repo).
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('javyduck/any-depth-alignment', 'eval/deep_prefill/hexphi_references.jsonl', \
repo_type='dataset', local_dir='data')"

# 3. Re-join with your local HEx-PHI (loads prompts via your accepted license).
python -m ada.datagen.hexphi_reference reconstruct
# -> data/eval/deep_prefill/hexphi_responses.jsonl  (identical to ours)
```

This moves **no HEx-PHI text** off the original source: prompts come from your
licensed copy, matched by hash. The result is byte-for-byte our continuations
paired with the prompts.

## Alternative: regenerate from scratch

If you prefer not to use our continuations, regenerate them:

```bash
python -m ada.datagen.gen_harmful_gpt --dataset hexphi \
    --gen-model <your-jailbroken-generator> --judge-model gpt-4o
```

The generator is a deliberately misaligned GPT fine-tuned via the OpenAI SFT API
(paper appendix, "Jailbreaking GPT via SFT"). Any source of long harmful
continuations to the HEx-PHI prompts works.

## Then evaluate

```bash
DATASETS="advbench jailbreakbench strongreject hexphi" bash scripts/20_e2_prefill.sh
```

## Producing the references (dataset owner)

From a machine holding the licensed `hexphi_responses.jsonl`:

```bash
python -m ada.datagen.hexphi_reference export     # -> hexphi_references.jsonl (prompt-free)
```

`scripts/upload_to_hf.py` runs this automatically before publishing, and its
`DATASET_IGNORE` excludes every prompt-bearing HEx-PHI file while keeping the
reference file. This is compliant because the references contain only irreversible
hashes and our own continuations — never HEx-PHI prompt text.
