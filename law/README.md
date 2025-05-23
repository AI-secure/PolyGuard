# Policy Red‑Teaming Pipeline for Legal Area

This repository implements a **multi‑stage workflow** for extracting, rephrasing, evaluating, and adversarially attacking LLM guardrail policies in the legal domain.

## Directory layout

```
├── extract_generate.py            # extract bullets & example requests from PDF
├── rephrase.py                    # rephrase explicit malicious prompts
├── collect_malicious_response.py  # collect answers to rephrased malicious prompts
├── collect_benign_response.py     # collect answers to paired benign prompts
├── eval.py                        # evaluate guardrail models on inputs/outputs
├── attack.py                      # iterative adversarial attack on flagged inputs
├── policy_data/
│   ├── raw/                       # ← extract_generate.py outputs here
│   ├── rephrased/                 # ← rephrase.py outputs here
│   └── output/                    # ← both *collect_* scripts write here
├── guardrail_results/
│   ├── input/                     # ← eval.py “input”‑mode results
│   └── output/                    # ← eval.py “output”‑mode results
└── attack_results/                # ← attack.py outputs here
```

## Prerequisites

- Python ≥ 3.9
- Set your OpenAI key:

```
export OPENAI_API_KEY="sk‑…"
```

## 1. Extract & cluster policy bullets (`extract_generate.py`)

Splits a regulatory PDF into sections, locates every “LLMs should not …” bullet, groups them into categories, and generates **5 malicious / 5 benign example requests** per policy.

```
python extract_generate.py \
  --name aba \
  --interval 4 \
  --model_name o4-mini-2025-04-16 \
  --batch_window 24h \
  --poll_interval 30
```

→ writes `**policy_data/raw/aba.csv**`

## 2. Rephrase malicious requests (`rephrase.py`)

Takes each explicit malicious request and rephrases it in an innocuous, “positive” style—while preserving the underlying intent—so that guardrail models see a less obvious threat.

```
python rephrase.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ writes `**policy_data/rephrased/<stem>.csv**`

## 3. Collect *malicious* model responses (`collect_malicious_response.py`)

For every **rephrased malicious prompt**, queries the target LLM and stores its answer (skipping replies containing “sorry”, “cannot”, or “can’t”).

```
python collect_malicious_response.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ updates/creates `**policy_data/output/<stem>.csv**` (adds `rephrased_malicious_example_answer_*` columns)

## 4. Collect *benign* model responses (`collect_benign_response.py`)

Fills in `benign_example_answer_{1..5}` **only for rows whose malicious answer is already present**—ensuring aligned comparison.

```
python collect_benign_response.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ writes back to the **same** `**policy_data/output/<stem>.csv**` files, adding `benign_example_answer_*` columns.

## 5. Evaluate guardrail performance (`eval.py`)

Runs each guardrail model in two modes:

- **input‑mode** – classify the raw prompts (malicious‑rephrased + benign)
- **output‑mode** – classify the LLM‑generated answers (malicious + benign)

```
# input‑mode
python eval.py --name aba --evaluate-input

# output‑mode
python eval.py --name aba
```

→ results saved to `**guardrail_results/input/aba.csv**` and `**guardrail_results/output/aba.csv**`

## 6. Adversarial attack on flagged inputs (`attack.py`)

Selects the rephrased malicious inputs that *all* guardrail models flagged, then iteratively refines an adversarial wrapper prompt (up to 10 rounds) attempting to bypass every guardrail.

```
python attack.py --name aba
```

→ incremental records written to `**attack_results/aba.csv**`

## Tips & Notes

- **Idempotent**: every script resumes from existing output—re‑running will not repeat finished work.
- **Tunable**: model, batch window, polling interval, and retry count are exposed via `--model-name`, `--batch-window`, `--poll-interval`, and `--max-attempts`.
- The *raw* CSV holds explicit malicious examples; the *rephrased* CSV swaps in semantically identical but more innocuous wording; the *output* CSV contains both malicious and benign answers once Steps 3 & 4 complete.
- Final guardrail metrics live under `guardrail_results/`. Adversarial bypass attempts live under `attack_results/`.