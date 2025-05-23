# Policy Red‑Teaming Pipeline for **Financial** Area

This repository implements a **multi‑stage workflow** for extracting, rephrasing, evaluating, and adversarially attacking LLM guardrail policies drawn from financial regulators (e.g., FINRA, BIS, OECD, …).

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
│   ├── input/                     # ← eval.py “input”-mode results
│   └── output/                    # ← eval.py “output”-mode results
└── attack_results/                # ← attack.py outputs here
```

## Prerequisites

- Python ≥ 3.9
- Export your OpenAI key:

```
export OPENAI_API_KEY="sk-…"
```

## 1. Extract & cluster policy bullets (`extract_generate.py`)

Splits a regulatory PDF into sections, finds every "LLMs should not …" bullet, clusters them into high‑level categories, and generates **5 malicious / 5 benign** example requests per policy.

```
python extract_generate.py \
  --name finra \
  --interval 4 \
  --model_name o4-mini-2025-04-16 \
  --batch_window 24h \
  --poll_interval 30
```

→ writes `**policy_data/raw/finra.csv**`

## 2. Rephrase malicious requests (`rephrase.py`)

Transforms each explicit malicious request into a more innocuous, “positive” variant that retains the same intent—making it harder for guardrails to detect.

```
python rephrase.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ writes `**policy_data/rephrased/<stem>.csv**`

## 3. Collect *malicious* model responses (`collect_malicious_response.py`)

Queries the LLM with each rephrased malicious prompt and stores the answer, skipping any reply containing “sorry”, “cannot”, or “can’t”.

```
python collect_malicious_response.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ updates/creates `**policy_data/output/<stem>.csv**` (adds `rephrased_malicious_example_answer_*` columns)

## 4. Collect *benign* model responses (`collect_benign_response.py`)

Fills `benign_example_answer_{1..5}` **only** for rows that already have a malicious answer, ensuring a one‑to‑one comparison.

```
python collect_benign_response.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ writes back to the **same** `**policy_data/output/<stem>.csv**` files.

## 5. Evaluate guardrail performance (`eval.py`)

Assess each guardrail model in two modes:

- **input‑mode** – classify the prompts (malicious‑rephrased + benign)
- **output‑mode** – classify the LLM answers (malicious + benign)

```
# input‑mode
a python eval.py --name finra --evaluate-input

# output‑mode
python eval.py --name finra
```

→ results in `**guardrail_results/input/finra.csv**` and `**guardrail_results/output/finra.csv**`

## 6. Adversarial attack on flagged inputs (`attack.py`)

Picks the rephrased malicious prompts that all guardrails flagged and iteratively crafts a wrapper prompt (up to 10 rounds) to flip every guardrail to **safe**.

```
python attack.py --name finra
```

→ attack logs written to `**attack_results/finra.csv**`

## Tips & Notes

- **Resumable** – each script detects existing output and skips finished work.
- **Customisable** – model, batch window, poll interval, and retries via `--model-name`, `--batch-window`, `--poll-interval`, and `--max-attempts`.
- The *raw* CSV holds explicit malicious examples; *rephrased* swaps in innocuous wording; *output* gains malicious and benign answers after Steps 3 & 4.
- Final guardrail metrics live in `guardrail_results/`; adversarial bypass attempts in `attack_results/`.