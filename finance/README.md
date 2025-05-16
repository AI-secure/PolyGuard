# Policy Red-Teaming Pipeline for Financial Area

This repository implements a multi-stage workflow for extracting, rephrasing, evaluating, and adversarially attacking LLM guardrail policies.

## Directory layout

```
├── extract_generate.py      # extract bullets & example requests from PDF
├── rephrase.py              # rephrase explicit malicious prompts
├── collect_output.py        # collect LLM-generated answers
├── eval.py                  # evaluate guardrail models on inputs/outputs
├── attack.py                # iterative adversarial attack on flagged inputs
├── policy_data/
│   ├── raw/                 # → extract_generate.py outputs here
│   ├── rephrased/           # → rephrase.py outputs here
│   └── output/              # → collect_output.py outputs here
└── guardrail_results/
    ├── input/               # → eval.py “input” mode results
    └── output/              # → eval.py “output” mode results
└── attack_results/  # → attack.py outputs here
```

## Prerequisites

- Python 3.9+

- Set your OpenAI key in the environment:

  ```
  export OPENAI_API_KEY="sk-…"
  ```

## 1. Extract & cluster policy bullets (`extract_generate.py`)

Splits a regulatory PDF into sections, locates every “LLMs should not…” bullet, groups them into categories, and generates 5 malicious/benign example requests per policy.

```
python extract_generate.py \
  --name finra \
  --interval 4 \
  --model_name o4-mini-2025-04-16 \
  --batch_window 24h \
  --poll_interval 30
```

→ writes `policy_data/raw/finra.csv`

## 2. Rephrase malicious requests (`rephrase.py`)

Takes each explicit malicious request (to illustrate intent) and reframes it more “positive” so that guardrail models see it as innocuous—but preserving the same underlying instruction.

```
python rephrase.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ writes `policy_data/rephrased/<stem>.csv`

## 3. Collect model outputs (`collect_output.py`)

For each rephrased prompt, collects the LLM’s **answers** (used later in “output”-mode evaluation).

```
python collect_output.py \
  --model-name o4-mini-2025-04-16 \
  --batch-window 24h \
  --poll-interval 30 \
  --max-attempts 10
```

→ writes `policy_data/output/<stem>.csv`

## 4. Evaluate guardrail performance (`eval.py`)

Runs each guardrail model over both:

- **input** mode: the rephrased malicious & original benign requests
- **output** mode: the LLM’s generated answers & original benign requests

```
# input-mode
python eval.py --name finra --evaluate-input

# output-mode
python eval.py --name finra
```

→ results saved under `guardrail_results/input/finra.csv`
 and `guardrail_results/output/finra.csv`

## 5. Adversarial attack on flagged inputs (`attack.py`)

Selects those rephrased malicious inputs that **all** target guardrail models flagged, then iteratively refines an adversarial “wrapper” prompt (up to 10 rounds) to try to force the guardrail to classify them as **safe**.

```
python attack.py --name finra
```

→ incremental attack records written to
 `rephrased_requests_attack/finra.csv`

------

**Tips & Notes**

- All scripts resume from any existing output files—re-running won’t repeat completed work.
- You can adjust the model, batch window, polling interval, and retry counts via each script’s `--model-name`, `--batch-window`, `--poll-interval`, and `--max-attempts` flags.
- Intermediate CSVs are under `policy_data/`; final performance metrics under `guardrail_results/`.
- The “raw” CSV shows original malicious examples, while “rephrased” replaces them with semantically identical but more innocuous wording for evaluation.