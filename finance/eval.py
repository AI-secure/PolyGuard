#!/usr/bin/env python3
# coding: utf-8
"""
Evaluate guardrail models on malicious/benign examples from a policy CSV.

Input  : policy_data/output/<name>.csv
Output : guardrail_results/<mode>/<name>.csv, where mode is “input” or “output”

• Resumes from existing results if present
• Use --evaluate-input to test inputs (rephrased malicious & original benign);
  otherwise tests outputs (rephrased answers & original benign)
"""

import setGPU
import os
import argparse
import pathlib
import pandas as pd
import torch
from guardrail_model import get_guardrail


MODEL_IDS = [
    "meta-llama/Llama-Guard-4-12B",
    "meta-llama/Llama-Guard-3-8B",
    "meta-llama/Meta-Llama-Guard-2-8B",
    "meta-llama/Llama-Guard-3-1B",
    "meta-llama/LlamaGuard-7b",
    "google/shieldgemma-2b",
    "google/shieldgemma-9b",
    "text-moderation-latest",
    "omni-moderation-latest",
    "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b",
    "OpenSafetyLab/MD-Judge-v0.1",
    "allenai/wildguard",
    "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0",
    "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    "ibm-granite/granite-guardian-3.2-3b-a800m",
    "ibm-granite/granite-guardian-3.2-5b",
    "llmjudge_gpt_4o",
    "azure",
    "aws",
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate guardrail models on policy examples"
    )
    parser.add_argument(
        "--name", required=True,
        help="Base name of the policy CSV (no extension)"
    )
    parser.add_argument(
        "--evaluate-input", action="store_true",
        help="If set, evaluate inputs (rephrased malicious & benign examples); "
             "otherwise evaluate outputs (rephrased answers & benign examples)"
    )
    args = parser.parse_args()
    evaluate_input = args.evaluate_input

    # Load the policy CSV
    csv_path = pathlib.Path("policy_data") / "output" / f"{args.name}.csv"
    df = pd.read_csv(csv_path)

    # Prepare output directory and file
    mode = "input" if evaluate_input else "output"
    out_dir = pathlib.Path("guardrail_results") / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / f"{args.name}.csv"

    # Resume existing results if present
    if agg_path.exists():
        print(f"Resuming from existing results: {agg_path}")
        existing = pd.read_csv(agg_path)
        all_results = existing.to_dict("records")
        done_models = set(existing["model_id"].unique())
    else:
        all_results = []
        done_models = set()

    client = None
    for model_index, model_id in enumerate(MODEL_IDS):
        if model_id in done_models:
            print(f"Skipping already-done model '{model_id}'")
            continue

        print(f"\n=== [{model_index}] Evaluating model '{model_id}' ===")
        model = get_guardrail(model_id)

        for row_index, row in df.iterrows():
            for example_type in ("malicious", "benign"):
                for i in range(1, 6):
                    # select column & prompt based on mode
                    if evaluate_input:
                        if example_type == "malicious":
                            col = f"rephrased_malicious_example_{i}"
                        else:
                            col = f"benign_example_{i}"
                        prompt = row.get(col, "")
                        if not isinstance(prompt, str) or not prompt.strip():
                            continue
                        resp = model.moderate(prompt)
                    else:
                        if example_type == "malicious":
                            user_col = f"rephrased_malicious_example_{i}"
                            ans_col  = f"rephrased_malicious_example_answer_{i}"
                            user  = row.get(user_col, "")
                            reply = row.get(ans_col, "")
                            if not isinstance(reply, str) or not reply.strip():
                                continue
                            resp = model.moderate([
                                {"role": "user",      "content": user},
                                {"role": "assistant", "content": reply}
                            ])
                            col = ans_col
                        else:
                            col = f"benign_example_{i}"
                            prompt = row.get(col, "")
                            if not isinstance(prompt, str) or not prompt.strip():
                                continue
                            resp = model.moderate(prompt)

                    flagged = bool(getattr(resp, "flagged", False))
                    print(f"[{model_id}][row {row_index}][{col}]: flagged={flagged}")
                    all_results.append({
                        "model_index": model_index,
                        "model_id":    model_id,
                        "row_index":   row_index,
                        "example_type": example_type,
                        "example_num":  i,
                        "flagged":      flagged
                    })

        # clean up GPU between models
        del model
        torch.cuda.empty_cache()

        # save aggregated results after each model
        pd.DataFrame(all_results).to_csv(agg_path, index=False)
        print(f"Saved progress to {agg_path}")

    print(f"\nAll done. Final results in {agg_path}")


if __name__ == "__main__":
    main()
