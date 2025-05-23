#!/usr/bin/env python3
# coding: utf-8
"""
Populate benign_example_answer_{1..5} columns in policy_data/output/*.csv.

Input  : policy_data/output/*.csv   (already contain malicious answers)
Output : policy_data/output/*.csv   (same files, now with benign answers)

• Model, batch window, poll interval, and max attempts configurable via argparse
• Answers skipped if they contain “sorry”, “cannot”, or “can’t” (case-insensitive)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import pathlib
from typing import List, Dict, Tuple

import pandas as pd
from openai import OpenAI

# ───────── constants ─────────
BAN_RE        = re.compile(r"\b(?:sorry|cannot|can[’']t)\b", re.I)
ANSWER_PROMPT = "{req}"   # simple pass-through; wrap here if desired


# ───────── helpers (kept identical in spirit to the malicious script) ─────────
def _build_jsonl_line(custom_id: str, content: str, model: str) -> str:
    body = {"model": model,
            "messages": [{"role": "user", "content": content}]}
    return json.dumps({
        "custom_id": custom_id,
        "method":   "POST",
        "url":      "/v1/chat/completions",
        "body":     body,
    }, ensure_ascii=False)


def _launch_batch(
    client: OpenAI,
    requests: List[str],
    batch_window: str,
    poll_interval: int
) -> List[str]:
    tmpdir  = pathlib.Path(tempfile.mkdtemp(prefix="batch_benign_"))
    in_path = tmpdir / "input.jsonl"
    in_path.write_text("\n".join(requests), encoding="utf-8")

    up   = client.files.create(file=open(in_path, "rb"), purpose="batch")
    bjob = client.batches.create(
        input_file_id     = up.id,
        endpoint          = "/v1/chat/completions",
        completion_window = batch_window,
        metadata          = {"description": "Generate benign answers"},
    )
    print("Launched batch", bjob.id)

    while bjob.status not in {"completed", "failed", "expired", "cancelled"}:
        time.sleep(poll_interval)
        bjob = client.batches.retrieve(bjob.id)
        print(f"[{time.strftime('%X')}] {bjob.status} "
              f"(done={bjob.request_counts.completed})")

    if bjob.status != "completed":
        raise RuntimeError(f"Batch ended with status {bjob.status}")

    return client.files.content(bjob.output_file_id).text.splitlines()


def _fill_benign_answers(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    batch_window: str,
    poll_interval: int,
    max_attempts: int,
) -> None:
    mal_ans_cols = [f"rephrased_malicious_example_answer_{k}" for k in range(1, 6)]
    ben_req_cols = [f"benign_example_{k}"                      for k in range(1, 6)]
    ben_ans_cols = [f"benign_example_answer_{k}"               for k in range(1, 6)]

    # make sure benign answer columns exist
    for col in ben_ans_cols:
        if col not in df.columns:
            df[col] = pd.NA

    def remaining() -> List[Tuple[int, int, str]]:
        """Return (row_idx, k, benign_request_text) still needing answers."""
        for idx, row in df.iterrows():
            for k in range(1, 6):
                mal_ans = str(row[mal_ans_cols[k - 1]]).strip()
                ben_req = str(row[ben_req_cols[k - 1]]).strip()
                ben_ans = row[ben_ans_cols[k - 1]]
                if (
                    ben_req
                    and mal_ans not in {"", "None", pd.NA}          # malicious answer exists
                    and (pd.isna(ben_ans) or ben_ans in {"", "None"})  # benign answer missing
                ):
                    yield idx, k, ben_req

    for attempt in range(1, max_attempts + 1):
        jobs = list(remaining())
        if not jobs:
            print("✓ all benign answers filled")
            return

        print(f"Attempt {attempt}: {len(jobs)} prompts")
        id2loc: Dict[str, Tuple[int, int]] = {}
        reqs: List[str] = []

        for idx, k, req in jobs:
            cid = f"{idx}_{k}"
            id2loc[cid] = (idx, k)
            reqs.append(_build_jsonl_line(cid, ANSWER_PROMPT.format(req=req), model))

        # launch batch, stream results back
        for line in _launch_batch(client, reqs, batch_window, poll_interval):
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            if cid not in id2loc:
                continue
            i, k = id2loc[cid]

            resp = obj.get("response", {}).get("body", {})
            if obj.get("error") or resp.get("error"):
                continue
            text = resp["choices"][0]["message"]["content"].strip()
            df.at[i, ben_ans_cols[k - 1]] = None if BAN_RE.search(text) else text


# ───────── main ─────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate benign answers for rows that already have malicious answers"
    )
    ap.add_argument("--model-name",  type=str, default="o4-mini-2025-04-16",
                    help="OpenAI model to use")
    ap.add_argument("--batch-window", type=str, default="24h",
                    help="Batch-API completion window")
    ap.add_argument("--poll-interval", type=int, default=30,
                    help="Seconds between batch-status polls")
    ap.add_argument("--max-attempts", type=int, default=10,
                    help="Max retry attempts per missing benign answer")
    ap.add_argument("--output-dir", type=pathlib.Path,
                    default=pathlib.Path("policy_data") / "output",
                    help="Directory containing *_response.csv files")
    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    csv_files = sorted(args.output_dir.glob("*.csv"))
    if not csv_files:
        print("No CSVs found in", args.output_dir)
        return

    for csv_path in csv_files:
        print("\n—", csv_path.name)
        df = pd.read_csv(csv_path)
        _fill_benign_answers(
            df,
            client,
            args.model_name,
            args.batch_window,
            args.poll_interval,
            args.max_attempts,
        )
        df.to_csv(csv_path, index=False)
        print("  ⭑ written", csv_path.name)


if __name__ == "__main__":
    main()
