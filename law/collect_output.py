#!/usr/bin/env python3
# coding: utf-8
"""
Generate model answers for malicious examples in policy CSVs (Batch API).

Input  : policy_data/rephrased/*.csv  
Output : policy_data/output/<stem>.csv  

• Model, batch window, poll interval, and max attempts configurable via argparse  
• Answers stored only if they do not contain “sorry”, “cannot”, or “can’t” (case-insensitive); otherwise None  
• Progress saved after every batch; existing output CSVs are resumed  
"""

import argparse
import json
import os
import re
import time
import tempfile
import pathlib
import textwrap

from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ───────── helpers ─────────

BAN_RE = re.compile(r"\b(?:sorry|cannot|can[’']t)\b", re.I)
ANSWER_PROMPT = "{req}"  # pass-through by default


def _build_jsonl_line(custom_id: str, content: str, model: str) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }, ensure_ascii=False)


def _launch_batch(
    client: OpenAI,
    requests: List[str],
    batch_window: str,
    poll_interval: int
) -> List[str]:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="batch_ans_"))
    in_path = tmp / "input.jsonl"
    in_path.write_text("\n".join(requests), encoding="utf-8")

    up = client.files.create(file=open(in_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id     = up.id,
        endpoint          = "/v1/chat/completions",
        completion_window = batch_window,
        metadata          = {"description": "Generate answers"},
    )
    print("Launched batch", batch.id)

    while batch.status not in {"completed", "failed", "expired", "cancelled"}:
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        print(f"[{time.strftime('%X')}] {batch.status} (done={batch.request_counts.completed})")

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status {batch.status}")

    return client.files.content(batch.output_file_id).text.splitlines()


def _paths_for_output(reph_path: pathlib.Path) -> pathlib.Path:
    out_dir = pathlib.Path("policy_data") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{reph_path.stem}.csv"


def _answer_dataframe(
    df: pd.DataFrame,
    client: OpenAI,
    out_path: pathlib.Path,
    model: str,
    batch_window: str,
    poll_interval: int,
    max_attempts: int
) -> None:
    ans_cols = [f"rephrased_malicious_example_answer_{k}" for k in range(1, 6)]
    req_cols = [f"rephrased_malicious_example_{k}"        for k in range(1, 6)]
    for col in ans_cols:
        df.setdefault(col, pd.NA)

    def remaining() -> List[Tuple[int,int,str]]:
        mask = df[ans_cols].isna() | (df[ans_cols] == "") | (df[ans_cols] == "None")
        for idx, row in df[mask.any(axis=1)].iterrows():
            for k in range(1, 6):
                ans = row[ans_cols[k-1]]
                if pd.isna(ans) or ans in {"", "None"}:
                    req = str(row[req_cols[k-1]]).strip()
                    if req:
                        yield idx, k, req

    for attempt in range(1, max_attempts + 1):
        todo = list(remaining())
        if not todo:
            print("✓ all answers filled", out_path.name)
            break
        print(f"Attempt {attempt}: {len(todo)} prompts")

        id2loc: Dict[str, Tuple[int,int]] = {}
        reqs: List[str] = []
        for idx, k, req in todo:
            cid = f"{idx}_{k}"
            id2loc[cid] = (idx, k)
            content = ANSWER_PROMPT.format(req=req)
            reqs.append(_build_jsonl_line(cid, content, model))

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
            df.at[i, ans_cols[k-1]] = None if BAN_RE.search(text) else text

        df.to_csv(out_path, index=False)
        print("  ⭑ progress written", out_path.name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate model answers for rephrased malicious examples"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="o4-mini-2025-04-16",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--batch-window",
        type=str,
        default="24h",
        help="Batch API completion window"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between batch status polls"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Max retry attempts per missing example"
    )
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    reph_dir = pathlib.Path("policy_data") / "rephrased"
    files    = sorted(reph_dir.glob("*.csv"))
    if not files:
        print("No CSVs found in", reph_dir)
        return

    for reph in files:
        print("\n---", reph.name)
        out_path = _paths_for_output(reph)
        if out_path.exists():
            print("Resuming", out_path.name)
            df = pd.read_csv(out_path)
        else:
            df = pd.read_csv(reph)
        _answer_dataframe(
            df, client, out_path,
            args.model_name, args.batch_window,
            args.poll_interval, args.max_attempts
        )

if __name__ == "__main__":
    main()
