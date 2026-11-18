#!/usr/bin/env python3
"""Reproduce the Qwen precedent's 256 UltraChat + 256 Magicoder messages.

Gemma's source tokenizer applies its chat template during ModelOpt calibration;
this JSONL deliberately stores messages, not a Qwen-rendered text string.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED_SHA256 = "a3e005c39c3c71918ffff53504091ed32bd02baf3d37241072e5a616b0dc31eb"


def chat_rows(path, n, min_chars):
    rows = pq.read_table(path, columns=["prompt_id", "messages"]).to_pylist()
    stride = max(1, len(rows) // (n * 2))
    out = []
    for r in rows[::stride]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]
                if m.get("content") and m["role"] in ("system", "user", "assistant")]
        if (len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant"
                or sum(len(m["content"]) for m in msgs) < min_chars):
            continue
        out.append({"messages": msgs, "source": "HuggingFaceH4/ultrachat_200k", "id": r["prompt_id"]})
        if len(out) == n:
            break
    return out


def code_rows(path, n, min_chars):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    stride = max(1, len(rows) // (n * 2))
    out = []
    for r in rows[::stride]:
        if not r.get("problem") or not r.get("solution") or len(r["problem"]) + len(r["solution"]) < min_chars:
            continue
        msgs = [{"role": "user", "content": r["problem"]}, {"role": "assistant", "content": r["solution"]}]
        out.append({"messages": msgs, "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                    "id": r.get("index"), "lang": r.get("lang")})
        if len(out) == n:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chat", required=True, help="UltraChat test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet")
    p.add_argument("--code", required=True, help="Magicoder data-oss_instruct-decontaminated.jsonl")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    chat, code = chat_rows(a.chat, 256, 400), code_rows(a.code, 256, 400)
    assert len(chat) == len(code) == 256
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chat + code).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    assert digest == EXPECTED_SHA256, f"Dataset differs from precedent: {digest}"
    a.out.write_bytes(data)
    print(f"512 conversations; SHA-256 {digest}")


if __name__ == "__main__":
    main()
