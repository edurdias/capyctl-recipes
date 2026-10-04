#!/usr/bin/env python3
"""Build a chat + code calibration JSONL (a `messages` column, read by ModelOpt hf_ptq.py
through the model's chat template) from two public, ungated datasets:

  chat: HuggingFaceH4/ultrachat_200k (MIT)
        data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet  (columns prompt, prompt_id, messages)
  code: ise-uiuc/Magicoder-OSS-Instruct-75K (MIT)
        data-oss_instruct-decontaminated.jsonl                  (columns problem, solution, lang)

Rows are taken at an even stride over each file, so the set is deterministic for a given file.
"""

import argparse
import json

import pyarrow.parquet as pq


def chat_rows(path, n, min_chars):
    t = pq.read_table(path, columns=["prompt_id", "messages"]).to_pylist()
    stride = max(1, len(t) // (n * 2))
    out = []
    for r in t[::stride]:
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
        t = [json.loads(line) for line in f]
    stride = max(1, len(t) // (n * 2))
    out = []
    for r in t[::stride]:
        if not r.get("problem") or not r.get("solution") or len(r["problem"]) + len(r["solution"]) < min_chars:
            continue
        msgs = [{"role": "user", "content": r["problem"]}, {"role": "assistant", "content": r["solution"]}]
        out.append({"messages": msgs, "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                    "id": r.get("index"), "lang": r.get("lang")})
        if len(out) == n:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chat", required=True, help="ultrachat_200k test_sft parquet")
    p.add_argument("--code", required=True, help="Magicoder-OSS-Instruct-75K jsonl")
    p.add_argument("--per-split", type=int, default=128)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = chat_rows(a.chat, a.per_split, a.min_chars) + code_rows(a.code, a.per_split, a.min_chars)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
