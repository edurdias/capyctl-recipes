#!/usr/bin/env python3
"""Greedy next-token agreement with BF16, without activations.

  gen:   the BF16 model decodes each prompt greedily; its prompt and answer token ids are saved.
  score: a build reads each BF16 prompt + answer as one input (prompt_logprobs=1) and, at every answer
         position, gives the token it would pick next. Agreement is the share of positions where that
         is BF16's own token: each position is a greedy decision on the same prefix.

usage: next_token_agree.py gen   --model <bf16 dir> --prompts agreement-prompts.json --out bf16-answers.json
       next_token_agree.py score --model <dir> --answers bf16-answers.json --name <label> --out <label>.json

Same engine settings for every build (vLLM, offline): no drafter, BF16 KV cache, the DeltaNet state in
the checkpoint's own dtype, no prefix caching (prompt_logprobs needs it off).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time


def engine(model: str):
    from vllm import LLM
    return LLM(model=os.path.expanduser(model), max_model_len=4096, gpu_memory_utilization=0.62,
               enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)


def cmd_gen(a) -> int:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    prompts = json.load(open(a.prompts))["prompts"][: a.limit or None]
    llm = engine(a.model)
    tok = llm.get_tokenizer()
    # Render the template as text, then encode it: vLLM's tokenizer wrapper returned text for
    # tokenize=True, so the "ids" were characters.
    texts = [tok.apply_chat_template(p["messages"], tools=p.get("tools"), add_generation_prompt=True, tokenize=False)
             for p in prompts]
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    assert all(isinstance(x, int) for x in ids[0]), "token ids expected"
    t = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids],
                        SamplingParams(temperature=0, max_tokens=a.max_tokens))
    recs = [{"id": p["id"], "category": p["category"], "prompt_ids": list(i), "answer_ids": list(o.outputs[0].token_ids)}
            for p, i, o in zip(prompts, ids, outs)]
    json.dump({"model": a.model, "max_tokens": a.max_tokens, "seconds": round(time.time() - t, 1), "records": recs},
              open(a.out, "w"))
    print(f"wrote {a.out}: {len(recs)} answers, {sum(len(r['answer_ids']) for r in recs)} tokens")
    return 0


def cmd_score(a) -> int:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    ans = json.load(open(a.answers))["records"]
    llm = engine(a.model)
    t = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["answer_ids"]) for r in ans],
                        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
    recs = []
    for r, o in zip(ans, outs):
        n0 = len(r["prompt_ids"])
        ranks, gaps = [], []
        for k, tid in enumerate(r["answer_ids"]):
            lp = o.prompt_logprobs[n0 + k]  # the distribution that predicts answer token k
            mine = lp[tid]
            ranks.append(mine.rank)
            top = max(lp.values(), key=lambda x: x.logprob)
            gaps.append(round(top.logprob - mine.logprob, 4))  # 0 when the build picks BF16's token
        agree = [x == 1 for x in ranks]
        first = next((k for k, ok in enumerate(agree) if not ok), len(agree))
        recs.append({"id": r["id"], "category": r["category"], "n": len(agree), "agree": sum(agree),
                     "first_disagreement": first, "ranks": ranks, "logprob_gap": gaps})
    tot = sum(x["n"] for x in recs)
    summ = {"positions": tot, "agree": sum(x["agree"] for x in recs),
            "agreement": round(sum(x["agree"] for x in recs) / tot, 5),
            "median_first_disagreement": st.median(x["first_disagreement"] for x in recs),
            "by_category": {c: round(sum(x["agree"] for x in recs if x["category"] == c)
                                     / sum(x["n"] for x in recs if x["category"] == c), 5)
                            for c in sorted({x["category"] for x in recs})}}
    json.dump({"model": a.model, "name": a.name, "seconds": round(time.time() - t, 1), "summary": summ, "records": recs},
              open(a.out, "w"))
    print(f"{a.name}: {json.dumps(summ)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--answers", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_gen(a) if a.cmd == "gen" else cmd_score(a)


if __name__ == "__main__":
    raise SystemExit(main())
