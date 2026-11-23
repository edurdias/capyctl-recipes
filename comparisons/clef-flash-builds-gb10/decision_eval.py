#!/usr/bin/env python3
"""Ordered distribution capture and laptop timing. Dataset text stays private."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "capyctl-bench"))
import capyctl_bench as bench

PRIVATE = Path.home() / "clef-decision-eval"


def encoded(payload):
    # Insertion order is part of the request contract, including the question map.
    return json.dumps(payload, ensure_ascii=False).encode()


def digest(payload):
    return hashlib.sha256(encoded(payload)).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare():
    PRIVATE.mkdir(exist_ok=True)
    diagnostic = json.loads(Path("/var/tmp/clef16/eval.json").read_text())
    assert len(diagnostic) == 41 and sum(len(r["questions"]) for r in diagnostic) == 104
    shutil.copyfile("/var/tmp/clef16/eval.json", PRIVATE / "diagnostic-104.json")
    heldout = []
    cache = PRIVATE / "datasets"
    for name, spec in bench.ACCURACY_SETS.items():
        if spec["kind"] == "di":
            _, rows = bench.load_di_rows(name, cache)
            for row in rows:
                heldout.append({"id": f"{name}:{row['id']}", "suite": name, "state": row["state"], "questions": row["questions"]})
        else:
            rows = bench.load_dataset(name, "https://datasets-server.huggingface.co", cache)
            if spec["kind"] == "typed":
                for row in rows:
                    heldout.append({"id": f"{name}:{row['id']}", "suite": name,
                                    "state": json.loads(row["state"]), "questions": json.loads(row["questions"])})
            else:
                for i in bench.sample_indices(len(rows), 500, 20261121, name):
                    state, q, _ = bench.accuracy_request(spec, rows[i])
                    heldout.append({"id": f"{name}:{i}", "suite": name, "state": state, "questions": {"answer": q}})
    # No held-out rows are used to calibrate quantization or choose candidates.
    diagnostic_hashes = {digest({"state": r["state"], "questions": r["questions"]}) for r in diagnostic}
    assert not diagnostic_hashes.intersection(digest({"state": r["state"], "questions": r["questions"]}) for r in heldout)
    save(PRIVATE / "heldout.json", heldout)
    manifest = {"diagnostic": {"requests": 41, "questions": 104, "sha256": hashlib.sha256((PRIVATE / "diagnostic-104.json").read_bytes()).hexdigest()},
                "heldout": {"requests": len(heldout), "questions": sum(len(r["questions"]) for r in heldout),
                            "sha256": hashlib.sha256((PRIVATE / "heldout.json").read_bytes()).hexdigest(), "selection_seed": 20261121,
                            "sources": {name: {k: v for k, v in spec.items() if k in ("repo", "revision", "sha256", "source", "kind", "split")} for name, spec in bench.ACCURACY_SETS.items()}},
                "disjoint_from_diagnostic": True, "calibration_or_selection_use": False,
                "redistribution": "Request text and dataset caches remain private; publish hashes, row identifiers and outputs only."}
    for source in manifest["heldout"]["sources"].values():
        if "source" in source:
            source["source"] = {k: v for k, v in source["source"].items() if k != "password"}
    assert manifest["heldout"]["questions"] >= 2000
    save(HERE / "cohort-manifest.json", manifest)
    print(json.dumps({"requests": len(heldout), "questions": manifest["heldout"]["questions"]}), flush=True)


def capture(ep, cohort, output):
    rows = json.loads((PRIVATE / cohort).read_text())
    with output.open("w") as stream:
        for i, row in enumerate(rows):
            payload = {"model": "clef-flash", "state": row["state"], "questions": row["questions"]}
            rec = bench.post_decision(ep, payload, keep_answers=True)
            rec.update({"id": row["id"], "suite": row.get("suite", "diagnostic"), "payload_sha256": digest(payload),
                        "question_order": list(row["questions"])})
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stream.flush()
            if rec["error"]:
                raise RuntimeError(f"{row['id']}: {rec['error']}")
            if (i + 1) % 50 == 0:
                print(f"{cohort}: {i + 1}/{len(rows)}", flush=True)


def measure(tag, timing=True):
    ep = bench.Endpoint(os.environ["CLEF_ENDPOINT"])
    result_dir = HERE / "results"
    result_dir.mkdir(exist_ok=True)
    capture(ep, "diagnostic-104.json", result_dir / f"{tag}-diagnostic.jsonl")
    capture(ep, "heldout.json", result_dir / f"{tag}-heldout.jsonl")
    if not timing:
        return
    for rnd in range(1, 4):
        points = []
        seed = 9000000 + rnd * 100000
        calibration = {}
        for qn in (1, 3, 5):
            questions = bench.latency_questions(qn)
            probes = [bench.post_decision(ep, {"model": "clef-flash", "state": bench.filler_text(n, seed + n), "questions": questions}) for n in (1, 100)]
            if any(p["error"] for p in probes):
                raise RuntimeError(probes)
            per_sentence = (probes[1]["input_tokens"] - probes[0]["input_tokens"]) / 99
            overhead = probes[0]["input_tokens"] - per_sentence
            calibration[qn] = (per_sentence, overhead)
            for target in (200, 2000, 8000, 15800):
                n = max(1, round((target - overhead) / per_sentence))
                reqs = []
                for i in range(105):
                    seed += 1
                    rec = bench.post_decision(ep, {"model": "clef-flash", "state": bench.filler_text(n, seed), "questions": questions})
                    if rec["error"]:
                        raise RuntimeError(rec["error"])
                    if i >= 5:
                        reqs.append(rec)
                point = {"round": rnd, "target_total_tokens": target, "questions": qn, "sentences": n,
                         "schema_overhead_tokens_estimate": overhead, "requests": reqs, "summary": bench.summarize_decisions(reqs)}
                points.append(point)
                save(result_dir / f"{tag}-latency-round{rnd}.json", points)
                print(f"round {rnd}: {qn}q target {target}: {point['summary']}", flush=True)
        levels = []
        questions = bench.latency_questions(3)
        per_sentence, overhead = calibration[3]
        n = max(1, round((2000 - overhead) / per_sentence))
        for c in (1, 2, 4, 8):
            def one(state):
                return bench.post_decision(ep, {"model": "clef-flash", "state": state, "questions": questions})
            with ThreadPoolExecutor(max_workers=c) as pool:
                list(pool.map(one, [bench.filler_text(n, seed + i) for i in range(max(c, 5))]))
                seed += max(c, 5)
                states = [bench.filler_text(n, seed + i) for i in range(200)]
                seed += 200
                reqs = list(pool.map(one, states))
            level = bench.summarize_throughput({"concurrency": c, "requests": reqs})
            levels.append(level)
            save(result_dir / f"{tag}-throughput-round{rnd}.json", levels)
            print(f"round {rnd}: concurrency {c}: {level['summary']}", flush=True)
            if level["summary"]["errors"]:
                raise RuntimeError("throughput errors")

def distribution(answer):
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


def compare(reference, candidate):
    import math
    ref = {r["id"]: r for r in map(json.loads, reference.read_text().splitlines())}
    cand = {r["id"]: r for r in map(json.loads, candidate.read_text().splitlines())}
    assert ref.keys() == cand.keys()
    rows = []
    for rid, row in cand.items():
        base = ref[rid]
        assert not row["error"] and not base["error"]
        assert row["payload_sha256"] == base["payload_sha256"] and row["question_order"] == base["question_order"]
        for qid, answer in row["answers"].items():
            p, q = distribution(base["answers"][qid]), distribution(answer)
            assert p.keys() == q.keys()
            rows.append({"id": rid, "question": qid, "same_top": max(p, key=p.get) == max(q, key=q.get),
                         "max_gap": max(abs(p[k] - q[k]) for k in p), "tv": sum(abs(p[k] - q[k]) for k in p) / 2,
                         "kl": sum(p[k] * math.log(p[k] / max(q[k], 1e-9)) for k in p if p[k] > 0)})
    print(json.dumps({"questions": len(rows), "same_top": sum(r["same_top"] for r in rows),
                      "gap_p50": bench.percentile([r["max_gap"] for r in rows], 50),
                      "gap_p95": bench.percentile([r["max_gap"] for r in rows], 95),
                      "gap_max": max(r["max_gap"] for r in rows), "mean_kl": bench.mean(r["kl"] for r in rows),
                      "mean_tv": bench.mean(r["tv"] for r in rows), "flips": [r for r in rows if not r["same_top"]]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "capture", "measure", "compare"))
    parser.add_argument("--tag", default="laptop-fp8-sglang0522")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode in ("capture", "measure"):
        measure(args.tag, timing=args.mode == "measure")
    else:
        compare(args.reference, args.candidate)
