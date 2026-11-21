#!/usr/bin/env python3
"""Load one approved native engine, execute a short prompt, then stop.

Run only with the integration owner's clearance, no other session-a GPU work, and
QWEN36_LOAD_ENGINE/PYTHONPATH selecting the adjacent diagnostics module.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine", choices=["vllm", "sglang"])
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    assert os.getuid() != 0
    assert os.environ.get("QWEN36_LOAD_ENGINE") == a.engine
    # No KV connector is configured: LMCache is off, not merely idle.
    print("QWEN36-LOAD-CONFIG " + json.dumps({"engine": a.engine, "checkpoint": Path(a.checkpoint).name,
          "lmcache": "off", "kv_cache_dtype_requested": "fp8_e4m3", "tp": 1,
          "context_length": 2048, "speculation": "off", "purpose": "load-only short prompt"}), flush=True)
    if a.engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.checkpoint, tensor_parallel_size=1, dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=.75, max_num_seqs=1, max_num_batched_tokens=2048,
                  kv_cache_dtype="fp8_e4m3", enforce_eager=True, enable_prefix_caching=False,
                  seed=0, disable_log_stats=False)
        outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=16))
        print("QWEN36-LOAD-OUTPUT " + json.dumps({"text": outputs[0].outputs[0].text,
              "token_ids": list(outputs[0].outputs[0].token_ids)}), flush=True)
        llm.llm_engine.engine_core.shutdown()
    else:
        import sglang as sgl
        llm = sgl.Engine(model_path=a.checkpoint, tp_size=1, dtype="bfloat16", context_length=2048,
                         max_total_tokens=4096, max_running_requests=1, mem_fraction_static=.75,
                         kv_cache_dtype="fp8_e4m3", moe_runner_backend="flashinfer_cutlass",
                         disable_cuda_graph=True, random_seed=0, log_level="info")
        try:
            output = llm.generate("The capital of France is", {"temperature": 0, "max_new_tokens": 16})
            print("QWEN36-LOAD-OUTPUT " + json.dumps(output), flush=True)
        finally:
            llm.shutdown()
    print("QWEN36-LOAD-DONE", flush=True)

if __name__ == "__main__":
    main()
