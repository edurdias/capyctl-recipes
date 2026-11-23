#!/usr/bin/env python3
"""Launch only the approved local engine, measure, then release the GPU."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PRIVATE = Path.home() / "clef-decision-eval"
PYTHON = Path.home() / "sglang-0.5.22-venv/bin/python"
MODEL = Path(os.environ.get("CLEF_MODEL", str(Path.home() / "clef-flash-fp8-7b43c2d7")))
TAG = os.environ.get("CLEF_TAG", "laptop-fp8-sglang0522")


def main():
    # Let the smoke subprocess exit before serving, releasing its CUDA context.
    smoke = subprocess.check_output([str(PYTHON), "-c",
        "import json,sglang,torch; print(json.dumps(dict(torch=torch.__version__,"
        "torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),"
        "gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0))))"], text=True)
    assert metadata.version("sglang") == "0.5.22"
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        packages[name] = {"version": dist.version, "wheel": dist.read_text("WHEEL")}
    receipt = {"python": sys.version, "sglang": metadata.version("sglang"),
               **json.loads(smoke), "smoke_import": "passed: import sglang; import torch", "packages": packages}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / f"results/{TAG}-environment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "packages"}), flush=True)
    PRIVATE.mkdir(exist_ok=True)
    endpoint = os.environ["CLEF_ENDPOINT"]
    port = urllib.parse.urlsplit(endpoint).port
    host = urllib.parse.urlsplit(endpoint).hostname
    command = [str(PYTHON), "-m", "sglang.launch_server", "--model-path", str(MODEL),
               "--served-model-name", "clef-flash", "--context-length", "16384", "--mem-fraction-static", "0.85",
               "--max-running-requests", "1", "--max-total-tokens", "16384", "--kv-cache-dtype", "fp8_e4m3",
               "--mamba-ssm-dtype", "bfloat16", "--host", host, "--port", str(port)]
    redacted = command.copy()
    redacted[redacted.index("--host") + 1] = "<loopback>"
    memory = []
    stop = threading.Event()
    started = time.time()
    def sample():
        import pynvml
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            with (HERE / f"results/{TAG}-gpu-memory.jsonl").open("w") as stream:
                while not stop.is_set():
                    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
                    sample = {"time": time.time(), "used_bytes": pynvml.nvmlDeviceGetMemoryInfo(device).used, "mem_available_bytes": available}
                    memory.append(sample)
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    stop.wait(0.25)
        finally:
            pynvml.nvmlShutdown()
    sampler = threading.Thread(target=sample)
    sampler.start()
    ready_at = None
    server = None
    outcome = "failed"
    try:
        with (PRIVATE / f"sglang-{TAG}.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                      env={**os.environ, "MAX_JOBS": "2"})
            for _ in range(900):
                if server.poll() is not None:
                    raise RuntimeError(f"engine exited {server.returncode}; see private engine log")
                try:
                    with urllib.request.urlopen(endpoint.removesuffix("/v1") + "/health", timeout=2) as response:
                        if response.status == 200:
                            ready_at = time.time()
                            break
                except OSError:
                    pass
                time.sleep(1)
            if ready_at is None:
                raise RuntimeError("engine did not become healthy within 900 seconds")
            print(f"Ready after {ready_at - started:.3f}s", flush=True)
            subprocess.run([str(PYTHON), "-B", str(HERE / "decision_eval.py"), "measure", "--tag", TAG], check=True)
            outcome = "completed"
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=30)
        stop.set()
        sampler.join()
        startup = [r["used_bytes"] for r in memory if ready_at is None or r["time"] <= ready_at]
        steady = [r["used_bytes"] for r in memory if ready_at is not None and r["time"] > ready_at]
        summary = {"outcome": outcome, "launch_command": redacted, "started_at": started, "ready_at": ready_at,
                   "startup_seconds": ready_at - started if ready_at else None, "finished_at": time.time(),
                   "sample_interval_seconds": 0.25, "startup_peak_bytes": max(startup, default=None),
                   "steady_peak_bytes": max(steady, default=None), "whole_run_peak_bytes": max((r["used_bytes"] for r in memory), default=None),
                   "server_exit_code": server.returncode if server else None}
        (HERE / f"results/{TAG}-run.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
