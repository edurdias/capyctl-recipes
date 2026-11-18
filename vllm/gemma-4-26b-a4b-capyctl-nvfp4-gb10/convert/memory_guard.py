"""Keep a host-wide 8-GiB MemAvailable margin during offline GPU work.

Only interrupts the process that creates this guard. No other process is signaled.
The main-thread SIGINT handler unwinds Python and releases its CUDA allocations.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import signal
import subprocess
import threading


def memory_info():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable"):
            values[name] = int(value.split()[0]) * 1024
    return values


class MemoryGuard:
    def __init__(self, minimum_gib=8):
        self.limit = int(minimum_gib * 2**30)
        self.minimum = None
        self.total = None
        self.baseline = None
        self.exceeded = False
        self.samples = 0
        self.stop = threading.Event()
        self.thread = None
        self.residents = []

    def sample(self):
        info = memory_info()
        self.total = info["MemTotal"]
        available = info["MemAvailable"]
        self.minimum = available if self.minimum is None else min(self.minimum, available)
        self.samples += 1
        return available

    def __enter__(self):
        self.baseline = self.sample()
        if self.baseline < self.limit:
            raise MemoryError(f"MemAvailable {self.baseline / 2**30:.2f} GiB is below the 8-GiB safety margin")
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
        if result.returncode == 0:
            self.residents = [{"workload": "another GPU process", "gpu_memory": row[1].strip()}
                              for row in csv.reader(result.stdout.splitlines())
                              if len(row) == 2 and row[0].strip() != str(os.getpid())]

        def monitor():
            while not self.stop.wait(.1):
                if self.sample() < self.limit:
                    self.exceeded = True
                    print(f"[memory-guard] MemAvailable {self.minimum / 2**30:.2f} GiB < 8 GiB; interrupting this conversion only", flush=True)
                    os.kill(os.getpid(), signal.SIGINT)
                    return
        self.thread = threading.Thread(target=monitor, name="gemma-memory-guard", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.exceeded:
            raise MemoryError("Conversion stopped to preserve at least approximately 8 GiB of host MemAvailable") from exc
        return False

    def metrics(self):
        return {"minimum_allowed_available_bytes": self.limit, "minimum_observed_available_bytes": self.minimum,
                "baseline_available_bytes": self.baseline, "system_total_bytes": self.total,
                "system_peak_non_available_bytes": self.total - self.minimum,
                "samples": self.samples, "sample_interval_seconds": .1, "stopped_for_memory": self.exceeded,
                "resident_gpu_processes_before": self.residents}
