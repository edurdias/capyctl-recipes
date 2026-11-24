#!/usr/bin/env python3
"""Run an owned build/engine, stop only its process group below 3 GiB available."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--log', type=Path, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command and a.command[0] == '--' else a.command
assert command
assert os.environ.get('QWEN35_GPU_RELEASE') == 'GPU owner GPU free'
env = dict(os.environ, MAX_JOBS='2')
a.log.parent.mkdir(parents=True, exist_ok=True)
minimum = None
with a.log.open('w') as output:
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            values = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
            available = values['MemAvailable']
            minimum = available if minimum is None else min(minimum, available)
            if available < 3 * 1024 ** 3:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError('Stopped owned process group below 3 GiB MemAvailable')
            time.sleep(5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        a.log.with_suffix(a.log.suffix + '.resources.json').write_text(json.dumps({'pid': process.pid, 'minimum_mem_available_bytes': minimum, 'returncode': process.returncode, 'max_jobs': 2}, indent=2) + '\n')
raise SystemExit(process.returncode)
