#!/usr/bin/env python3
"""Stock-loader held-out captures and the established three-round timing grid."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

H = Path.home()
ROOT = H / 'clef-build'
HERE = H / 'clef-reference/comparisons/clef-flash-builds-gb10'
PY = H / 'sglang-0.5.22-venv/bin/python'
sys.path.insert(0, str(HERE))
import decision_eval as harness
sys.path.insert(0, str(ROOT))
from analyze_candidates import compare


@contextmanager
def engine(tag, path, hooks=False):
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not active, 'GPU occupied; refusing launch'
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1', 31180)) != 0, 'Port occupied'
    mem = lambda: int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))) * 1024
    assert mem() > 40 * 1024**3
    command = [str(PY), '-m', 'sglang.launch_server', '--model-path', str(path), '--served-model-name', 'clef-flash', '--context-length', '16384', '--mem-fraction-static', '0.50', '--max-running-requests', '1', '--max-total-tokens', '16384', '--kv-cache-dtype', 'fp8_e4m3', '--mamba-ssm-dtype', 'bfloat16', '--host', '127.0.0.1', '--port', '31180']
    env = {**os.environ, 'MAX_JOBS': '2', 'PATH': str(PY.parent)+os.pathsep+os.environ['PATH']}
    env.pop('CLEF_SENSITIVITY_ROOT', None); env.pop('PYTHONPATH', None)
    if hooks: env.update(CLEF_SENSITIVITY_ROOT=str(ROOT), PYTHONPATH=str(ROOT))
    stop = threading.Event(); samples = []; server = None; ready = None; start = time.time(); outcome = 'failed'
    def sample():
        with (HERE / f'results/{tag}-memory.jsonl').open('w') as stream:
            while not stop.is_set():
                row = {'time': time.time(), 'mem_available_bytes': mem(), 'used_bytes': None}
                samples.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
                if row['mem_available_bytes'] < 8 * 1024**3 and server and server.poll() is None: os.killpg(server.pid, signal.SIGINT)
                stop.wait(1)
    thread = threading.Thread(target=sample); thread.start()
    try:
        with (ROOT / f'{tag}-engine.log').open('w') as stream:
            server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            for _ in range(1800):
                if server.poll() is not None: raise RuntimeError('Engine exited: '+str(server.returncode))
                try:
                    with urllib.request.urlopen('http://127.0.0.1:31180/health', timeout=2) as response:
                        if response.status == 200: ready = time.time(); break
                except OSError: pass
                time.sleep(1)
            else: raise RuntimeError('Startup timed out')
            print('Ready', tag, ready-start, flush=True)
            yield harness.bench.Endpoint('http://127.0.0.1:31180/v1')
            outcome = 'completed'
    finally:
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try: server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGTERM); server.wait(timeout=30)
        stop.set(); thread.join()
        receipt = {'tag': tag, 'outcome': outcome, 'launch_command': command, 'stock_loader': not hooks, 'started_at': start, 'startup_seconds': ready-start if ready else None, 'server_exit_code': server.returncode if server else None, 'minimum_mem_available_bytes': min((r['mem_available_bytes'] for r in samples), default=None), 'gpu_peak_bytes': None, 'gpu_measurement': 'NVML device memory unsupported; system MemAvailable independently sampled'}
        (HERE / f'results/{tag}-run.json').write_text(json.dumps(receipt, indent=2)+'\n')


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    assert len(spec['candidates']) <= 3
    for candidate in spec['candidates']:
        tag = candidate['tag']
        with engine(tag, H/'models'/tag):
            os.environ['CLEF_ENDPOINT'] = 'http://127.0.0.1:31180/v1'
            harness.measure(tag, timing=True)
        for cohort in ('diagnostic', 'heldout'):
            result = compare(HERE/f'results/gb10-bf16-{cohort}.jsonl', HERE/f'results/{tag}-{cohort}.jsonl')
            (HERE/f'results/{tag}-vs-bf16-{cohort}.json').write_text(json.dumps(result, indent=2)+'\n')
            print(tag, cohort, json.dumps({k:v for k,v in result.items() if k not in ('flips','by_suite','by_length')}), flush=True)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not active
    (HERE/'results/capyctl-gpu-release.json').write_text(json.dumps({'compute_inventory_empty': True, 'time': time.time(), 'owned_engines_stopped': True}, indent=2)+'\n')
    print('host A GPU released', flush=True)


if __name__ == '__main__':
    main()
