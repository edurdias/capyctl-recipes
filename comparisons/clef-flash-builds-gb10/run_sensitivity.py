#!/usr/bin/env python3
"""Controlled BF16 numerical ablations, with separate calibration and selection."""
import importlib.util
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
bench = harness.bench


def control(**kwargs):
    tmp = ROOT / 'control.tmp'
    tmp.write_text(json.dumps({'generation': time.time_ns(), **kwargs}))
    tmp.replace(ROOT / 'control.json')


def main():
    ROOT.mkdir(exist_ok=True)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not active, 'GPU occupied'
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', 31180)) != 0, 'Port occupied'
    mem = lambda: int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))) * 1024
    assert mem() > 40 * 1024**3
    resume = '--resume' in sys.argv
    control(mode='bf16', calibrate=not resume)
    command = [str(PY), '-m', 'sglang.launch_server', '--model-path', str(H / 'models/clef-flash'), '--served-model-name', 'clef-flash', '--context-length', '16384', '--mem-fraction-static', '0.50', '--max-running-requests', '1', '--max-total-tokens', '16384', '--kv-cache-dtype', 'fp8_e4m3', '--mamba-ssm-dtype', 'bfloat16', '--host', '127.0.0.1', '--port', '31180']
    env = {**os.environ, 'PATH': str(PY.parent) + os.pathsep + os.environ['PATH'], 'MAX_JOBS': '2', 'CLEF_SENSITIVITY_ROOT': str(ROOT), 'PYTHONPATH': str(ROOT)}
    stop = threading.Event()
    samples = []
    server = None
    def monitor():
        with (ROOT / 'memory.jsonl').open('w') as stream:
            while not stop.is_set():
                row = {'time': time.time(), 'mem_available_bytes': mem()}
                samples.append(row)
                stream.write(json.dumps(row) + '\n'); stream.flush()
                if row['mem_available_bytes'] < 8 * 1024**3 and server and server.poll() is None:
                    os.killpg(server.pid, signal.SIGINT)
                stop.wait(1)
    monitor_thread = threading.Thread(target=monitor); monitor_thread.start()
    try:
        with (ROOT / 'engine.log').open('w') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            for _ in range(1800):
                if server.poll() is not None:
                    raise RuntimeError('Engine exited: ' + str(server.returncode))
                try:
                    with urllib.request.urlopen('http://127.0.0.1:31180/health', timeout=2) as response:
                        if response.status == 200: break
                except OSError: pass
                time.sleep(1)
            else: raise RuntimeError('Startup timed out')
            ep = bench.Endpoint('http://127.0.0.1:31180/v1')
            # Synthetic workload is generated independently; no held-out row is used.
            calibration = [{'id': f'calibration-{i}', 'state': bench.filler_text(n, 474700 + i), 'questions': bench.latency_questions(q)} for i, (n, q) in enumerate((n, q) for n in (2, 20, 100, 300, 700, 1300) for q in (1, 3, 5))]
            heldout = json.loads((H / 'clef-decision-eval/heldout.json').read_text())
            def identity(row): return harness.digest({'state': row['state'], 'questions': row['questions']})
            assert not {identity(r) for r in calibration} & {identity(r) for r in heldout}
            manifest = {'requests': len(calibration), 'sha256': harness.digest(calibration), 'disjoint_from_heldout': True, 'selection': 'independent deterministic synthetic workload; not heldout'}
            (ROOT / 'calibration-manifest.json').write_text(json.dumps(manifest, indent=2))
            if not resume:
                for row in calibration:
                    rec = bench.post_decision(ep, {'model': 'clef-flash', 'state': row['state'], 'questions': row['questions']}, keep_answers=True)
                    if rec['error']: raise RuntimeError(rec['error'])
                time.sleep(4)
                control(mode='bf16', calibrate=False)
                harness.capture(ep, 'diagnostic-104.json', ROOT / 'bf16-diagnostic.jsonl')
            print('CALIBRATION_AND_REFERENCE_COMPLETE', flush=True)
            # Converter is a separate approved environment; serving remains unchanged.
            subprocess.run([str(H / 'modelopt-0.47.0-venv/bin/python'), str(ROOT / 'build_bank.py')], check=True, env={**os.environ, 'MAX_JOBS': '2', 'PATH': str(PY.parent) + os.pathsep + os.environ['PATH']})
            units = json.loads((ROOT / 'units.json').read_text())
            for mode in ('fp8', 'nvfp4'):
                for unit, prefixes in units.items():
                    if mode == 'nvfp4':
                        prior = ROOT / f'fp8-{unit}-comparison.json'
                        if prior.exists() and json.loads(prior.read_text())['flips']:
                            continue
                    for activation in (('dynamic', 'static') if mode == 'fp8' else ('nvfp4',)):
                        name = f'{mode}-{unit}-{activation}'
                        control(mode=mode, prefixes=prefixes, activation=activation)
                        output = ROOT / f'{name}-diagnostic.jsonl'
                        harness.capture(ep, 'diagnostic-104.json', output)
                        comparison = compare(ROOT / 'bf16-diagnostic.jsonl', output)
                        (ROOT / f'{name}-comparison.json').write_text(json.dumps(comparison, indent=2))
                        print(name, json.dumps(comparison), flush=True)
                        if mode == 'fp8' and activation == 'dynamic':
                            (ROOT / f'fp8-{unit}-comparison.json').write_text(json.dumps(comparison, indent=2))
    finally:
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try: server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGTERM); server.wait(timeout=30)
        stop.set(); monitor_thread.join()
        (ROOT / 'sensitivity-run.json').write_text(json.dumps({'server_exit_code': server.returncode if server else None, 'minimum_mem_available_bytes': min((r['mem_available_bytes'] for r in samples), default=None), 'method': 'BF16 arithmetic with ModelOpt weight QDQ and explicit activation QDQ, one family at a time; not candidate kernel latency'}, indent=2))


def compare(reference, candidate):
    import math
    refs = [json.loads(s) for s in reference.read_text().splitlines()]
    candidates = [json.loads(s) for s in candidate.read_text().splitlines()]
    kl = []; tv = []; flips = []
    for ref, cand in zip(refs, candidates, strict=True):
        assert ref['payload_sha256'] == cand['payload_sha256']
        for q in ref['question_order']:
            p = harness.distribution(ref['answers'][q]); r = harness.distribution(cand['answers'][q])
            assert p.keys() == r.keys()
            kl.append(sum(v * math.log(max(v, 1e-9) / max(r[k], 1e-9)) for k, v in p.items()))
            tv.append(sum(abs(v-r[k]) for k,v in p.items()) / 2)
            if max(p, key=p.get) != max(r, key=r.get): flips.append([ref['id'], q])
    return {'questions': len(kl), 'agreement': 1-len(flips)/len(kl), 'mean_kl': sum(kl)/len(kl), 'mean_tv': sum(tv)/len(tv), 'flips': flips}


if __name__ == '__main__':
    main()
