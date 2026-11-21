#!/usr/bin/env python3
"""Own exactly one preliminary attention engine and its memory watcher."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter

HERE = Path(__file__).resolve().parent

def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024

def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return [int(p) for p in text.split() if p.isdigit()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('row', choices=['baseline', 'fp8-plain', 'fp8-dspark'])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--warmup-only', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    if gpu_pids():
        raise SystemExit('GPU occupied; no launch')
    args.work.mkdir(parents=True, exist_ok=False)
    original = args.row == 'baseline'
    checkpoint = Path.home()/'models'/('Inkling-2-Mini-ed40e7da' if original else 'Inkling-2-Mini-capyctl-fp8attn')
    argv = [str(Path(sys.executable).parent/'sglang'), 'serve', '--model-path', str(checkpoint), '--served-model-name', 'inkling-2-mini', '--tp', '1', '--ep', '1', '--attention-backend', 'fa4', '--dtype', 'bfloat16', '--page-size', '128', '--mem-fraction-static', '0.45', '--grammar-backend', 'xgrammar', '--enable-multimodal', '--reasoning-parser', 'inkling', '--tool-call-parser', 'inkling2', '--host', '127.0.0.1', '--port', '18130']
    opt = {}
    admin_key = None if original else secrets.token_urlsafe(32)
    if args.row == 'fp8-dspark':
        argv += ['--speculative-algorithm', 'DSPARK', '--speculative-num-draft-tokens', '8']
        opt = {key: '1' for key in ('SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL', 'SGLANG_OPT_DSV41_DSPARK_MID_FOLD', 'SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE', 'SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS')}
    result = {'row': args.row, 'argv': argv, 'environment': {'MAX_JOBS': '2', **opt}, 'profiling_authenticated': admin_key is not None, 'profile_only': args.profile_only, 'memory_before': available(), 'status': 'starting'}
    engine = None
    stop = threading.Event()
    floor = threading.Event()
    def save():
        (args.work/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    def terminate():
        if engine is not None and engine.poll() is None:
            os.killpg(engine.pid, signal.SIGTERM)
    def watch():
        with (args.work/'memory.jsonl').open('w') as f:
            while not stop.is_set():
                memory = available()
                f.write(json.dumps({'unix_s': time.time(), 'MemAvailable': memory})+'\n')
                f.flush()
                if memory < 3*1024**3:
                    floor.set()
                    terminate()
                    return
                stop.wait(2)
    watcher = None
    def command(arguments, label):
        with (args.work/(label+'.log')).open('w') as log:
            subprocess.run([str(x) for x in arguments], stdout=log, stderr=subprocess.STDOUT, check=True)
        if floor.is_set():
            raise RuntimeError('MemAvailable safety floor fired')
    try:
        save()
        start = time.monotonic()
        with (args.work/'engine.log').open('w') as log:
            child_env = {**os.environ, **opt}
            if admin_key is not None:
                child_env['SGLANG_ADMIN_API_KEY'] = admin_key
            engine = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result['engine_pid'] = engine.pid
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        while True:
            if engine.poll() is not None:
                raise RuntimeError('Engine exited before readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18130/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(2)
        result.update(ready_s=time.monotonic()-start, memory_ready=available(), load_only_passed=True)
        save()
        if not args.warmup_only:
            if not args.profile_only:
                if args.row != 'fp8-dspark':
                    phase = 'baseline' if original else 'candidate'
                    arguments = [sys.executable, HERE/'fp8-attention-check.py', phase, '--checkpoint', checkpoint, '--out', args.work/'sanity.json']
                    if not original:
                        arguments += ['--baseline', args.baseline]
                    command(arguments, 'sanity')
                command([sys.executable, HERE/'fp8-attention-check.py', 'speed', '--checkpoint', checkpoint, '--out', args.work/'speed.json'], 'speed')
            if not original:
                profile_dir = args.work/'profile'
                payload = json.dumps({'output_dir': str(profile_dir), 'activities': ['CPU', 'GPU']}).encode()
                req = urllib.request.Request('http://127.0.0.1:18130/start_profile', data=payload, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+admin_key})
                with urllib.request.urlopen(req, timeout=120) as response:
                    response.read()
                payload = json.dumps({'text': 'Explain floating point quantization.', 'sampling_params': {'temperature': 0, 'max_new_tokens': 16, 'ignore_eos': True}}).encode()
                req = urllib.request.Request('http://127.0.0.1:18130/generate', data=payload, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=1800) as response:
                    response.read()
                req = urllib.request.Request('http://127.0.0.1:18130/stop_profile', data=b'{}', headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+admin_key})
                with urllib.request.urlopen(req, timeout=1800) as response:
                    response.read()
                kernels = Counter()
                for path in profile_dir.rglob('*'):
                    if path.name.endswith('.json.gz'):
                        with gzip.open(path, 'rt') as f:
                            trace = json.load(f)
                    elif path.suffix == '.json':
                        trace = json.loads(path.read_text())
                    else:
                        continue
                    # Kineto adds the physical machine name independently of logs.
                    trace.pop('host_name', None)
                    trace.pop('hostname', None)
                    if path.name.endswith('.json.gz'):
                        with gzip.open(path, 'wt') as f:
                            json.dump(trace, f)
                    else:
                        path.write_text(json.dumps(trace)+'\n')
                    kernels.update(event['name'] for event in trace.get('traceEvents', []) if event.get('cat') == 'kernel')
                result['profile_kernels'] = dict(kernels)
                (args.work/'kernels.json').write_text(json.dumps(dict(kernels), indent=2)+'\n')
        result['status'] = 'warmed' if args.warmup_only else 'completed'
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc))
    finally:
        terminate()
        if engine is not None:
            try:
                engine.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(engine.pid, signal.SIGKILL)
                engine.wait()
        stop.set()
        if watcher is not None:
            watcher.join(timeout=10)
        log = args.work/'engine.log'
        if log.exists():
            text = log.read_text()
            if admin_key is not None:
                text = text.replace(admin_key, '[redacted]')
            text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[redacted address]', text)
            log.write_text(text)
        result.update(memory_after=available(), gpu_pids_after=gpu_pids(), safety_floor_fired=floor.is_set())
        save()
    print(json.dumps(result), flush=True)
    raise SystemExit(result['status'] == 'failed' or bool(result['gpu_pids_after']))

if __name__ == '__main__':
    main()
