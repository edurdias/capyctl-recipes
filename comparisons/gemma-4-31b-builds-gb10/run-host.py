#!/usr/bin/env python3
"""Run one pinned Gemma row under a 95GiB scope; stop only its owned scope."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
HOME = Path.home()
GIB = 1024 ** 3


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def gpu_pids():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('row', choices=['mse-sg', 'mse-dflash2', 'google-vl', 'redhat-vl', 'nvidia-vl', 'mse-sg-bf16'])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--load-only', action='store_true')
    parser.add_argument('--quality-only', action='store_true')
    parser.add_argument('--probes-only', action='store_true')
    args = parser.parse_args()
    if gpu_pids():
        raise SystemExit('GPU occupied: no launch')
    # Conservative planning bounds, not measured footprints: weights + explicit KV,
    # loader copies / workspace / graphs. The cgroup enforces the hard envelope.
    planned_peak = {'mse-sg': 65, 'mse-dflash2': 70, 'google-vl': 75,
                    'redhat-vl': 70, 'nvidia-vl': 85, 'mse-sg-bf16': 90}[args.row]
    if available() < max(100, planned_peak + 20) * GIB:
        raise SystemExit('Insufficient available memory for the planned peak plus 20GiB free')
    args.work.mkdir(parents=True, exist_ok=False)
    diagnostic = args.row == 'mse-sg-bf16'
    sglang = args.row in ('mse-sg', 'mse-sg-bf16')
    model_key = {'mse-sg': 'mse', 'mse-sg-bf16': 'mse', 'mse-dflash2': 'mse', 'google-vl': 'google',
                 'redhat-vl': 'redhat', 'nvidia-vl': 'nvidia'}[args.row]
    checkpoint = args.models / model_key
    context = 67584 if diagnostic else 98304
    engine_env = 'sglang-0.5.22-venv' if sglang else 'vllm-0.32.0-venv'
    python = HOME / engine_env / 'bin/python'
    command = [str(python), '-B', '-m', 'sglang.launch_server' if sglang else 'vllm.entrypoints.openai.api_server']
    command += ['--model-path' if sglang else '--model', str(checkpoint), '--host', '127.0.0.1', '--port', '18631',
                '--served-model-name', 'gemma31', '--reasoning-parser', 'gemma4', '--tool-call-parser', 'gemma4']
    if sglang:
        command += ['--context-length', str(context), '--chunked-prefill-size', '4096']
        if diagnostic:
            command += ['--kv-cache-dtype', 'bf16', '--max-total-tokens', '67584',
                        '--mem-fraction-static', '0.7', '--max-running-requests', '1',
                        '--cuda-graph-max-bs-decode', '1']
        else:
            command += ['--kv-cache-dtype', 'fp8_e4m3', '--swa-full-tokens-ratio', '0.25',
                        '--max-total-tokens', '98304', '--mem-fraction-static', '0.6',
                        '--max-running-requests', '8', '--cuda-graph-max-bs-decode', '8']
    else:
        command += ['--max-model-len', str(context), '--max-num-seqs', '8',
                    '--kv-cache-dtype', 'fp8', '--language-model-only', '--enable-auto-tool-choice',
                    '--kv-cache-memory-bytes', str(12 * GIB)]
        if args.row == 'mse-dflash2':
            command += ['--speculative-config', json.dumps({'method': 'dflash', 'model': str(args.models / 'dflash2'), 'num_speculative_tokens': 7})]
    unit = 'gemma31-' + args.row + '-' + str(os.getpid())
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'MAX_JOBS': '2',
           'CUDA_HOME': '/usr/local/cuda-13.0',
           'PATH': str(HOME / engine_env / 'bin') + ':/usr/local/cuda-13.0/bin:' + os.environ['PATH']}
    result = {'row': args.row, 'status': 'starting', 'engine_environment': engine_env,
              'command': [part.replace(str(HOME), '/home/me') for part in command],
              'context': context, 'planned_peak_gib_inference': planned_peak,
              'memory_scope_limit_gib': 95, 'guard_gib': 4, 'prompt_ceiling': 64000,
              'min_planned_free_gib': available() / GIB - planned_peak,
              'mem_available_before_bytes': available(), 'measurements': 'native engine; no controller or router overhead'}
    save(args.work / 'run.json', result)
    stop = threading.Event()
    samples = []
    guarded = threading.Event()
    engine = None

    def stop_scope():
        subprocess.run(['systemctl', '--user', 'stop', unit + '.scope'], capture_output=True, timeout=180)

    def watch():
        while not stop.wait(.5):
            value = available()
            samples.append({'unix_s': time.time(), 'mem_available_bytes': value})
            if value < 4 * GIB:
                guarded.set()
                stop_scope()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    with (args.work / 'engine.log').open('wb') as engine_log, (args.work / 'clients.log').open('wb') as client_log:
        try:
            started = time.monotonic()
            engine = subprocess.Popen(['systemd-run', '--user', '--scope', '--unit', unit, '-p', 'MemoryMax=95G', *command],
                                      env=env, stdout=engine_log, stderr=engine_log, start_new_session=True)
            deadline = started + 1800
            while time.monotonic() < deadline:
                if engine.poll() is not None:
                    raise RuntimeError('engine exited before readiness')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18631/v1/models', timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError('engine readiness: 1800 seconds')
            result.update(ready_s=time.monotonic() - started, load_completed=True,
                          mem_available_ready_bytes=available())
            save(args.work / 'run.json', result)
            if not args.load_only:
                quality_python = HOME / 'sglang-0.5.22-venv/bin/python'
                if not args.probes_only:
                    if not args.quality_only:
                        measure = [str(python), '-B', str(HERE / 'measure.py'), '--checkpoint', str(checkpoint),
                                   '--out', str(args.work / 'measure.json'), '--context', str(context)]
                        subprocess.run(measure, env=env, stdout=client_log, stderr=client_log, check=True, timeout=28800)
                    quality = [str(quality_python), '-B', str(HERE / 'accuracy.py'), '--endpoint', 'http://127.0.0.1:18631',
                               '--model', 'gemma31', '--out', str(args.work / ('full' if args.full else 'screen'))]
                    if not args.full:
                        quality.append('--screen')
                    subprocess.run(quality, env=env, stdout=client_log, stderr=client_log, check=True, timeout=86400)
                probes = [str(quality_python), '-B', str(HERE / 'gates.py'), '--checkpoint', str(checkpoint),
                          '--out', str(args.work / 'gates.json'), '--context', str(context)]
                if not args.full:
                    probes.append('--retrieval-copy-only')
                subprocess.run(probes, env=env, stdout=client_log, stderr=client_log, check=True, timeout=86400)
            result['status'] = 'measured' if not args.load_only else 'loaded'
        except Exception as exc:
            result.update(status='blocked', error=str(exc).replace(str(HOME), '/home/me'))
        finally:
            try:
                props = subprocess.run(['systemctl', '--user', 'show', unit + '.scope', '-p', 'MemoryPeak', '-p', 'MemoryCurrent'], capture_output=True, text=True)
                result['scope_memory'] = props.stdout.splitlines()
                stop_scope()
            except Exception as exc:
                result['cleanup_error'] = str(exc)
            if engine is not None:
                try:
                    engine.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(engine.pid, signal.SIGTERM)
            stop.set()
            watcher.join(timeout=10)
            result.update(memory_guard_triggered=guarded.is_set(), gpu_pids_after=gpu_pids())
            result['min_mem_available_bytes'] = min((s['mem_available_bytes'] for s in samples), default=available())
            result['peak_mem_available_drop_bytes'] = result['mem_available_before_bytes'] - result['min_mem_available_bytes']
            save(args.work / 'memory.json', samples)
            save(args.work / 'run.json', result)
    print(json.dumps(result), flush=True)
    raise SystemExit(result['status'] == 'blocked' or bool(result['gpu_pids_after']))


if __name__ == '__main__':
    main()
