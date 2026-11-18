#!/usr/bin/env python3
"""Run the published native vLLM configuration in an existing environment.

Only newly created process groups are stopped. Occupied-host preflight failures
leave no files; all runs admitted to a new work directory write run.json and
memory.json, including failed launches. No packages or installed files change.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


class RunFailure(Exception):
    """A fixed, non-secret failure description suitable for saved results."""


def gpu_apps():
    try:
        query = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True)
        apps = []
        for line in query.stdout.splitlines():
            if not line.strip():
                continue
            pid, used = (field.strip() for field in line.split(',', 1))
            apps.append({'pid': int(pid), 'used_memory_bytes':
                         int(used) * 1024 * 1024 if used.isdigit() else None})
        return apps
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RunFailure('GPU process query failed; occupancy is unknown') from None


def engine_processes():
    engines = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            name = (proc / 'comm').read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            raise RunFailure('Process inspection failed; occupancy is unknown') from None
        if name == 'capyctl' or name.startswith(('VLLM::', 'sglang::')):
            engines.append(int(proc.name))
    return sorted(engines)


def require_empty():
    active = gpu_apps()
    engines = engine_processes()
    if active or engines:
        raise RunFailure('Host occupied; existing processes were not signaled')


def host_memory():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('MemTotal', 'MemAvailable'):
            fields[key] = int(value.split()[0]) * 1024
    available = fields['MemAvailable']
    return available, fields['MemTotal'] - available


def child_environment():
    env = os.environ.copy()
    removed = [key for key in env if key.upper().startswith(
        ('LMCACHE', 'VLLM_KV_TRANSFER_CONFIG'))]
    for key in removed:
        del env[key]
    return env, bool(removed), len(removed)


def log_redactor():
    # Do not persist arbitrary exception text, URLs, node names, or credentials
    # printed by dependencies. Never dump the inherited environment or argv.
    secrets = sorted({value for key, value in os.environ.items() if value and
                      re.search(r'TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|ACCESS_KEY|AUTHORIZATION',
                                key, re.I)}, key=len, reverse=True)
    hostname = socket.gethostname()
    def redact(line):
        for value in secrets:
            line = line.replace(value, '[redacted]')
        if hostname:
            line = line.replace(hostname, '[node]')
        line = re.sub(r'(?i)\b(?:https?|wss?|tcp)://[^\s\x1b<>\"\']+', '[endpoint]', line)
        line = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b', '[address]', line)
        line = re.sub(r'(?i)(?<![\w/])(?:[0-9a-f]{0,4}:){2,}[0-9a-f:.%]+', '[address]', line)
        line = re.sub(r'(?i)\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+'
                      r'(?:com|net|org|io|ai|local|internal|lan)\b', '[hostname]', line)
        line = re.sub(r'(?i)\b(?:authorization|proxy-authorization|cookie|set-cookie|x-api-key)'
                      r'\s*[:=].*', '[credential header redacted]', line)
        line = re.sub(r'(?i)\bbearer\s+\S+', '[credential redacted]', line)
        return line
    return redact


class ProcessLog:
    def __init__(self, process, path, redact):
        self.process = process
        self.path = path
        self.redact = redact
        self.error = False
        self.evidence = {'fp8_seen': False, 'kv_cache_dtype_seen': False,
                         'kv_transfer_config_none_seen': False}
        self.thread = threading.Thread(target=self.copy, daemon=True)
        self.thread.start()

    def copy(self):
        try:
            with self.path.open('a', encoding='utf-8') as log:
                for raw in iter(self.process.stdout.readline, b''):
                    line = self.redact(raw.decode('utf-8', errors='replace'))
                    lower = line.lower()
                    self.evidence['fp8_seen'] |= 'fp8' in lower
                    self.evidence['kv_cache_dtype_seen'] |= 'kv_cache_dtype' in lower
                    self.evidence['kv_transfer_config_none_seen'] |= bool(re.search(
                        r'kv_transfer_config[\"\']?\s*[:=]\s*(?:none|null)', lower))
                    log.write(line)
                    log.flush()
        except (OSError, ValueError):
            self.error = True
        finally:
            self.process.stdout.close()


class MemorySampler:
    def __init__(self, baseline_available, baseline_used):
        self.baseline_available = baseline_available
        self.baseline_used = baseline_used
        self.samples = []
        self.stopping = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        next_sample = time.monotonic()
        while not self.stopping.is_set():
            row = {'unix_s': time.time(), 'elapsed_s': time.monotonic() - self.started,
                   'host_available_bytes': None, 'host_used_bytes': None,
                   'host_used_delta_bytes': None, 'gpu_compute_apps': None,
                   'gpu_used_sum_bytes': None}
            try:
                available, used = host_memory()
                row.update(host_available_bytes=available, host_used_bytes=used,
                           host_used_delta_bytes=self.baseline_available - available)
            except (OSError, KeyError, ValueError):
                row['host_sample_error'] = 'Host memory query failed'
            try:
                apps = gpu_apps()
                row['gpu_compute_apps'] = apps
                if all(app['used_memory_bytes'] is not None for app in apps):
                    row['gpu_used_sum_bytes'] = sum(app['used_memory_bytes'] for app in apps)
            except RunFailure:
                row['gpu_sample_error'] = 'GPU memory query failed'
            self.samples.append(row)
            next_sample += 0.5
            self.stopping.wait(max(0, next_sample - time.monotonic()))

    def finish(self):
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join()
        used = [self.baseline_used] + [row['host_used_bytes'] for row in self.samples
                                      if row['host_used_bytes'] is not None]
        deltas = [0] + [row['host_used_delta_bytes'] for row in self.samples
                        if row['host_used_delta_bytes'] is not None]
        # Empty-device samples must not turn an unavailable GPU metric into zero.
        gpu = [row['gpu_used_sum_bytes'] for row in self.samples
               if row['gpu_compute_apps'] and row['gpu_used_sum_bytes'] is not None]
        return {'sample_interval_s': 0.5,
                'host_available_before_bytes': self.baseline_available,
                'host_used_before_bytes': self.baseline_used,
                'peak_host_used_bytes': max(used),
                'peak_host_used_delta_bytes': max(deltas),
                'peak_gpu_used_sum_bytes': max(gpu) if gpu else None,
                'samples': self.samples}


def group_alive(process):
    process.poll()  # Reap an exited root, but still inspect its surviving workers.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if (int(fields[2]) == process.pid and int(fields[3]) == process.pid
                    and fields[0] not in ('Z', 'X')):
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def launch_owned(argv, env, owned):
    # Defer termination handlers through Popen and ownership registration, so a
    # signal cannot strand a newly spawned child between those two operations.
    pending = []
    handlers = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[sig] = signal.signal(sig, lambda number, frame: pending.append(number))
        process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        owned.append(process)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if pending:
        raise RunFailure('Run interrupted by signal ' + str(pending[0]))
    return process


def stop_owned(process):
    if not group_alive(process):
        process.wait(timeout=5)
        return
    for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not group_alive(process):
                break
            time.sleep(0.1)
        if not group_alive(process):
            break
    if group_alive(process):
        raise RunFailure('An owned process group survived cleanup')
    process.wait(timeout=5)


def wait_ready(process):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    deadline = started + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunFailure('Native engine exited before readiness')
        try:
            with opener.open('http://127.0.0.1:18400/health',
                             timeout=min(2, max(0.01, deadline - time.monotonic()))) as response:
                if response.status == 200 and process.poll() is None:
                    return time.monotonic() - started
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    raise RunFailure('Native engine was not ready within 1800 seconds')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--drafter', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True, help='new directory outside /tmp')
    parser.add_argument('--vllm-venv', type=Path,
                        default=Path.home() / 'vllm-0.32.0-lmcache-venv')
    parser.add_argument('--utilization', type=float, default=0.85)
    parser.add_argument('--engine-args-file', type=Path, help='JSON list of exact published native engine flags')
    parser.add_argument('--benchmark-script', type=Path, help='alternate native workload client using the same endpoint/model/out contract')
    parser.add_argument('--benchmark-arg', action='append', default=[], help='one additional workload-client argument')
    parser.add_argument('--cpu-affinity', help='published engine CPU list, applied with taskset to the owned engine only')
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    if work.is_relative_to(Path('/tmp')):
        parser.error('--work must be outside /tmp')
    if work.exists():
        parser.error('--work must be a new directory')
    if not math.isfinite(args.utilization) or not 0 < args.utilization <= 1:
        parser.error('--utilization must be finite and in (0, 1]')
    try:
        require_empty()
    except RunFailure as error:
        print(str(error), file=sys.stderr)
        return 1
    work.mkdir(parents=True, exist_ok=False)
    result = {'launcher': 'native vLLM published recipe', 'status': 'failed',
              'ready_s': None, 'gpu_memory_utilization': args.utilization,
              'kv_cache_dtype': 'fp8', 'max_model_len': 4096,
              'speculative_method': 'mtp', 'num_speculative_tokens': 4,
              'lmcache': False, 'kv_transfer_config_supplied': False,
              'ambient_lmcache_injection_removed': None,
              'ambient_lmcache_variables_removed_count': None,
              'engine_returncode': None, 'benchmark_returncode': None,
              'gpu_pids_after_cleanup': None, 'error': None}
    result['checkpoint'] = str(args.checkpoint.expanduser())
    result['drafter_path'] = str(args.drafter.expanduser())
    result['cpu_affinity'] = args.cpu_affinity
    memory = {'sample_interval_s': 0.5, 'host_available_before_bytes': None,
              'host_used_before_bytes': None, 'peak_host_used_bytes': None,
              'peak_host_used_delta_bytes': None, 'peak_gpu_used_sum_bytes': None,
              'samples': []}
    owned, logs, handlers = [], [], {}
    sampler = engine = benchmark = None
    stage = 'initialization'
    def interrupted(signum, frame):
        raise RunFailure('Run interrupted by signal ' + str(signum))
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[sig] = signal.signal(sig, interrupted)
        env, removed, removed_count = child_environment()
        result['ambient_lmcache_injection_removed'] = removed
        result['ambient_lmcache_variables_removed_count'] = removed_count
        python = args.vllm_venv.expanduser() / 'bin/python'
        checkpoint, drafter = args.checkpoint.expanduser(), args.drafter.expanduser()
        if not python.is_file() or not os.access(python, os.X_OK):
            raise RunFailure('Existing vLLM environment Python is unavailable')
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        env.setdefault('CUDA_HOME', '/usr/local/cuda')
        result['toolchain_activation'] = 'existing vLLM bin prepended to PATH; existing CUDA home /usr/local/cuda, matching the registered CapyCTL profile'
        if not checkpoint.is_dir() or not drafter.is_dir():
            raise RunFailure('Checkpoint and drafter must be existing local directories')
        bench_script = args.benchmark_script or Path(__file__).with_name('published-bench.py')
        if not bench_script.is_file():
            raise RunFailure('Published benchmark script is unavailable')
        redact = log_redactor()
        baseline_available, baseline_used = host_memory()
        sampler = MemorySampler(baseline_available, baseline_used)
        sampler.thread.start()
        argv = [str(python), '-m', 'vllm.entrypoints.openai.api_server',
                '--host', '127.0.0.1', '--port', '18400',
                '--served-model-name', 'gemma4-published', '--model', str(checkpoint),
                '--speculative-config', json.dumps({'method': 'mtp', 'model': str(drafter),
                                                    'num_speculative_tokens': 4}),
                '--kv-cache-dtype', 'fp8', '--gpu-memory-utilization', str(args.utilization),
                '--max-model-len', '4096', '--limit-mm-per-prompt',
                '{"image":0,"audio":0,"video":0}']
        if args.engine_args_file:
            published_args = json.loads(args.engine_args_file.read_text())
            if not isinstance(published_args, list) or not all(isinstance(item, str) for item in published_args):
                raise RunFailure('Published engine arguments must be a JSON string list')
            if any(item.split('=', 1)[0] in ('--host', '--port', '--model', '--api-key', '--served-model-name')
                   for item in published_args):
                raise RunFailure('Published flags cannot override the owned listener or target')
            argv = argv[:argv.index('--speculative-config')] + published_args
            def argument(flag, default):
                return published_args[published_args.index(flag)+1] if flag in published_args else default
            result.update(kv_cache_dtype=argument('--kv-cache-dtype', 'auto'),
                          max_model_len=int(argument('--max-model-len', '0')),
                          gpu_memory_utilization=float(argument('--gpu-memory-utilization', str(args.utilization))))
            if '--speculative-config' in published_args:
                spec = json.loads(argument('--speculative-config', '{}'))
                result.update(speculative_method=spec.get('method', 'mtp'), num_speculative_tokens=spec['num_speculative_tokens'])
            else:
                result.update(speculative_method=None, num_speculative_tokens=0)
        result['native_engine_flags'] = argv[argv.index('--model')+2:]
        (work / 'engine.log').write_text(
            'Launcher request: kv_cache_dtype=' + result['kv_cache_dtype'] + '; LMCache configuration absent; '
            'no --kv-transfer-config; ambient LMCache injection removed: '
            + ('yes' if removed else 'no') + '\n')
        if args.cpu_affinity:
            argv = ['taskset', '--cpu-list', args.cpu_affinity, *argv]
        stage = 'prelaunch occupancy check'
        require_empty()
        stage = 'engine launch'
        engine = launch_owned(argv, env, owned)
        logs.append(ProcessLog(engine, work / 'engine.log', redact))
        stage = 'readiness'
        result['ready_s'] = wait_ready(engine)
        stage = 'benchmark launch'
        benchmark = launch_owned(
            [sys.executable, str(bench_script), '--recipes', str(Path(__file__).resolve().parents[2]),
             '--endpoint', 'http://127.0.0.1:18400/v1', '--model', 'gemma4-published',
             '--out', str(work / 'measure.json'), *args.benchmark_arg],
            env, owned)
        logs.append(ProcessLog(benchmark, work / 'benchmark.log', redact))
        stage = 'benchmark'
        while benchmark.poll() is None:
            if engine.poll() is not None:
                raise RunFailure('Native engine exited during benchmark')
            time.sleep(0.5)
        result['benchmark_returncode'] = benchmark.returncode
        if benchmark.returncode != 0:
            raise RunFailure('Published benchmark failed; see sanitized benchmark log')
        if engine.poll() is not None:
            raise RunFailure('Native engine exited during benchmark')
        if not (work / 'measure.json').is_file():
            raise RunFailure('Published benchmark did not produce measurement output')
        result['status'] = 'measured'
    except RunFailure as error:
        result['error'] = str(error)
    except BaseException as error:
        # Exception messages can include argv, environment values, and endpoints.
        result['error'] = stage + ' failed (' + type(error).__name__ + ')'
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        cleanup_errors = []
        for process in reversed(owned):
            try:
                stop_owned(process)
            except BaseException:
                cleanup_errors.append('Owned process cleanup failed')
        if sampler is not None:
            try:
                memory = sampler.finish()
            except BaseException:
                cleanup_errors.append('Memory sampler shutdown failed')
        for log in logs:
            log.thread.join(timeout=10)
            if log.thread.is_alive() or log.error:
                cleanup_errors.append('Sanitized process log capture failed')
        if logs:
            result['native_log_evidence'] = logs[0].evidence
        if engine is not None:
            result['engine_returncode'] = engine.poll()
        if benchmark is not None:
            result['benchmark_returncode'] = benchmark.poll()
        try:
            result['gpu_pids_after_cleanup'] = sorted({app['pid'] for app in gpu_apps()})
        except RunFailure:
            cleanup_errors.append('GPU process query after cleanup failed')
        if cleanup_errors:
            result['cleanup_errors'] = cleanup_errors
            result['status'] = 'failed'
        if result['gpu_pids_after_cleanup']:
            result['status'] = 'failed'
            result['post_cleanup_gpu_occupied'] = True
        result.update({key: memory[key] for key in (
            'host_used_before_bytes', 'peak_host_used_bytes',
            'peak_host_used_delta_bytes', 'peak_gpu_used_sum_bytes')})
        result['completed_unix_s'] = time.time()
        try:
            write_json(work / 'memory.json', memory)
        finally:
            write_json(work / 'run.json', result)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    print(json.dumps({'status': result['status'], 'error': result['error']}))
    return 0 if result['status'] == 'measured' else 1


if __name__ == '__main__':
    raise SystemExit(main())
