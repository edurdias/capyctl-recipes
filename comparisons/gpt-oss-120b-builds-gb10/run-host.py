#!/usr/bin/env python3
"""Sequential GPT-OSS-120B qualification using released CapyCTL and existing clients."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRECEDENT = HERE.parent/'gpt-oss-20b-builds-gb10'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def gpu_pids():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()


def mem_available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('row', choices=('capyctl-vl032', 'openai-vl032', 'published-vl030'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--load-only', action='store_true')
    parser.add_argument('--load-strategy', choices=('eager', 'lazy'))
    parser.add_argument('--expandable-segments', action='store_true')
    parser.add_argument('--recipe-file', type=Path)
    parser.add_argument('--performance-only', action='store_true',
                        help='Measure throughput and exact-token decode without repeating quality or probes.')
    parser.add_argument('--retrieval-copy', action='store_true',
                        help='Run pinned retrieval and copy probes, including for OpenAI.')
    a = parser.parse_args()
    if gpu_pids():
        raise SystemExit('GPU occupied; no launch')
    a.work.mkdir(parents=True, exist_ok=False)
    candidate = a.row == 'capyctl-vl032'
    published = a.row == 'published-vl030'
    executable = Path.home()/('[private identity removed]' if published else 'vllm-0.32.0-lmcache-venv')
    python = executable/'bin/python'
    capyctl = Path.home()/'gptoss120-work/capyctl-0.1.12-linux-aarch64/bin/capyctl'
    checkpoint = Path.home()/'models'/('gpt-oss-120b-capyctl-fp8attn' if candidate else 'gpt-oss-120b')
    recipe_original_model = None
    if a.recipe_file:
        if published or a.load_strategy or a.expandable_segments:
            raise SystemExit('shipped recipe permits only the local checkpoint substitution')
        import yaml
        recipe = yaml.safe_load(a.recipe_file.read_text())
        recipe_original_model = recipe['model']
        recipe['model'] = str(checkpoint)
    elif published:
        import yaml
        recipe = yaml.safe_load((ROOT/'vllm/gpt-oss-120b-gb10/deployment.yaml').read_text())
        recipe['model'] = str(checkpoint)
    else:
        recipe = {'schema_version': 1, 'kind': 'deployment', 'name': 'gptoss120-'+a.row,
                  'engine': 'vllm', 'model': str(checkpoint), 'residency': 'restart_only', 'lmcache': 'off',
                  'timeouts': {'initialize': '1800s'},
                  'engine_config': {'context_length': 131072, 'max_concurrent_requests': 8,
                                    'memory': {'kv_cache': '8GiB', 'startup': '90GiB'},
                                    'kv_cache_dtype': 'fp8_e4m3', 'accept_extra_args': True,
                                    'extra_args': ['--no-enable-prefix-caching', '--moe-backend', 'marlin'],
                                    'vllm': {'tool_call_parser': 'openai', 'reasoning_parser': 'openai_gptoss'}}}
    if a.load_strategy:
        if published:
            raise SystemExit('published recipe must run as written')
        recipe['engine_config']['vllm']['safetensors_load_strategy'] = a.load_strategy
    save(a.work/'deployment.yaml', recipe)
    env = {**os.environ, 'MAX_JOBS': '2', 'CUDA_HOME': '/usr/local/cuda-13.0',
           'PATH': str(executable/'bin')+':/usr/local/cuda-13.0/bin:'+os.environ['PATH'],
           'HF_HOME': str(Path.home()/'gptoss120-work/hf-cache'),
           'XDG_CONFIG_HOME': str(a.work/'config'), 'XDG_STATE_HOME': str(a.work/'xdg-state')}
    state = a.work/'state'
    cli = [str(capyctl), '--state-dir', str(state)]
    result = {'row': a.row, 'settings': recipe, 'status': 'starting', 'quality_cap': 32768,
              'client_timeout_s': 2400, 'cold_definition': 'fresh process; file/JIT caches not cleared'}
    result['engine_environment'] = {'MAX_JOBS': '2', **recipe['engine_config'].get('env', {})}
    if a.recipe_file:
        import hashlib
        result['recipe_source_sha256'] = hashlib.sha256(a.recipe_file.read_bytes()).hexdigest()
        result['recipe_substitution'] = {'source': recipe_original_model, 'local': str(checkpoint)}
    if a.expandable_segments:
        if published:
            raise SystemExit('published recipe must run as written')
        result['engine_environment']['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    result['mem_total_bytes'] = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))*1024
    result['mem_available_before_bytes'] = mem_available()
    stop = threading.Event()
    samples = []
    owned = set()
    engine_pids = set()
    guard_triggered = threading.Event()
    role = None
    deployed = False
    with (a.work/'cli.log').open('wb') as log, (a.work/'standalone.log').open('wb') as rolelog:
        def command(args, timeout=3600, cpu=False):
            if engine_pids and str(args[0]) != str(capyctl):
                foreign = set(gpu_pids())-engine_pids
                if foreign:
                    raise RuntimeError('foreign GPU workload detected at phase boundary; no new measurement started')
            log.write((json.dumps(list(map(str, args)))+'\n').encode()); log.flush()
            child = subprocess.Popen(list(map(str, args)), stdout=log, stderr=log,
                                     env={**env, **({'CUDA_VISIBLE_DEVICES': ''} if cpu else {})}, start_new_session=True)
            owned.add(child.pid)
            try:
                code = child.wait(timeout=timeout)
                if code:
                    raise RuntimeError(f'command failed ({code}): {args[1:3]}')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                owned.discard(child.pid)
        def watch():
            while not stop.wait(.5):
                available = mem_available()
                samples.append({'unix_s': time.time(), 'mem_available_bytes': available})
                if available < 3*1024**3:
                    guard_triggered.set()
                    for pid in list(owned):
                        try:
                            os.killpg(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            version = subprocess.check_output([str(capyctl), '--version'], text=True).strip()
            if version != 'capyctl 0.1.12':
                raise RuntimeError('requires released CapyCTL 0.1.12')
            result['capyctl_version'] = version
            declaration_env = recipe['engine_config'].get('env', {})
            profile_env_args = [part for key, value in result['engine_environment'].items() if key not in declaration_env for part in ('--env', key+'='+value)]
            profile_env_args += [part for key in declaration_env for part in ('--approve-env', key)]
            command(cli+['engine', 'add', executable, '--name', 'vllm', '--approve-path', Path.home()/'models',
                         '--env', 'TIKTOKEN_ENCODINGS_BASE='+str(Path.home()/'catalog18/tiktoken')]+profile_env_args)
            command(cli+['validate', 'config', '--file', a.work/'deployment.yaml'])
            role = subprocess.Popen(cli+['start', 'standalone', '--listen', '127.0.0.1:18543', '--management-listen', '127.0.0.1:18544',
                                         '--engine-ports', '8300-8319', '--set', 'host.resource_policy.memory.system.managed_limit=96GiB', '--json'],
                                    stdout=rolelog, stderr=rolelog, env=env, start_new_session=True)
            owned.add(role.pid)
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                if role.poll() is not None:
                    raise RuntimeError('standalone exited')
                if re.search(r'"ready"\s*:\s*true', (a.work/'standalone.log').read_text()):
                    break
                time.sleep(1)
            else:
                raise TimeoutError('standalone readiness')
            credentials = list(state.rglob('identity/credentials'))+list((a.work/'xdg-state').rglob('identity/credentials'))
            if not credentials:
                raise RuntimeError('missing standalone credentials')
            command(cli+['deploy', 'model', '--file', a.work/'deployment.yaml'])
            deployed = True
            if gpu_pids():
                raise RuntimeError('GPU occupied before engine launch; no engine started')
            started = time.monotonic()
            command(cli+['start', 'deployment', recipe['name'], '--wait'])
            result['ready_s'] = time.monotonic()-started
            engine_pids.update(gpu_pids())
            native = '\n'.join(p.read_text(errors='replace') for p in (state/'logs').rglob('*.log'))
            result['fp8_kv_confirmed'] = bool(re.search('fp8_e4m3|float8_e4m3', native))
            result['fp8_attention_confirmed'] = 'Selected CutlassFP8ScaledMMLinearKernel for QuarkW8A8Fp8' in native
            result['load_completed'] = True
            save(a.work/'run.json', result)
            if not published and not result['fp8_kv_confirmed']:
                raise RuntimeError('FP8 KV not confirmed')
            if candidate and not result['fp8_attention_confirmed']:
                raise RuntimeError('Quark FP8 attention kernel not confirmed')
            if not a.load_only:
                common = ['--endpoint', 'http://127.0.0.1:18543/v1', '--model', recipe['name'], '--api-key-file', credentials[0]]
                command([python, PRECEDENT/'make-context.py', '--checkpoint', checkpoint, '--out', a.work/'context-prompts.json', '--rounds', '3', '--reasoning-effort', 'low'], cpu=True)
                command([python, PRECEDENT/'measure.py', *common, '--out', a.work/'measure.json', '--context-prompts', a.work/'context-prompts.json', '--rounds', '3', '--short-only'], timeout=28800)
                command([python, PRECEDENT/'final-probes.py', 'decode', *common, '--fixtures', a.fixtures, '--out', a.work/'decode.json'], timeout=28800)
                decoded = json.loads((a.work/'decode.json').read_text())
                decoded['method'] = 'Three single-stream streaming rounds at each exact input length; 512 completion-token cap; decode excludes first-token latency; aggregate includes prefill. Shared fixed exact-token prompt. Cache policy comes from retained native launch evidence.'
                save(a.work/'decode.json', decoded)
                if a.performance_only:
                    command([python, PRECEDENT/'final-probes.py', 'sweep', *common, '--fixtures', a.fixtures, '--out', a.work/'sweep.json'], timeout=28800)
                    swept = json.loads((a.work/'sweep.json').read_text())
                    swept['method'] = 'Three streaming rounds per exact input length and concurrency; 512 completion-token cap; decode excludes first-token latency; aggregate includes prefill. Prefix caching disabled by shipped recipe.'
                    save(a.work/'sweep.json', swept)
                if not a.performance_only:
                    command([Path.home()/'sglang-0.5.22-venv/bin/python', HERE/'accuracy.py', '--endpoint', 'http://127.0.0.1:18543', '--model', recipe['name'], '--api-key-file', credentials[0], '--out', a.work/'accuracy', '--max-tokens', '32768']+(['--screen'] if published else []), timeout=86400)
                if (candidate and not a.performance_only) or a.retrieval_copy:
                    for phase in ('retrieval', 'copy'):
                        command([python, PRECEDENT/'final-probes.py', phase, *common, '--fixtures', a.fixtures, '--out', a.work/(phase+'.json')], timeout=28800)
            result['status'] = 'measured'
        except Exception as exc:
            result.update(status='blocked', error=str(exc))
        finally:
            if deployed:
                try:
                    command(cli+['stop', 'deployment', recipe['name']], timeout=180)
                except Exception as exc:
                    result['stop_error'] = str(exc)
            if role is not None and role.poll() is None:
                role.send_signal(signal.SIGTERM)
                try:
                    role.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    result['cleanup_error'] = 'owned controller did not exit'
            stop.set(); watcher.join()
            result['memory_guard_triggered'] = guard_triggered.is_set()
            result['min_mem_available_bytes'] = min((s['mem_available_bytes'] for s in samples), default=None)
            result['peak_system_used_bytes'] = result['mem_total_bytes']-result['min_mem_available_bytes'] if samples else None
            result['gpu_pids_after'] = gpu_pids()
            native = '\n'.join(p.read_text(errors='replace') for p in (state/'logs').rglob('*.log'))
            selected = [line for line in native.splitlines() if re.search('quark|fp8|marlin|memory|KV cache|prefix|lmcache', line, re.I) and not re.search('secret|api.key|authorization|credentials', line, re.I)]
            (a.work/'engine-evidence.txt').write_text('\n'.join(selected)+'\n')
            save(a.work/'memory.json', samples)
            save(a.work/'run.json', result)
    print(json.dumps(result), flush=True)
    raise SystemExit(result['status'] != 'measured' or bool(result['gpu_pids_after']))


if __name__ == '__main__':
    main()
