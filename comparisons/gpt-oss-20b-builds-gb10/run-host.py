#!/usr/bin/env python3
"""One owned CapyCTL session on host A; reuse the Gemma measurement method."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from matrix import DRAFTS, ROWS, deployment


def gpu_pids():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()


def memory():
    fields = {x.split(':')[0]: int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:', 'MemAvailable:'))}
    return fields['MemTotal']-fields['MemAvailable']


def llama_kv_evidence(model_file):
    """Owner-approved b12242 evidence when info logs omit KV allocation types."""
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if Path(argv[0]).name == 'llama-server' and model_file in argv:
                flags = {}
                for option in ('--cache-type-k', '--cache-type-v', '--ctx-size', '--parallel', '--flash-attn'):
                    if argv.count(option) == 1:
                        flags[option] = argv[argv.index(option)+1]
                matches.append({'pid': int(proc.name), 'flags': flags})
        except (OSError, UnicodeError, IndexError):
            continue
    evidence = {'method': 'owned native process argv; b12242 info logs omit KV types',
                'processes': matches, 'props_kv': {}, 'props_status': None}
    try:
        with urllib.request.urlopen('http://127.0.0.1:8300/props', timeout=10) as response:
            props = json.load(response)
        evidence['props_status'] = 'read'
        def select(value, prefix=''):
            if isinstance(value, dict):
                for key, child in value.items():
                    name = prefix+'.'+key if prefix else key
                    if key in ('cache_type_k', 'cache_type_v', 'type_k', 'type_v', 'n_ctx', 'n_ctx_slot', 'total_slots'):
                        evidence['props_kv'][name] = child
                    elif isinstance(child, dict):
                        select(child, name)
        select(props)
    except (OSError, ValueError) as exc:
        evidence['props_status'] = type(exc).__name__
    evidence['confirmed'] = (len(matches) == 1 and
                             all(matches[0]['flags'].get(k) == 'q8_0'
                                 for k in ('--cache-type-k', '--cache-type-v')))
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('key', choices=ROWS)
    p.add_argument('--capyctl', type=Path, default=Path.home()/'hostA-v020-assets/bin/capyctl-0.1.12')
    p.add_argument('--checkpoint')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--concurrency', default='1,2,4,8')
    p.add_argument('--long-only', action='store_true')
    p.add_argument('--short-only', action='store_true')
    p.add_argument('--baseline', type=Path, help='same-engine no-draft runtime/evidence directory')
    p.add_argument('--full-drafter', action='store_true', help='complete a winning drafter with c1/c2/c4/c8 and exact 24k probes')
    p.add_argument('--reasoning-effort', default='low', choices=('low', 'medium', 'high'))
    p.add_argument('--skip-accuracy', action='store_true')
    p.add_argument('--final-phase', choices=('decode', 'crossover', 'drafter-decode', 'retrieval', 'copy', 'sweep', 'acceptance', 'tool-diagnostic'))
    p.add_argument('--fixtures', type=Path)
    p.add_argument('--recipe-file', type=Path)
    p.add_argument('--require-fp8-attention', action='store_true')
    p.add_argument('--attention-backend', choices=('TRITON_ATTN', 'FLASHINFER'))
    p.add_argument('--coherence-check', action='store_true')
    a = p.parse_args()
    active = gpu_pids()
    engines = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                name = (proc/'comm').read_text().strip()
                if name == 'llama-server' or name.startswith(('capyctl', 'VLLM::', 'sglang::')):
                    engines.append((proc.name, name))
            except OSError:
                pass
    if active or engines:
        raise SystemExit(f'host occupied; no action taken: GPU PIDs={active}; engines={engines}')
    a.work.mkdir(parents=True, exist_ok=False)
    state = a.work/'state'
    env = {**os.environ, 'XDG_CONFIG_HOME': str(a.work/'config'), 'XDG_STATE_HOME': str(a.work/'xdg-state')}
    c = [str(a.capyctl), '--state-dir', str(state)]
    profile, engine, backend = ROWS[a.key]
    executable = Path.home()/({'sg522': 'sglang-0.5.22-venv', 'vl032': 'vllm-0.32.0-venv', 'vl033': 'vllm-0.33.0-venv', 'llama12242': 'llamacpp-b12242/build/bin/llama-server'}[profile])
    python = executable/'bin/python' if engine == 'vllm' else Path.home()/'sglang-0.5.22-venv/bin/python'
    profile_env = {}
    if engine != 'llamacpp':
        profile_env = {'CUDA_HOME': '/usr/local/cuda-13.0',
                       'PATH': str(executable/'bin')+':/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'}
    if engine == 'vllm':
        profile_env['TIKTOKEN_ENCODINGS_BASE'] = str(Path.home()/'models/gpt-oss-tokenizer')
    env.update(profile_env)
    root = Path(__file__).resolve().parents[2]
    if a.recipe_file:
        import yaml
        recipe = yaml.safe_load(a.recipe_file.read_text())
        profile = recipe['engine']
        recipe_substitutions = {}
        if a.checkpoint:
            recipe_substitutions['model'] = {'source': recipe['model'], 'local': a.checkpoint}
            recipe['model'] = str(Path(a.checkpoint).expanduser())
        if '/home/me/drafters/gpt-oss-20b-DFlash' in recipe['engine_config']['extra_args']:
            recipe_substitutions['/home/me/drafters/gpt-oss-20b-DFlash'] = str(Path.home()/'models/gpt-oss-20b-dflash')
        recipe['engine_config']['extra_args'] = [
            str(Path.home()/'models/gpt-oss-20b-dflash') if arg == '/home/me/drafters/gpt-oss-20b-DFlash' else arg
            for arg in recipe['engine_config']['extra_args']]
    else:
        recipe = deployment(a.key, a.checkpoint)
    if a.attention_backend:
        recipe['engine_config']['extra_args'] += ['--attention-backend', a.attention_backend]
    (a.work/'deployment.yaml').write_text(json.dumps(recipe, indent=2)+'\n')
    non_draft = json.loads(json.dumps(recipe['engine_config']))
    if backend == 'eagle3':
        non_draft['extra_args'] = non_draft['extra_args'][:-6]
    if a.key in DRAFTS:
        non_draft = deployment(DRAFTS[a.key]['baseline'], a.checkpoint)['engine_config']
        if a.attention_backend:
            non_draft['extra_args'] += ['--attention-backend', a.attention_backend]
    if (a.key in DRAFTS or backend == 'eagle3') and not a.baseline:
        raise SystemExit('drafter measurements require --baseline for the same-engine no-draft gate')
    result = {'key': a.key, 'checkpoint': recipe['model'], 'engine': engine, 'backend_requested': backend,
              'settings': recipe['engine_config'], 'non_draft_settings': non_draft,
              'env': [f'{k}={v}' for k, v in profile_env.items()],
              'drafter_path': str(Path.home()/'models/gpt-oss-20b-gguf/eagle3-gpt-oss-20b-Q8_0.gguf') if backend == 'eagle3' else None,
              'reasoning_effort': a.reasoning_effort, 'lmcache': 'off',
              'cold_definition': 'fresh engine process; OS file and JIT caches not cleared', 'memory_before_bytes': memory()}
    if a.key in DRAFTS:
        result['drafter_path'] = DRAFTS[a.key]['path']
        result['draft'] = DRAFTS[a.key]
    if result['drafter_path']:
        result['quality_scope'] = 'unqualified screen until exact-setting GSM8K/tools, divergence and output review pass'
    samples, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            data = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            apps = []
            for line in data.stdout.splitlines():
                pid, used = (s.strip() for s in line.split(',', 1))
                apps.append({'pid': int(pid), 'used_memory_bytes': int(used)*1024*1024 if used.isdigit() else None})
            samples.append({'unix_s': time.time(), 'system_used_bytes': memory(), 'gpu_compute_apps': apps})
            stop.wait(.5)
    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    role = None
    deployed = False
    with (a.work/'cli.log').open('wb') as log, (a.work/'standalone.log').open('wb') as rolelog:
        def command(argv, timeout=1800, run_env=env):
            log.write((json.dumps(list(map(str, argv)))+'\n').encode()); log.flush()
            subprocess.run(list(map(str, argv)), stdout=log, stderr=log, check=True, timeout=timeout, env=run_env)
        try:
            version = subprocess.check_output([str(a.capyctl), '--version'], text=True).strip()
            result['capyctl_version'] = version
            if version != 'capyctl 0.1.12':
                raise RuntimeError('requires CapyCTL 0.1.12')
            if not a.final_phase:
                command([python, Path(__file__).with_name('make-context.py'), '--checkpoint', Path.home()/'models/gpt-oss-20b', '--out', a.work/'context-prompts.json', '--rounds', a.rounds, '--reasoning-effort', a.reasoning_effort], run_env={**env, 'CUDA_VISIBLE_DEVICES': ''})
            profile_args = [arg for key, value in profile_env.items() if key not in ('CUDA_HOME', 'PATH')
                            for arg in ('--env', f'{key}={value}')]
            if engine == 'llamacpp':
                profile_args += ['--approve-option=--spec-draft-model']
            if a.key in DRAFTS:
                profile_args += ['--approve-option='+('--speculative-draft-model-path' if engine == 'sglang' else '--speculative-config')]
            command(c+['engine', 'add', executable, '--name', profile, '--approve-path', Path.home()/'models']+profile_args)
            with (a.work/'fingerprint.txt').open('wb') as f:
                if engine == 'llamacpp':
                    subprocess.run([str(executable), '--version'], stdout=f, stderr=f, check=True)
                else:
                    subprocess.run([str(executable/'bin/python'), str(Path(__file__).with_name('inspect-engine.py'))], stdout=f, stderr=f, check=True)
            command(c+['validate', 'config', '--file', a.work/'deployment.yaml'])
            result['config_validated'] = True
            if a.recipe_file:
                import hashlib
                result['recipe_source_sha256'] = hashlib.sha256(a.recipe_file.read_bytes()).hexdigest()
                result['recipe_substitutions'] = recipe_substitutions
            role = subprocess.Popen(c+['start', 'standalone', '--listen', '127.0.0.1:18543', '--management-listen', '127.0.0.1:18544', '--engine-ports', '8300-8319', '--set', 'host.resource_policy.memory.system.managed_limit=108GiB', '--set', 'host.resource_policy.memory.system.free_reserve=10GiB', '--json'], stdout=rolelog, stderr=rolelog, env=env)
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                if role.poll() is not None:
                    raise RuntimeError('standalone exited before readiness')
                if re.search(r'"ready"\s*:\s*true', (a.work/'standalone.log').read_text()):
                    break
                time.sleep(1)
            else:
                raise TimeoutError('standalone not ready')
            credentials = list(state.rglob('identity/credentials'))+list((a.work/'xdg-state').rglob('identity/credentials'))
            if not credentials:
                raise RuntimeError('missing standalone credentials')
            command(c+['deploy', 'model', '--file', a.work/'deployment.yaml'])
            deployed = True
            if gpu_pids():
                raise RuntimeError('GPU occupied before launch; no engine started')
            started = time.monotonic()
            command(c+['start', 'deployment', recipe['name'], '--wait'])
            result['ready_s'] = time.monotonic()-started
            with (a.work/'status.json').open('wb') as f:
                subprocess.run(c+['status', 'deployment', recipe['name'], '--json'], stdout=f, stderr=log, check=True, env=env)
            native_logs = '\n'.join(path.read_text(errors='replace') for path in (state/'logs').rglob('*.log'))
            expected_kv = r'q8_0' if engine == 'llamacpp' else r'fp8_e4m3|float8_e4m3'
            result['native_kv_confirmed'] = bool(re.search(expected_kv, native_logs))
            if engine == 'llamacpp':
                result['native_kv_evidence'] = llama_kv_evidence(str(Path(recipe['model'])/recipe['engine_config']['llamacpp']['gguf_file']))
                result['native_kv_confirmed'] = result['native_kv_evidence']['confirmed']
            if not result['native_kv_confirmed']:
                raise RuntimeError('requested KV dtype not confirmed in approved native evidence')
            if a.require_fp8_attention:
                fp8_lines = [line for line in native_logs.splitlines()
                             if re.search(r'fp8.*linear|linear.*fp8', line, re.I)]
                result['fp8_attention_kernel_lines'] = fp8_lines
                result['native_fp8_attention_confirmed'] = any(
                    'Selected CutlassFP8ScaledMMLinearKernel for QuarkW8A8Fp8' in line
                    for line in fp8_lines)
                result['fp8_attention_mapping'] = (
                    'Audited checkpoint: only attention q/k/v/o use QuarkW8A8Fp8; '
                    'experts retain MXFP4 and head is unquantized. Kernel log is global, not per-module.')
                if not result['native_fp8_attention_confirmed']:
                    raise RuntimeError('FP8 linear kernel for qkv/o attention not confirmed; no measurements run')
            if a.coherence_check:
                command([python, Path(__file__).with_name('public-coherence.py'),
                         '--credentials', credentials[0], '--model', recipe['name'],
                         '--out', a.work/'coherence.json'], timeout=1800)
            try:
                speed_args = ['--concurrency', '1,8' if backend == 'eagle3' and not a.full_drafter else a.concurrency]
                if (backend == 'eagle3' and not a.full_drafter) or a.short_only:
                    speed_args += ['--short-only']
                if a.long_only:
                    speed_args += ['--long-only']
                if result['drafter_path']:
                    speed_args += ['--native-metrics-endpoint', 'http://127.0.0.1:8300',
                                   '--baseline', str(a.baseline/'measure.json'), '--public-mtp-screen']
                if a.final_phase:
                    command([python, Path(__file__).with_name('final-probes.py'), a.final_phase,
                             '--endpoint', 'http://127.0.0.1:18543/v1', '--model', recipe['name'],
                             '--api-key-file', credentials[0], '--out', a.work/'measure.json',
                             '--fixtures', a.fixtures], timeout=28800)
                    if a.final_phase == 'sweep':
                        command(c+['stop', 'deployment', recipe['name']], timeout=180)
                        started = time.monotonic()
                        command(c+['start', 'deployment', recipe['name'], '--wait'])
                        result['ready_warm_s'] = time.monotonic()-started
                        command([python, Path(__file__).with_name('final-probes.py'), 'acceptance',
                                 '--endpoint', 'http://127.0.0.1:18543/v1', '--model', recipe['name'],
                                 '--api-key-file', credentials[0], '--out', a.work/'acceptance.json',
                                 '--fixtures', a.fixtures], timeout=3600)
                else:
                    command([sys.executable, Path(__file__).with_name('measure.py'), '--endpoint', 'http://127.0.0.1:18543/v1', '--model', recipe['name'], '--api-key-file', credentials[0], '--out', a.work/'measure.json', '--context-prompts', a.work/'context-prompts.json', '--rounds', a.rounds, '--reasoning-effort', a.reasoning_effort]+speed_args, timeout=14400)
            except subprocess.CalledProcessError as exc:
                result['speed_error'] = str(exc)
            if not a.skip_accuracy and not a.long_only and not a.final_phase:
                command([sys.executable, Path(__file__).with_name('accuracy.py'), '--endpoint', 'http://127.0.0.1:18543', '--model', recipe['name'], '--api-key-file', credentials[0], '--out', a.work/'accuracy', '--reasoning-effort', 'low', '--max-tokens', '8192'], timeout=14400)
            result['status'] = 'blocked' if result.get('speed_error') else 'measured'
            if result.get('speed_error'):
                result['error'] = result['speed_error']
        except Exception as exc:
            result.update(status='blocked', error=str(exc))
        finally:
            if deployed:
                try:
                    command(c+['stop', 'deployment', recipe['name']], timeout=180)
                except Exception as exc:
                    result['stop_error'] = str(exc)
                deadline = time.monotonic()+180
                while gpu_pids() and time.monotonic() < deadline:
                    time.sleep(1)
            if role is not None and role.poll() is None:
                role.send_signal(signal.SIGTERM)
                try:
                    role.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    result['cleanup_error'] = 'owned standalone did not exit after TERM'
            stop.set(); watcher.join()
            result['peak_system_used_bytes'] = max(s['system_used_bytes'] for s in samples)
            result['peak_delta_bytes'] = result['peak_system_used_bytes']-result['memory_before_bytes']
            totals = [sum(x['used_memory_bytes'] for x in s['gpu_compute_apps']) for s in samples if s['gpu_compute_apps'] and all(x['used_memory_bytes'] is not None for x in s['gpu_compute_apps'])]
            result['peak_gpu_compute_used_bytes'] = max(totals, default=None)
            result['gpu_pids_after'] = gpu_pids()
            evidence = []
            pattern = re.compile(r'kv.cache|dtype|token.*pool|cache|backend|mxfp4|marlin|quark|fp8.*linear|model loading|swa|reasoning.parser|tool.call.parser|n_ctx|n_seq|accept|draft|speculat', re.I)
            all_log_text = '\n'.join(path.read_text(errors='replace') for path in (state/'logs').rglob('*.log'))
            evidence.extend(line for line in all_log_text.splitlines()
                            if pattern.search(line) and not re.search(r'api.key|authorization|credentials|secret', line, re.I))
            launch_fields = {}
            for field in ('context_length', 'max_model_len', 'max_running_requests', 'max_num_seqs',
                          'cuda_graph_max_bs_decode', 'swa_full_tokens_ratio', 'kv_cache_dtype',
                          'reasoning_parser', 'tool_call_parser', 'moe_runner_backend', 'moe_backend',
                          'attention_backend', 'disable_radix_cache', 'enable_prefix_caching'):
                values = re.findall(r'\b'+field+r'''['"]?\s*[=:]\s*['"]?([A-Za-z0-9_.-]+)''', all_log_text)
                if values:
                    launch_fields[field] = values[-1]
            (a.work/'engine-evidence.txt').write_text('\n'.join(evidence)+'\n')
            def policies(value):
                if isinstance(value, dict):
                    return ([value['lmcache']] if 'lmcache' in value else []) + [p for child in value.values() for p in policies(child)]
                if isinstance(value, list):
                    return [p for child in value for p in policies(child)]
                return []
            status_path = a.work/'status.json'
            native = {'controller_lmcache': policies(json.loads(status_path.read_text())) if status_path.exists() else [],
                      'resolved_launch_fields': launch_fields,
                      'native_kv_lines': [line for line in evidence if re.search(r'fp8_e4m3|float8_e4m3|q8_0|KV cache|KV Cache', line)],
                      'cache_policy_lines': [line for line in evidence if re.search(r'prefix|radix|cache.*enabled|cache.*disabled', line, re.I)]}
            if result.get('native_kv_evidence'):
                native['owned_process_kv'] = result['native_kv_evidence']
            (a.work/'native-kv.json').write_text(json.dumps(native, indent=2)+'\n')
            (a.work/'memory.json').write_text(json.dumps(samples)+'\n')
            (a.work/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'measured' or bool(result['gpu_pids_after']) or 'cleanup_error' in result)


if __name__ == '__main__':
    main()
