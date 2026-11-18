#!/usr/bin/env python3
"""One host, one owned CapyCTL session; never stop an existing engine.

Run on the lab host, from a checkout containing tools/capyctl-bench.
Default is NVIDIA's build on SGLang with a 131k production configuration. A new checkpoint
plugs in with --checkpoint PATH and a key such as vl-capyctl-mtp-2.
"""
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
from matrix import deployment


def command(argv, log, *, env=None, timeout=1800):
    log.write((json.dumps(list(map(str, argv))) + '\n').encode())
    log.flush()
    return subprocess.run(list(map(str, argv)), stdout=log, stderr=log, env=env, timeout=timeout, check=True)


def gpu_pids():
    data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return [x.strip() for x in data.splitlines() if x.strip()]


def memory():
    fields = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return fields['MemTotal'] - fields['MemAvailable']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('key', nargs='?', default='sg-nvidia-none')
    p.add_argument('--capyctl', type=Path, default=os.environ.get('CAPYCTL'), required='CAPYCTL' not in os.environ)
    p.add_argument('--sglang-venv', type=Path, default=Path.home()/'sglang-0.5.22-venv')
    p.add_argument('--vllm-venv', type=Path, default=Path.home()/'vllm-0.32.0-lmcache-venv')
    p.add_argument('--checkpoint')
    p.add_argument('--drafter-path', type=Path, help='approved local metadata wrapper for the pinned drafter')
    p.add_argument('--deployment-file', type=Path, help='shipped YAML; substitute only target checkpoint and /home/me drafter prefix')
    p.add_argument('--full-sweep', action='store_true', help='Qwen run-5 full capyctl-bench sweep and warm restart')
    p.add_argument('--public-mtp-screen', action='store_true', help='public checkpoint MTP3 comparison; review output/acceptance before deciding native quality')
    p.add_argument('--probe-script', type=Path, help='native acceptance measurement script, run before the parser guard for a cold first request')
    p.add_argument('--probe-soak', action='store_true', help='include four simultaneous long-context retrieval requests')
    p.add_argument('--probe-arg', action='append', default=[], help='one additional argument for the selected native probe client')
    p.add_argument('--setting', action='append', default=[])
    p.add_argument('--baseline', type=Path)
    quality = p.add_mutually_exclusive_group()
    quality.add_argument('--quality-verdict', type=Path, help='approved exact-config GSM8K/tools comparison')
    quality.add_argument('--sweep-quality-verdict', type=Path, help='approved same drafter at another depth; screening only, best depth needs exact native quality')
    p.add_argument('--work', type=Path, required=True, help='new run directory, outside /tmp')
    p.add_argument('--concurrency', default='1,2,4,8')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--env', action='append', default=[], help='explicit non-secret ENGINE_VARIABLE=value')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--guard-only', action='store_true', help='capture a no-draft reference without speed workloads')
    mode.add_argument('--accuracy-only', action='store_true', help='canonical GSM8K-250 and 30 tool cases, after the speed matrix')
    args = p.parse_args()
    if args.work.resolve().is_relative_to(Path('/tmp')):
        p.error('use a disk-backed work directory, never /tmp')
    if args.key.split('-')[2] != 'none' and not args.baseline:
        p.error('every speculative config requires --baseline for the same engine/checkpoint/settings')
    if args.public_mtp_screen and (args.key.split('-')[1] not in ('nvidia', 'redhat', 'unsloth', 'r2') or args.key.split('-')[2:] != ['mtp', '3']):
        p.error('public screening is restricted to the requested public-build MTP3 comparison')
    active = gpu_pids()
    # Inspect executable basenames, not argv: the invoking shell can name engine flags.
    engines = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            name = (proc/'comm').read_text().strip()
            if name in ('capyctl', 'llama-server') or name.startswith(('VLLM::', 'sglang::')):
                engines.append((proc.name, name))
        except (OSError, ProcessLookupError):
            continue
    if active or engines:
        raise SystemExit(f'host occupied; no action taken: GPU PIDs={active}; engines={engines}')
    args.work.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({'XDG_CONFIG_HOME': str(args.work/'config'), 'XDG_STATE_HOME': str(args.work/'xdg-state')})
    for item in args.env:
        key, value = item.split('=', 1)
        if key not in ('VLLM_USE_V2_MODEL_RUNNER', 'VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_NVFP4_GEMM_BACKEND', 'VLLM_MOE_BACKEND'):
            p.error('only declared engine tuning variables may be recorded')
        env[key] = value
    state = args.work/'state'
    c = [args.capyctl, '--state-dir', state]
    sg = args.sglang_venv.expanduser()
    vl = args.vllm_venv.expanduser()
    recipe = deployment(args.key, args.checkpoint, args.setting, args.drafter_path)
    if args.deployment_file:
        data = subprocess.check_output([sg/'bin/python', '-c',
            'import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))',
            args.deployment_file], text=True)
        recipe = json.loads(data)
        recipe['model'] = args.checkpoint
        recipe['engine_config']['extra_args'] = [
            value.replace('/home/me/drafters/', str(Path.home()/'drafters')+'/')
            for value in recipe['engine_config']['extra_args']]
    name = recipe['name']
    (args.work/'deployment.yaml').write_text(json.dumps(recipe, indent=2)+'\n')
    root = Path(__file__).resolve().parents[2]
    samples, stopping = [], threading.Event()
    def sample():
        while not stopping.is_set():
            allocation = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10)
            apps = []
            for line in allocation.stdout.splitlines():
                pid, used = (field.strip() for field in line.split(',', 1))
                apps.append({'pid': int(pid), 'used_memory_bytes': int(used)*1024*1024 if used.isdigit() else None})
            samples.append({'unix_s': time.time(), 'system_used_bytes': memory(),
                            'gpu_compute_apps': apps, 'gpu_sample_error': allocation.stderr if allocation.returncode else None})
            stopping.wait(.5)
    watcher = threading.Thread(target=sample, daemon=True)
    role = None
    deployed = False
    before = memory()
    result = {'key': args.key, 'launcher': 'CapyCTL standalone', 'checkpoint': recipe['model'],
              'lmcache': recipe['lmcache'],
              'kv_cache_dtype': recipe['engine_config']['kv_cache_dtype'],
              'drafter_path': str(args.drafter_path.expanduser()) if args.drafter_path else None,
              'settings': args.setting, 'env': args.env, 'baseline': str(args.baseline) if args.baseline else None,
              'cold_definition': 'fresh engine process; OS file/JIT caches are NOT evicted', 'memory_before_bytes': before}
    watcher.start()
    with (args.work/'cli.log').open('wb') as log, (args.work/'standalone.log').open('wb') as rolelog:
        try:
            verdict_path = args.quality_verdict or args.sweep_quality_verdict
            if verdict_path:
                verdict = json.loads(verdict_path.read_text())
                identity = {k: result[k] for k in ('key', 'checkpoint', 'settings', 'env', 'drafter_path')}
                approved_identity = verdict.get('candidate', {})
                if args.sweep_quality_verdict:
                    source_key = approved_identity.get('key', '').split('-')
                    target_key = args.key.split('-')
                    if len(source_key) != 4 or len(target_key) != 4 or source_key[:3] != target_key[:3]:
                        raise RuntimeError('sweep approval must belong to the same engine/build/drafter')
                    identity['key'] = approved_identity['key']
                if not verdict.get('approved') or approved_identity != identity:
                    raise RuntimeError('quality verdict must approve this checkpoint/settings/env/drafter path')
                result['quality_verdict'] = str(verdict_path)
                result['quality_scope'] = ('family screening; best depth requires exact native quality'
                                           if args.sweep_quality_verdict else 'exact configuration')
            if not args.accuracy_only and not args.guard_only and not args.full_sweep:
                command([sg/'bin/python', Path(__file__).with_name('make-context.py'),
                         '--checkpoint', recipe['model'], '--out', args.work/'context-prompts.json',
                         '--rounds', str(args.rounds)], log, env={**env, 'CUDA_VISIBLE_DEVICES': ''})
            for profile, venv, option in (('sg522', sg, '--speculative-draft-model-path'), ('vl032', vl, '--speculative-config')):
                command(c+['engine', 'add', venv, '--name', profile, '--approve-option='+option, '--approve-path', Path.home()/'models', '--approve-path', Path.home()/'drafters'], log, env=env)
            if args.deployment_file:
                venv = sg if args.key.startswith('sg-') else vl
                option = '--speculative-draft-model-path' if args.key.startswith('sg-') else '--speculative-config'
                command(c+['engine', 'add', venv, '--name', recipe['engine'], '--approve-option='+option,
                           '--approve-path', Path.home()/'models', '--approve-path', Path.home()/'drafters'], log, env=env)
                command(c+['validate', 'config', '--file', args.work/'deployment.yaml'], log, env=env)
            for tag, venv in (('sglang', sg), ('vllm', vl)):
                with (args.work/(tag+'-fingerprint.txt')).open('wb') as fingerprint:
                    command([venv/'bin/python', Path(__file__).with_name('inspect-engine.py')], fingerprint, env=env)
            command([args.capyctl, '--version'], log)
            role = subprocess.Popen(list(map(str, c+['start', 'standalone', '--listen', '127.0.0.1:18543', '--management-listen', '127.0.0.1:18544', '--engine-ports', '8300-8319', '--set', 'host.resource_policy.memory.system.managed_limit=100GiB', '--set', 'host.resource_policy.memory.system.free_reserve=10GiB', '--json'])), stdout=rolelog, stderr=rolelog, env=env)
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                if role.poll() is not None:
                    raise RuntimeError('standalone exited before readiness')
                if re.search(r'"ready"\s*:\s*true', (args.work/'standalone.log').read_text()):
                    break
                time.sleep(1)
            else:
                raise TimeoutError('standalone not ready after 180 seconds')
            credentials = list(state.rglob('identity/credentials')) + list((args.work/'xdg-state').rglob('identity/credentials'))
            if not credentials:
                raise RuntimeError('no standalone credential file found')
            command(c+['deploy', 'model', '--file', args.work/'deployment.yaml'], log, env=env)
            deployed = True
            occupied = gpu_pids()
            if occupied:
                raise RuntimeError(f'GPU became occupied before engine launch; no engine started: {occupied}')
            started = time.monotonic()
            command(c+['start', 'deployment', name, '--wait'], log, env=env)
            result['ready_s'] = time.monotonic()-started
            result['ready_system_used_bytes'] = memory()
            with (args.work/'status.json').open('wb') as status:
                subprocess.run(list(map(str, c+['status', 'deployment', name, '--json'])), stdout=status, stderr=log, env=env, check=True, timeout=60)
            measure = [sys.executable, Path(__file__).with_name('measure.py'), '--recipes', root,
                       '--endpoint', 'http://127.0.0.1:18543/v1', '--model', name,
                       '--api-key-file', credentials[0], '--out', args.work/'measure.json',
                       '--concurrency', args.concurrency, '--rounds', str(args.rounds)]
            measure += ['--guard-only'] if args.guard_only or args.accuracy_only or args.full_sweep else ['--context-prompts', args.work/'context-prompts.json']
            if verdict_path:
                measure += ['--quality-verdict', verdict_path]
            if args.baseline:
                baseline_run = json.loads((args.baseline.parent/'run.json').read_text())
                if (baseline_run['checkpoint'], baseline_run['settings'], baseline_run['env']) != (result['checkpoint'], result['settings'], result['env']) or baseline_run['key'].split('-')[0] != args.key.split('-')[0] or baseline_run['key'].split('-')[2] != 'none':
                    raise RuntimeError('baseline must have same checkpoint, engine, settings/env and no drafter')
                measure += ['--baseline', args.baseline]
            result['mode'] = 'accuracy-only' if args.accuracy_only else 'guard-only' if args.guard_only else 'speed'
            if args.public_mtp_screen:
                result['quality_scope'] = 'owner-authorized public-build MTP3 screening; output/acceptance review required, not native task approval'
                measure += ['--public-mtp-screen']
            if args.probe_script:
                result['mode'] = 'native acceptance probes'
                probe = [sg/'bin/python', args.probe_script, '--endpoint', 'http://127.0.0.1:18543/v1',
                         '--model', name, '--api-key-file', credentials[0], '--checkpoint', recipe['model'],
                         '--out', args.work/'probes.json']
                if args.probe_soak:
                    probe += ['--soak']
                probe += args.probe_arg
                command(probe, log, env={**env, 'CUDA_VISIBLE_DEVICES': ''}, timeout=14400)
            command(measure, log, env=env)
            if args.accuracy_only:
                command([sys.executable, Path(__file__).with_name('accuracy.py'),
                         '--endpoint', 'http://127.0.0.1:18543', '--model', name,
                         '--api-key-file', credentials[0], '--out', args.work/'accuracy'], log, env=env, timeout=7200)
            if args.full_sweep:
                result['mode'] = 'shipped full sweep'
                command([sys.executable, root/'tools/capyctl-bench/capyctl_bench.py', 'run',
                         '--endpoint', 'http://127.0.0.1:18543/v1', '--api-key-file', credentials[0],
                         '--model', name, '--label', name, '--concurrency', '1-8', '--rounds', '5',
                         '--context-sweep', '0.5k,1k,2k,4k,8k,16k,32k', '--runs', '3',
                         '--out', args.work/'full-sweep.json'], log, env=env, timeout=7200)
                command(c+['stop', 'deployment', name], log, env=env, timeout=180)
                deadline = time.monotonic()+180
                while gpu_pids() and time.monotonic() < deadline:
                    time.sleep(1)
                if gpu_pids():
                    raise RuntimeError('GPU still occupied before warm restart; refusing launch')
                started = time.monotonic()
                command(c+['start', 'deployment', name, '--wait'], log, env=env)
                result['warm_ready_s'] = time.monotonic()-started
            result['status'] = 'measured'
        except Exception as exc:
            result.update(status='failed', error=str(exc))
        finally:
            # Only stop the deployment and process created by this invocation.
            if deployed:
                try:
                    command(c+['stop', 'deployment', name], log, env=env, timeout=180)
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
                    result['cleanup_error'] = 'owned standalone did not exit after TERM; inspect before another run'
            stopping.set()
            watcher.join()
            result['peak_system_used_bytes'] = max(x['system_used_bytes'] for x in samples)
            result['peak_delta_bytes'] = result['peak_system_used_bytes']-before
            gpu_totals = [sum(app['used_memory_bytes'] for app in sample['gpu_compute_apps'])
                          for sample in samples if sample['gpu_compute_apps']
                          and all(app['used_memory_bytes'] is not None for app in sample['gpu_compute_apps'])]
            result['peak_gpu_compute_used_bytes'] = max(gpu_totals, default=None)
            result['gpu_allocation_sample_errors'] = sum(sample['gpu_sample_error'] is not None for sample in samples)
            result['gpu_pids_after'] = gpu_pids()
            def cache_status(value):
                if isinstance(value, dict):
                    return ([value['lmcache']] if 'lmcache' in value else []) + [
                        item for child in value.values() for item in cache_status(child)]
                if isinstance(value, list):
                    return [item for child in value for item in cache_status(child)]
                return []
            evidence = []
            pattern = re.compile(r'kv.cache|KV cache|dtype|token.*pool|max.*running|accept.*len|accepted|draft.*token|attention.backend|MoE.*backend', re.I)
            for path in (state/'logs').rglob('*.log'):
                for line in path.read_text(errors='replace').splitlines():
                    if pattern.search(line):
                        evidence.append(line)
            (args.work/'engine-evidence.txt').write_text('\n'.join(evidence)+'\n')
            if (args.work/'status.json').exists():
                native_kv = {'lmcache': cache_status(json.loads((args.work/'status.json').read_text())),
                             'native_kv_lines': [line for line in evidence if
                                                'fp8_e4m3' in line or 'torch.float8_e4m3fn' in line
                                                or 'GPU KV cache size' in line],
                             'source': 'selected native logs and controller status; no host identity'}
                (args.work/'native-kv.json').write_text(json.dumps(native_kv, indent=2)+'\n')
            (args.work/'memory.json').write_text(json.dumps(samples)+'\n')
            (args.work/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'measured' or bool(result['gpu_pids_after']) or 'cleanup_error' in result)


if __name__ == '__main__':
    main()
