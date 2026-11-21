#!/usr/bin/env python3
"""One host, one owned CapyCTL session. Refuse any occupied GPU."""
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
from kernel_evidence import extract as kernel_evidence


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return [int(p) for p in text.split() if p.isdigit()]


def memory():
    fields = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    return fields['MemTotal']-fields['MemAvailable']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('key')
    p.add_argument('--capyctl', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--phase', choices=('all', 'speed', 'long', 'accuracy', 'probes'), default='all')
    p.add_argument('--screen', action='store_true', help='50-case GSM8K/HumanEval screen; never full qualification')
    p.add_argument('--with-long', action='store_true', help='include the context sweep after the selected phase')
    p.add_argument('--sglang-venv', type=Path, default=Path.home()/'sglang-0.5.22-venv')
    p.add_argument('--vllm-venv', type=Path, default=Path.home()/'vllm-0.32.0-venv')
    p.add_argument('--deployment-file', type=Path)
    p.add_argument('--fixtures', type=Path)
    a = p.parse_args()
    occupied = gpu_pids()
    if occupied:
        raise SystemExit('host occupied; no launch; GPU PIDs='+str(occupied))
    a.work.mkdir(parents=True, exist_ok=False)
    state = a.work/'state'
    env = {**os.environ, 'XDG_CONFIG_HOME': str(a.work/'config'), 'XDG_STATE_HOME': str(a.work/'xdg-state'),
           'HF_HOME': str(Path.home()/'qwen36-work/hf-cache'), 'MAX_JOBS': '2',
           'PATH': str((a.sglang_venv if a.key.startswith('sg-') else a.vllm_venv)/'bin')+os.pathsep+os.environ['PATH']}
    recipe = deployment(a.key, a.checkpoint)
    if a.deployment_file:
        import yaml
        recipe = yaml.safe_load(a.deployment_file.read_text())
        recipe['model'] = {
            'source': {'type': 'local', 'path': a.checkpoint},
            'draft': {'type': 'local', 'path': str(Path.home()/'models/qwen36-dflash2-51ef7b6923ad')},
        }
    (a.work/'deployment.yaml').write_text(json.dumps(recipe, indent=2)+'\n')
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    c = [str(a.capyctl), '--state-dir', str(state)]
    result = {'key': a.key, 'checkpoint': recipe['model'], 'phase': a.phase, 'lmcache': 'off', 'capyctl_version': '0.1.12',
              'settings': recipe['engine_config'], 'status': 'started',
              'cold_definition': 'fresh engine process; OS file and compiled-kernel caches retained',
              'memory_before_bytes': memory()}
    samples, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            samples.append({'unix_s': time.time(), 'system_used_bytes': memory()})
            stop.wait(.5)
    watcher = threading.Thread(target=sample, daemon=True)
    role, deployed = None, False
    with (a.work/'cli.log').open('wb') as log, (a.work/'standalone.log').open('wb') as rolelog:
        def command(argv, timeout=14400):
            subprocess.run(list(map(str, argv)), stdout=log, stderr=log, env=env, check=True, timeout=timeout)
        watcher.start()
        try:
            if subprocess.check_output([a.capyctl, '--version'], text=True).strip() != 'capyctl 0.1.12':
                raise RuntimeError('requires released CapyCTL 0.1.12')
            for tag, venv in [('sglang', a.sglang_venv), ('vllm', a.vllm_venv)]:
                with (a.work/(tag+'-fingerprint.json')).open('wb') as fingerprint:
                    subprocess.run([venv/'bin/python', here/'inspect-engine.py'], stdout=fingerprint,
                                   stderr=log, env=env, check=True, timeout=120)
            command([a.sglang_venv/'bin/python', here/'make-context.py', '--checkpoint', a.checkpoint or recipe['model'],
                     '--out', a.work/'context-prompts.json'])
            for profile, venv, option in [('sg522', a.sglang_venv, '--speculative-draft-model-path'),
                                           (recipe['engine'] if a.deployment_file else 'vl032', a.vllm_venv, '--speculative-config')]:
                command(c+['engine', 'add', venv, '--name', profile, '--approve-option='+option,
                           '--approve-path', Path.home()/'models'])
            role = subprocess.Popen(c+['start', 'standalone', '--listen', '127.0.0.1:18543',
                     '--management-listen', '127.0.0.1:18544', '--engine-ports', '8300-8319',
                     '--set', 'host.resource_policy.memory.system.managed_limit=110GiB',
                     '--set', 'host.resource_policy.memory.system.free_reserve=8GiB', '--json'],
                     stdout=rolelog, stderr=rolelog, env=env)
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                if role.poll() is not None:
                    raise RuntimeError('owned standalone exited before readiness')
                if re.search(r'"ready"\s*:\s*true', (a.work/'standalone.log').read_text()):
                    break
                time.sleep(1)
            else:
                raise TimeoutError('standalone readiness deadline')
            credentials = list(state.rglob('identity/credentials'))+list((a.work/'xdg-state').rglob('identity/credentials'))
            if not credentials:
                raise RuntimeError('standalone credentials missing')
            command(c+['deploy', 'model', '--file', a.work/'deployment.yaml'])
            deployed = True
            if gpu_pids():
                raise RuntimeError('GPU became occupied; refusing engine launch')
            started = time.monotonic()
            command(c+['start', 'deployment', recipe['name'], '--wait'], timeout=2100)
            result.update(ready_s=time.monotonic()-started, ready_system_used_bytes=memory())
            status = json.loads(subprocess.check_output(c+['status', 'deployment', recipe['name'], '--json'], env=env, text=True, timeout=60))
            def cache_status(value):
                if isinstance(value, dict):
                    return ([value['lmcache']] if 'lmcache' in value else []) + [
                        item for child in value.values() for item in cache_status(child)]
                if isinstance(value, list):
                    return [item for child in value for item in cache_status(child)]
                return []
            result['reported_lmcache'] = cache_status(status)
            client = [a.sglang_venv/'bin/python', here/'measure.py', '--endpoint', 'http://127.0.0.1:18543/v1',
                      '--model', recipe['name'], '--api-key-file', credentials[0],
                      '--context-prompts', a.work/'context-prompts.json']
            if a.phase in ('all', 'speed'):
                command(client+['--phase', 'speed', '--out', a.work/'measure.json'])
            if a.phase in ('all', 'accuracy'):
                command([a.sglang_venv/'bin/python', here/'accuracy.py', '--endpoint', 'http://127.0.0.1:18543',
                         '--model', recipe['name'], '--api-key-file', credentials[0], '--out', a.work/'accuracy']
                        + (['--screen'] if a.screen else []), timeout=28800)
            if a.phase == 'long' or a.with_long:
                command(client+['--phase', 'long', '--out', a.work/'long.json'])
            if a.phase == 'probes':
                if a.deployment_file:
                    command(client+['--phase', 'shipped', '--out', a.work/'shipped.json'])
                for phase in ('retrieval', 'copy'):
                    command([a.sglang_venv/'bin/python', here/'final-probes.py', phase,
                             '--endpoint', 'http://127.0.0.1:18543', '--model', recipe['name'],
                             '--api-key-file', credentials[0], '--fixtures', a.fixtures,
                             '--out', a.work/(phase+'.json')], timeout=28800)
            result['status'] = 'measured'
        except Exception as exc:
            result.update(status='failed', error=type(exc).__name__+': '+str(exc))
        finally:
            if deployed:
                try:
                    command(c+['stop', 'deployment', recipe['name']], timeout=180)
                except Exception as exc:
                    result['stop_error'] = type(exc).__name__
            if role is not None and role.poll() is None:
                role.send_signal(signal.SIGTERM)
                try:
                    role.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    result['cleanup_error'] = 'owned standalone did not exit after TERM'
            stop.set()
            watcher.join()
            result['peak_system_used_bytes'] = max((x['system_used_bytes'] for x in samples), default=None)
            result['peak_delta_bytes'] = result['peak_system_used_bytes']-result['memory_before_bytes']
            result['gpu_pids_after'] = gpu_pids()
            # Only explicit kernel/dtype lines leave the native runtime tree.
            (a.work/'engine-evidence.txt').write_text(kernel_evidence(state))
            (a.work/'memory.json').write_text(json.dumps(samples)+'\n')
            (a.work/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return int(result['status'] != 'measured' or bool(result['gpu_pids_after']) or 'cleanup_error' in result)

if __name__ == '__main__':
    raise SystemExit(main())
