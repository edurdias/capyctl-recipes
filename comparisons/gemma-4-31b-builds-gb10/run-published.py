#!/usr/bin/env python3
"""Exercise the existing Google QAT recipe with only a local checkpoint substitution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import yaml
from measure import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
common = load('gemma31_run_host', HERE / 'run-host.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--capyctl', type=Path, required=True)
    parser.add_argument('--quality-only', action='store_true')
    args = parser.parse_args()
    if common.gpu_pids() or common.available() < 100 * common.GIB:
        raise SystemExit('GPU occupied or insufficient planned memory headroom')
    args.work.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'vllm/gemma-4-31b-w4a16-32gib-gb10/deployment.yaml'
    recipe = yaml.safe_load(source.read_text())
    original = recipe['model']
    recipe['model'] = str(args.models / 'google')
    deployment = args.work / 'deployment.yaml'
    deployment.write_text(yaml.safe_dump(recipe, sort_keys=False))
    home = Path.home()
    state = args.work / 'state'
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'MAX_JOBS': '2',
           'XDG_CONFIG_HOME': str(args.work / 'config'), 'XDG_STATE_HOME': str(args.work / 'xdg-state'),
           'CAPYCTL_MANAGEMENT_ADDR': '127.0.0.1:18632'}
    cli = [str(args.capyctl), '--state-dir', str(state)]
    unit = 'gemma31-published-' + str(os.getpid())
    result = {'row': 'published-vl030', 'status': 'starting', 'recipe_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'only_recipe_substitution': {'original': original, 'local': '/home/me/gemma31-work/models/google'},
              'memory_scope_limit_gib': 95, 'guard_gib': 4, 'planned_peak_gib_inference': 65,
              'quality_only': args.quality_only,
              'mem_available_before_bytes': common.available(), 'measurements': 'published recipe through isolated controller endpoint'}
    samples, stop, guarded = [], threading.Event(), threading.Event()
    role, deployed = None, False

    def stop_scope():
        subprocess.run(['systemctl', '--user', 'stop', unit + '.scope'], capture_output=True, timeout=180)

    def watch():
        while not stop.wait(.5):
            value = common.available()
            samples.append({'unix_s': time.time(), 'mem_available_bytes': value})
            if value < 4 * common.GIB:
                guarded.set()
                stop_scope()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    with (args.work / 'cli.log').open('wb') as log, (args.work / 'standalone.log').open('wb') as rolelog:
        def command(parts, timeout=1800):
            subprocess.run(list(map(str, parts)), env=env, stdout=log, stderr=log, check=True, timeout=timeout)
        try:
            result['capyctl_version'] = subprocess.check_output([str(args.capyctl), '--version'], text=True).strip()
            command(cli + ['engine', 'add', home / 'vllm-0.30.0-venv', '--name', 'vllm', '--approve-path', args.models])
            command(cli + ['validate', 'config', '--file', deployment])
            role = subprocess.Popen(['systemd-run', '--user', '--scope', '--unit', unit, '-p', 'MemoryMax=95G',
                                     *cli, 'start', 'standalone', '--listen', '127.0.0.1:18631',
                                     '--management-listen', '127.0.0.1:18632', '--engine-ports', '18633-18639', '--json'],
                                    env=env, stdout=rolelog, stderr=rolelog, start_new_session=True)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if role.poll() is not None:
                    raise RuntimeError('isolated standalone exited')
                credentials = list(state.rglob('identity/credentials')) + list((args.work / 'xdg-state').rglob('identity/credentials'))
                ready_lines = (args.work / 'standalone.log').read_text(errors='replace').splitlines()
                if credentials and any('"ready":true' in line.replace(' ', '') for line in ready_lines):
                    break
                time.sleep(1)
            else:
                raise TimeoutError('isolated controller credentials/readiness')
            command(cli + ['deploy', 'model', '--file', deployment])
            deployed = True
            started = time.monotonic()
            command(cli + ['start', 'deployment', recipe['name'], '--wait'])
            result.update(load_completed=True, ready_s=time.monotonic() - started, mem_available_ready_bytes=common.available())
            common.save(args.work / 'run.json', result)
            python = home / 'sglang-0.5.22-venv/bin/python'
            credentials_arg = ['--api-key-file', credentials[0]]
            if not args.quality_only:
                command([python, '-B', HERE / 'measure.py', '--checkpoint', args.models / 'google',
                         '--context', '32768', '--endpoint', 'http://127.0.0.1:18631/v1', '--model', recipe['name'],
                         '--out', args.work / 'measure.json', *credentials_arg], timeout=28800)
            command([python, '-B', HERE / 'accuracy.py', '--endpoint', 'http://127.0.0.1:18631',
                     '--model', recipe['name'], '--out', args.work / 'screen', '--screen',
                     '--checkpoint', args.models / 'google', '--context', '32768', *credentials_arg], timeout=86400)
            if not args.quality_only:
                command([python, '-B', HERE / 'gates.py', '--checkpoint', args.models / 'google',
                         '--context', '32768', '--endpoint', 'http://127.0.0.1:18631', '--model', recipe['name'],
                         '--out', args.work / 'gates.json', '--retrieval-copy-only', *credentials_arg], timeout=86400)
            result['status'] = 'measured'
        except Exception as exc:
            result.update(status='blocked', error=str(exc).replace(str(home), '/home/me'))
        finally:
            if deployed:
                try:
                    command(cli + ['stop', 'deployment', recipe['name']], timeout=180)
                except Exception as exc:
                    result['stop_error'] = str(exc).replace(str(home), '/home/me')
            try:
                props = subprocess.run(['systemctl', '--user', 'show', unit + '.scope', '-p', 'MemoryPeak'], capture_output=True, text=True)
                result['scope_memory'] = props.stdout.splitlines()
                stop_scope()
            except Exception as exc:
                result['cleanup_error'] = str(exc)
            if role is not None:
                try:
                    role.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(role.pid, signal.SIGTERM)
            stop.set()
            watcher.join(timeout=10)
            result.update(memory_guard_triggered=guarded.is_set(), gpu_pids_after=common.gpu_pids())
            result['min_mem_available_bytes'] = min((sample['mem_available_bytes'] for sample in samples), default=common.available())
            result['peak_mem_available_drop_bytes'] = result['mem_available_before_bytes'] - result['min_mem_available_bytes']
            common.save(args.work / 'memory.json', samples)
            common.save(args.work / 'run.json', result)
    print(json.dumps(result), flush=True)
    raise SystemExit(result['status'] == 'blocked' or bool(result['gpu_pids_after']))


if __name__ == '__main__':
    main()
