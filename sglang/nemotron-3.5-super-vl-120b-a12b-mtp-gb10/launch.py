#!/usr/bin/env python3
"""Download the pinned package or launch its measured SGLang arguments."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is unavailable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('download', 'serve'))
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    config = json.loads((HERE / 'launch.json').read_text())
    source = config['source']
    model = args.model_dir.expanduser().absolute()
    if args.action == 'download':
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=source['repo'], revision=source['revision'],
                          allow_patterns=source['files'], local_dir=str(model))
        return 0
    missing = [name for name in source['files'] if not (model / name).is_file()]
    if missing:
        raise RuntimeError('Incomplete pinned package: ' + ', '.join(missing))
    python = args.python.expanduser().absolute()
    version = subprocess.check_output(
        [str(python), '-B', '-c',
         'import importlib.metadata; print(importlib.metadata.version("sglang"))'],
        text=True).strip()
    if version != config['engine_version']:
        raise RuntimeError('This recipe requires SGLang ' + config['engine_version'])
    if subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True).strip():
        raise RuntimeError('Compute device occupied; no unrelated process will be stopped')
    # [INFERENCE] 90 GiB: observed weights, bounded pools and runtime allowance.
    if available_memory() - 90 * GIB < 20 * GIB:
        raise RuntimeError('Planned footprint must leave at least 20 GiB available')
    unit = 'nemotron-super-mtp-' + uuid.uuid4().hex + '.scope'
    command = ['systemd-run', '--user', '--scope', '--unit=' + unit,
               '-p', 'MemoryMax=95G', '-p', 'MemorySwapMax=0', str(python),
               '-B', '-m', 'sglang.launch_server']
    command += [str(model) if item == '{model_path}' else item
                for item in config['arguments']]
    env = dict(os.environ, MAX_JOBS='2', PYTHONDONTWRITEBYTECODE='1')
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
    process = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True)
        while process.poll() is None:
            if available_memory() < 4 * GIB:
                raise RuntimeError('4 GiB available-memory guard triggered')
            time.sleep(0.5)
        return process.returncode
    finally:
        if process is not None:
            # Only this invocation's unique scope, including its workers.
            subprocess.run(['systemctl', '--user', 'stop', unit], check=False)
            if process.poll() is None:
                process.wait()


if __name__ == '__main__':
    raise SystemExit(main())
