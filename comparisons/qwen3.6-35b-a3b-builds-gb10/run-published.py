#!/usr/bin/env python3
"""Run the published NVIDIA recipe in an approved environment, one owned engine."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from runpy import run_path

HERE = Path(__file__).resolve().parent
helpers = run_path(str(HERE/'run-host.py'))
gpu_pids, memory = helpers['gpu_pids'], helpers['memory']
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--venv', type=Path, default=Path.home()/'vllm-0.32.0-venv')
p.add_argument('--recipe', choices=['nvidia', 'vllm-site'], default='nvidia')
p.add_argument('--draft', choices=['none','mtp','dflash2','dspark'], default='mtp')
p.add_argument('--depth', type=int)
p.add_argument('--screen', action='store_true')
p.add_argument('--long', action='store_true')
p.add_argument('--phase', choices=['all', 'accuracy', 'long', 'published8k', 'probes'], default='all')
p.add_argument('--fixtures', type=Path)
a = p.parse_args()
if a.depth is None:
    a.depth = 3 if a.recipe == 'vllm-site' else 4
if gpu_pids():
    raise SystemExit('occupied GPU; no launch')
a.work.mkdir(parents=True, exist_ok=False)
env = {**os.environ, 'PATH': str(a.venv/'bin')+os.pathsep+os.environ['PATH'],
       'HF_HOME': str(Path.home()/'qwen36-work/hf-cache')}
key = f'vl-nvidia-{a.draft}'+(f'-{a.depth}' if a.draft != 'none' else '')
model = 'q36-'+key
argv = [str(a.venv/'bin/vllm'), 'serve', str(a.checkpoint), '--host', '127.0.0.1', '--port', '8300',
        '--served-model-name', model, '--tensor-parallel-size', '1', '--trust-remote-code',
        '--kv-cache-dtype', 'fp8', '--attention-backend', 'flashinfer', '--moe-backend', 'marlin',
        '--gpu-memory-utilization', '0.8', '--max-model-len', '262144', '--max-num-seqs', '8',
        '--max-num-batched-tokens', '8192', '--enable-chunked-prefill', '--async-scheduling',
        '--enable-prefix-caching', '--load-format', 'safetensors', '--reasoning-parser', 'qwen3',
        '--tool-call-parser', 'qwen3_xml', '--enable-auto-tool-choice']
if a.recipe == 'vllm-site':
    env['VLLM_USE_RUST_FRONTEND'] = '1'
    for option, value in [('--gpu-memory-utilization', '0.5'), ('--load-format', 'fastsafetensors'),
                          ('--tool-call-parser', 'qwen3_coder')]:
        argv[argv.index(option)+1] = value
    argv += ['--language-model-only']
if a.draft != 'none':
    spec = {'method': 'dflash' if a.draft == 'dflash2' else a.draft, 'num_speculative_tokens': a.depth}
    if a.draft == 'mtp':
        spec['moe_backend'] = 'triton'
    else:
        from matrix import model_path
        spec['model'] = str(model_path(a.draft))
    argv += ['--speculative-config', json.dumps(spec)]
result = {'key': key, 'checkpoint': str(a.checkpoint), 'phase': a.phase, 'status': 'started',
          'lmcache': 'off', 'capyctl_version': None, 'launcher': 'native approved vLLM',
          'memory_before_bytes': memory(), 'settings': {'argv': argv[3:]},
          'command': argv,
          'client_python': str(Path.home()/'sglang-0.5.22-venv/bin/python'),
          'environment': {k: env.get(k) for k in ['PATH', 'HF_HOME', 'VLLM_USE_V1', 'VLLM_USE_RUST_FRONTEND', 'VLLM_ATTENTION_BACKEND', 'CUDA_VISIBLE_DEVICES']},
          'published_runtime': '0.28.0' if a.recipe == 'vllm-site' else '0.31.0+88641915.nv26.10.73893335',
          'published_recipe': 'recipes.vllm.ai GB10 NVFP4' if a.recipe == 'vllm-site' else 'NVIDIA/local-ai-inference-recipes@bb515086841e7a81e7d51925561338e7dee35023'}
engine = None
try:
    with (a.work/'vllm-fingerprint.json').open('wb') as f:
        subprocess.run([a.venv/'bin/python', HERE/'inspect-engine.py'], stdout=f, check=True, env=env)
    contexts = a.work/'context-prompts.json'
    subprocess.run([a.venv/'bin/python', HERE/'make-context.py', '--checkpoint', a.checkpoint, '--out', contexts], check=True, env=env)
    if gpu_pids():
        raise RuntimeError('GPU became occupied; no launch')
    with (a.work/'engine.log').open('wb') as log:
        started = time.monotonic()
        engine = subprocess.Popen(argv, stdout=log, stderr=log, env=env, start_new_session=True)
        while True:
            if engine.poll() is not None:
                raise RuntimeError(f'engine exited {engine.returncode}; see engine.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8300/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic()-started > 2100:
                raise TimeoutError('engine readiness')
            time.sleep(2)
        result.update(ready_s=time.monotonic()-started, ready_system_used_bytes=memory())
        client_python = Path.home()/'sglang-0.5.22-venv/bin/python'
        client = [str(client_python), str(HERE/'measure.py'), '--endpoint', 'http://127.0.0.1:8300/v1',
                  '--model', model, '--context-prompts', str(contexts)]
        if a.phase == 'all':
            subprocess.run(client+['--phase', 'speed', '--out', str(a.work/'measure.json')], check=True, env=env)
        if a.phase == 'published8k':
            subprocess.run(client+['--phase', 'published8k', '--out', str(a.work/'published8k.json')], check=True, env=env)
        if a.phase in ('all', 'accuracy'):
            subprocess.run([client_python, HERE/'accuracy.py', '--endpoint', 'http://127.0.0.1:8300',
                            '--model', model, '--out', a.work/'accuracy']+(['--screen'] if a.screen else []), check=True, env=env)
        if a.long or a.phase == 'long':
            subprocess.run(client+['--phase', 'long', '--out', str(a.work/'long.json')], check=True, env=env)
        if a.phase == 'probes':
            for phase in ('retrieval', 'copy'):
                subprocess.run([client_python, HERE/'final-probes.py', phase,
                                '--endpoint', 'http://127.0.0.1:8300', '--model', model,
                                '--fixtures', a.fixtures, '--out', a.work/(phase+'.json')],
                               check=True, env=env, timeout=28800)
        result['status'] = 'measured'
except BaseException as exc:
    result.update(status='failed', error=type(exc).__name__+': '+str(exc))
finally:
    if engine is not None and engine.poll() is None:
        os.killpg(engine.pid, signal.SIGTERM)
        try:
            engine.wait(timeout=180)
        except subprocess.TimeoutExpired:
            result['cleanup_error'] = 'owned engine process group did not exit after TERM'
    result['gpu_pids_after'] = gpu_pids()
    (a.work/'engine-evidence.txt').write_text(helpers['kernel_evidence'](a.work))
    (a.work/'run.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
raise SystemExit(result['status'] != 'measured' or bool(result['gpu_pids_after']))
