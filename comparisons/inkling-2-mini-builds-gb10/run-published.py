#!/usr/bin/env python3
"""Run one exact published Inkling tier, with owned-container memory protection."""
import argparse
import json
import os
import re
from pathlib import Path
from runpy import run_path
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
IMAGE = 'vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e'
REVISION = 'ed40e7da625f594fc9c4990f5f66be24873c22d7'
helpers = run_path(str(HERE/'run-host.py'))
gpu_pids, memory_fields = helpers['gpu_pids'], helpers['memory_fields']
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('key', choices=['tv-std', 'tv-fp8'])
p.add_argument('--work', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, default=Path.home()/'models/Inkling-2-Mini-ed40e7da')
p.add_argument('--client-python', type=Path, default=Path.home()/'sglang-0.5.22-venv/bin/python')
p.add_argument('--phase', choices=['all','sanity','speed','long','probe','accuracy','probes','needle950'], default='all')
p.add_argument('--needle950', action='store_true')
a = p.parse_args()
occupied = gpu_pids()
os.environ['HF_HOME'] = str(Path.home()/'inkling-work/hf-cache')
if occupied:
    raise SystemExit('occupied GPU; no launch; PIDs='+str(occupied))
if (a.checkpoint/'REVISION').read_text().strip() != REVISION:
    raise SystemExit('checkpoint revision mismatch')
receipt = json.loads((a.checkpoint/'download-verification.json').read_text())
if not receipt['verified'] or receipt['revision'] != REVISION:
    raise SystemExit('checkpoint verification missing')
a.work.mkdir(parents=True, exist_ok=False)
name = 'inkling-bench-'+a.key+'-'+str(os.getpid())
endpoint, model = 'http://127.0.0.1:18120', 'inkling-2-mini'
argv = ['docker', 'run', '-d', '--name', name, '--label', 'inkling.comparison=owned', '--gpus', 'all', '--ipc=host', '--restart', 'no',
        '-p', '127.0.0.1:18120:8000', '-v', str(a.checkpoint)+':/models/Inkling-2-Mini:ro', IMAGE,
        '/models/Inkling-2-Mini', '--served-model-name', model, 'thinkingmachines/Inkling-2-Mini', '--tensor-parallel-size', '1',
        '--max-model-len', '262144' if a.key == 'tv-std' else '1048576']
argv += ['--gpu-memory-utilization','0.80'] if a.key == 'tv-std' else ['--kv-cache-memory-bytes','79456894976','--kv-cache-dtype','fp8']
argv += ['--reasoning-parser','inkling','--enable-auto-tool-choice','--tool-call-parser','inkling',
         '--speculative-config','{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}']
result = {'key': a.key, 'checkpoint': 'thinkingmachines/Inkling-2-Mini@'+REVISION, 'phase': a.phase, 'status': 'started',
          'launcher': 'published community container', 'image': IMAGE, 'argv': argv[argv.index(IMAGE):],
          'community_revision': '300942daf6be61f39a7e6c6055bd7cb6e49c4e7c',
          'deviations': ['immutable image and weight pins', 'owned unique container; no replacement of existing containers',
                         'loopback-only port publication', 'sudo cache-drop step skipped; no sudo invoked'],
          'quality_scope': 'full' if a.key == 'tv-std' else 'screen-50', 'memory_before': memory_fields(), 'completed_phases': []}
stop, killed = threading.Event(), threading.Event()
container = None

def save():
    (a.work/'run.json').write_text(json.dumps(result, indent=2)+'\n')

def watch():
    with (a.work/'memory.jsonl').open('w') as f:
        while not stop.is_set():
            fields = memory_fields()
            f.write(json.dumps({'unix_s': time.time(), **fields})+'\n')
            f.flush()
            if fields['MemAvailable'] < 3*1024**3:
                killed.set()
                subprocess.run(['docker','kill',container], capture_output=True)
                return
            stop.wait(2)

def command(args, output=None, timeout=28800):
    print(json.dumps({'phase': str(args[1]) if len(args)>1 else str(args)}), flush=True)
    if output:
        with output.open('w') as f:
            subprocess.run([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    else:
        subprocess.run([str(x) for x in args], check=True, timeout=timeout)
    if killed.is_set():
        raise RuntimeError('owned container killed: MemAvailable below 3 GiB')

try:
    save()
    contexts, fixtures = a.work/'context-prompts.json', a.work/'fixtures.json'
    if a.phase in ('all','speed','long'):
        command([a.client_python,HERE/'make-context.py','--checkpoint',a.checkpoint,'--out',contexts])
    if a.phase in ('all','probes'):
        command([a.client_python,HERE/'final-probes.py','prepare','--checkpoint',a.checkpoint,'--out',fixtures])
    # Download pinned evaluation data before allocating the shared-memory pool.
    if a.phase in ('all','probe','accuracy'):
        command([a.client_python,'-c',"from datasets import load_dataset; d=load_dataset('openai/openai_humaneval',revision='7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',split='test'); assert len(d)==164"])
    setup = json.loads((HERE/'bench/setup.json').read_text())
    assert setup['image'] == IMAGE and setup['pull_completed']
    result['image_versions'] = setup['image_versions']
    if gpu_pids():
        raise RuntimeError('GPU became occupied; no launch')
    started = time.monotonic()
    container = subprocess.check_output(argv, text=True).strip()
    result['container_id'] = container
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    while True:
        if killed.is_set():
            raise RuntimeError('owned container killed: MemAvailable below 3 GiB')
        state = subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',container],text=True).strip()
        if state != 'true':
            raise RuntimeError('container exited before readiness; see engine.log')
        try:
            with urllib.request.urlopen(endpoint+'/health',timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic()-started > 2400:
            raise TimeoutError('container readiness')
        time.sleep(2)
    result.update(ready_s=time.monotonic()-started, memory_ready=memory_fields())
    save()
    if a.phase in ('all','sanity'):
        for task in ('chat','needle'):
            args = [sys.executable,HERE/'sanity.py',task]+(['--tokens','200000'] if task=='needle' else [])
            command(args,a.work/('sanity-'+task+'.jsonl'),timeout=7200)
        result['completed_phases'].append('sanity')
        save()
    client = [a.client_python,HERE/'measure.py','--endpoint',endpoint+'/v1','--native-metrics-endpoint',endpoint,
              '--model',model,'--context-prompts',contexts]
    for phase in ('speed','long'):
        if a.phase in ('all',phase):
            command(client+['--phase',phase,'--out',a.work/('measure.json' if phase=='speed' else 'long.json')])
            result['completed_phases'].append(phase)
            save()
    if a.phase == 'probe' or (a.phase == 'all' and a.key == 'tv-std'):
        command([a.client_python,HERE/'accuracy.py','--endpoint',endpoint,'--model',model,'--out',a.work/'probe','--probe'])
        probe = json.loads((a.work/'probe/gsm8k.json').read_text())['summary']
        result['high_effort_probe'] = {**probe, 'truncation_rate': probe['truncated']/probe['n']}
        save()
        if probe['truncated']/probe['n'] >= .20:
            raise RuntimeError('default high effort truncates >=20% of the 20-problem probe; recommend a separate medium-effort probe before changing the common quality method')
    if a.phase in ('all','accuracy'):
        args = [a.client_python,HERE/'accuracy.py','--endpoint',endpoint,'--model',model,'--out',a.work/'accuracy']
        command(args+(['--screen'] if a.key=='tv-fp8' else []))
        result['completed_phases'].append('accuracy')
        save()
    if a.phase in ('all','probes'):
        for phase in (('retrieval','copy') if a.key == 'tv-std' else ('retrieval',)):
            command([a.client_python,HERE/'final-probes.py',phase,'--endpoint',endpoint,'--model',model,
                     '--fixtures',fixtures,'--out',a.work/(phase+'.json')])
            result['completed_phases'].append(phase)
            save()
    if a.phase == 'needle950' or a.needle950:
        if a.key != 'tv-fp8':
            raise ValueError('950k needle requires the FP8 tier')
        command([sys.executable,HERE/'sanity.py','needle','--tokens','950000'],a.work/'sanity-needle950.jsonl',timeout=7200)
        result['completed_phases'].append('needle950')
    result['status'] = 'measured'
except BaseException as exc:
    result.update(status='failed', error=type(exc).__name__+': '+str(exc))
finally:
    if container:
        log = subprocess.run(['docker','logs',container],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[redacted address]', log.stdout)
        (a.work/'engine.log').write_text(text)
        subprocess.run(['docker','stop','--time','60',container],capture_output=True)
        subprocess.run(['docker','rm',container],capture_output=True)
    stop.set()
    if container:
        watcher.join(timeout=10)
    result['memory_after'] = memory_fields()
    result['gpu_pids_after'] = gpu_pids()
    save()
print(json.dumps(result,indent=2),flush=True)
raise SystemExit(result['status'] != 'measured' or bool(result['gpu_pids_after']))
