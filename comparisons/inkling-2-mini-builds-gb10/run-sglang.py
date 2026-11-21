#!/usr/bin/env python3
"""Run a pinned cookbook row using the unchanged comparison clients."""
import argparse
import json
import os
from pathlib import Path
import re
from runpy import run_path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
helpers = run_path(str(HERE/'run-host.py'))
gpu_pids, memory_fields = helpers['gpu_pids'], helpers['memory_fields']
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('key', choices=['sg-plain', 'sg-dspark', 'sg-fp8-plain', 'sg-fp8-dspark'])
p.add_argument('--work', type=Path, required=True)
p.add_argument('--mem-fraction-static', required=True, type=float)
p.add_argument('--full', action='store_true')
p.add_argument('--warmup-only', action='store_true', help='Compile cookbook graphs, then stop without measurements')
p.add_argument('--humaneval-ids', nargs='+', help='Repair only named transport-failed HumanEval requests')
p.add_argument('--humaneval-only', action='store_true', help='Run the full code gate only; retain prior screen observations')
p.add_argument('--draft-tokens', type=int, choices=[8], default=8, help='Pinned Inkling MTP supports only gamma 7 / block 8')
p.add_argument('--tune-only', action='store_true', help='Measure only c1/c8 and exact 32k decode')
p.add_argument('--quality-only', action='store_true', help='Reuse completed speed measurements without repeating them')
p.add_argument('--context-prompts', type=Path, help='Reuse exact seeded prompts across tuning settings')
p.add_argument('--resume-screen', type=Path, help='Extend this exact engine setting without rerunning screen cases')
a = p.parse_args()
checkpoint = Path.home()/'models'/('Inkling-2-Mini-capyctl-fp8attn' if 'fp8' in a.key else 'Inkling-2-Mini-ed40e7da')
revision = 'ed40e7da625f594fc9c4990f5f66be24873c22d7'
if gpu_pids():
    raise SystemExit('occupied GPU; no launch')
original = Path.home()/'models/Inkling-2-Mini-ed40e7da'
if (original/'REVISION').read_text().strip() != revision:
    raise SystemExit('checkpoint revision mismatch')
receipt = json.loads((original/'download-verification.json').read_text())
if 'verified' in receipt:
    verified = receipt['verified'] and receipt['revision'] == revision
else:
    # Host A retains the original download receipt; the owner verified the
    # later revision changes only config/card, not either weight payload.
    identities = {row['file']: row for row in receipt['files']}
    expected = {'model-00001-of-00001.safetensors': '65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1',
                'mtp.safetensors': '7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2'}
    verified = all(identities[name]['sha256'] == digest == identities[name]['lfs_sha256']
                   for name, digest in expected.items())
if not verified:
    raise SystemExit('checkpoint verification missing')
a.work.mkdir(parents=True, exist_ok=False)
endpoint, model = 'http://127.0.0.1:18120', 'inkling-2-mini'
argv = [str(Path(sys.executable).parent/'sglang'), 'serve', '--model-path', str(checkpoint),
        '--served-model-name', model, '--tp', '1', '--ep', '1', '--attention-backend', 'fa4',
        '--dtype', 'bfloat16', '--page-size', '128', '--mem-fraction-static', str(a.mem_fraction_static),
        '--grammar-backend', 'xgrammar', '--enable-multimodal', '--reasoning-parser', 'inkling',
        '--tool-call-parser', 'inkling2', '--host', '127.0.0.1', '--port', '18120', '--enable-metrics']
opt = {}
if a.key.endswith('dspark'):
    argv += ['--speculative-algorithm', 'DSPARK', '--speculative-num-draft-tokens', str(a.draft_tokens)]
    opt = {name: '1' for name in ('SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL',
           'SGLANG_OPT_DSV41_DSPARK_MID_FOLD', 'SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE',
           'SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS')}
result = {'key': a.key, 'checkpoint': ('capyctl-fp8-attention:' if 'fp8' in a.key else 'thinkingmachines/Inkling-2-Mini@')+revision,
          'argv': argv, 'environment': {'MAX_JOBS': '2', **opt}, 'status': 'started',
          'memory_before': memory_fields(), 'quality_scope': 'full' if a.full else 'screen-50',
          'completed_phases': []}
if a.humaneval_only:
    result['quality_scope'] = 'full-humaneval'
if a.key.endswith('dspark'):
    result['dspark_configuration'] = {'gamma': a.draft_tokens - 1,
                                     'tokens_per_block': a.draft_tokens}
if a.resume_screen:
    screen_run = json.loads((a.resume_screen/'run.json').read_text())
    if screen_run['argv'] != argv or screen_run['environment'] != result['environment']:
        raise SystemExit('screen continuation requires identical engine arguments and optimization variables')
    result['screen_continuation'] = {'screen_launch_run': a.resume_screen.name,
                                   'full_launch_run': a.work.name,
                                   'identical_engine_configuration': True,
                                   'per_observation_provenance_field': 'launch_run'}
stop, killed = threading.Event(), threading.Event()
engine = None
watcher = None

def save():
    (a.work/'run.json').write_text(json.dumps(result, indent=2)+'\n')

def stop_engine():
    if engine is not None and engine.poll() is None:
        os.killpg(engine.pid, signal.SIGTERM)

def watch():
    with (a.work/'memory.jsonl').open('w') as f:
        while not stop.is_set():
            fields = memory_fields()
            f.write(json.dumps({'unix_s': time.time(), **fields})+'\n')
            f.flush()
            if fields['MemAvailable'] < 3*1024**3:
                killed.set()
                stop_engine()
                return
            stop.wait(2)

def command(args, name):
    with (a.work/(name+'.log')).open('w') as log:
        subprocess.run([str(x) for x in args], stdout=log, stderr=subprocess.STDOUT, check=True)
    if killed.is_set():
        raise RuntimeError('owned engine stopped: MemAvailable below 3 GiB')

try:
    save()
    contexts = a.context_prompts or a.work/'context-prompts.json'
    if not a.context_prompts and not (a.warmup_only or a.humaneval_ids or a.quality_only):
        command([sys.executable, HERE/'make-context.py', '--checkpoint', checkpoint, '--out', contexts], 'prepare-context')
    if a.full and not a.warmup_only:
        fixtures = a.work/'fixtures.json'
        command([sys.executable, HERE/'final-probes.py', 'prepare', '--checkpoint', checkpoint, '--out', fixtures], 'prepare-fixtures')
    if not (a.warmup_only or a.tune_only):
        command([sys.executable, '-c', "from datasets import load_dataset; assert len(load_dataset('openai/openai_humaneval',revision='7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',split='test'))==164"], 'prepare-data')
    if gpu_pids():
        raise RuntimeError('GPU became occupied; no launch')
    started = time.monotonic()
    with (a.work/'engine.log').open('w') as log:
        engine = subprocess.Popen(argv, env={**os.environ, **opt}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result['engine_pid'] = engine.pid
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    while True:
        if engine.poll() is not None:
            raise RuntimeError('engine exited before readiness; see engine.log')
        try:
            with urllib.request.urlopen(endpoint+'/health', timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        time.sleep(2)
    result.update(ready_s=time.monotonic()-started, memory_ready=memory_fields())
    save()
    for phase in (() if a.warmup_only or a.humaneval_ids or a.quality_only else (('tune',) if a.tune_only else ('speed', 'long'))):
        command([sys.executable, HERE/'measure.py', '--endpoint', endpoint+'/v1', '--native-metrics-endpoint', endpoint,
                 '--model', model, '--context-prompts', contexts, '--phase', phase,
                 '--out', a.work/('long.json' if phase == 'long' else 'measure.json')], phase)
        result['completed_phases'].append(phase)
        save()
    if a.full and not a.warmup_only:
        command([sys.executable, HERE/'accuracy.py', '--endpoint', endpoint, '--model', model,
                 '--out', a.work/'probe', '--probe']+
                (['--resume-from', a.resume_screen/'accuracy'] if a.resume_screen else []), 'probe')
        probe = json.loads((a.work/'probe/gsm8k.json').read_text())['summary']
        result['high_effort_probe'] = {**probe, 'truncation_rate': probe['truncated']/probe['n']}
        save()
        if probe['truncated']/probe['n'] >= .20:
            raise RuntimeError('default high effort truncates >=20% of the 20-problem probe; pause full quality without changing effort')
    if not a.warmup_only and not a.tune_only:
        quality = [sys.executable, HERE/'accuracy.py', '--endpoint', endpoint, '--model', model, '--out', a.work/'accuracy']
        if a.humaneval_ids:
            quality += ['--humaneval-ids', *a.humaneval_ids]
        if a.humaneval_only:
            quality += ['--humaneval-only']
        if a.resume_screen:
            quality += ['--resume-from', a.resume_screen/'accuracy']
        command(quality+([] if a.full or a.humaneval_only else ['--screen']), 'accuracy')
        result['completed_phases'].append('accuracy')
        save()
    if a.full and not a.warmup_only:
        for phase in ('retrieval', 'copy'):
            command([sys.executable, HERE/'final-probes.py', phase, '--endpoint', endpoint, '--model', model,
                     '--fixtures', fixtures, '--out', a.work/(phase+'.json')], phase)
            result['completed_phases'].append(phase)
            save()
    result['status'] = 'warmed' if a.warmup_only else 'measured'
except BaseException as exc:
    result.update(status='failed', error=type(exc).__name__+': '+str(exc))
finally:
    stop_engine()
    if engine:
        try:
            engine.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(engine.pid, signal.SIGKILL)
            engine.wait()
    stop.set()
    if watcher:
        watcher.join(timeout=10)
    if (a.work/'engine.log').exists():
        text = (a.work/'engine.log').read_text()
        (a.work/'engine.log').write_text(re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[redacted address]', text))
    result.update(memory_after=memory_fields(), gpu_pids_after=gpu_pids())
    save()
print(json.dumps(result), flush=True)
raise SystemExit(result['status'] not in ('measured', 'warmed') or bool(result['gpu_pids_after']))
