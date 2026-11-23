import json,os,signal,subprocess,time
from pathlib import Path
home=Path.home(); root=home/'gptoss120-work'; source=home/'models/gpt-oss-120b'; output=home/'models/gpt-oss-120b-capyctl-fp8attn'
py=home/'vllm-0.32.0-lmcache-venv/bin/python'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
def run(args):
    process=subprocess.Popen([str(py),*map(str,args)],env=env,start_new_session=True)
    while process.poll() is None:
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0]) < 3*1024*1024:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(); raise RuntimeError('Stopped owned CPU job: MemAvailable below 3 GiB')
        time.sleep(2)
    if process.returncode: raise RuntimeError('CPU stage failed: '+str(process.returncode))
if not (source/'download-verification.json').exists(): raise RuntimeError('Verified download prerequisite missing')
run([root/'convert/mixed_attention.py',source,output,'--scale','channel'])
site=subprocess.check_output([str(py),'-c','import importlib.util;print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))'],text=True).strip()
run([root/'convert/audit_mixed.py',source,output,'--vllm-root',site,'--out',root/'checkpoint-audit.json'])
print('CPU CONVERSION AND AUDIT COMPLETE',flush=True)
