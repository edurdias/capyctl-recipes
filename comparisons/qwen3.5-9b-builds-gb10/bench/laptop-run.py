#!/usr/bin/env python3
"""Laptop-only matched qualification; no environment installation or remote execution."""
import argparse, concurrent.futures, importlib.util, json, os, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
PY=Path.home()/'sglang-0.5.22-venv/bin/python'
BASE='http://localhost:30935'
def save(path,data): path.write_text(json.dumps(data,indent=2)+'\n')
def gpu():
    return list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')))
def post(route,body):
    with urllib.request.urlopen(urllib.request.Request(BASE+route,json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=3600) as r:return json.load(r)
def speed(label, long_only=False):
    spec=importlib.util.spec_from_file_location('cb',ROOT/'tools/capyctl-bench/capyctl_bench.py'); cb=importlib.util.module_from_spec(spec);sys.modules['cb']=cb;spec.loader.exec_module(cb)
    runner=cb.Runner(SimpleNamespace(model='qwen3.5-9b',temperature=0,extra_body='{"ignore_eos":true}',record_chunks=False),cb.Endpoint(BASE+'/v1',None,3600),None)
    result={'output_tokens':128,'rounds':3,'thinking':'default','levels':[],'contexts':[]}
    prompts=cb.load_prompts(ROOT/'tools/capyctl-bench/prompts.json')
    for c in (() if long_only else (1,2,4,8)):
        runner.round_(c,prompts,0,128)
        rounds=[runner.round_(c,prompts,0,128) for _ in range(3)]
        result['levels'].append({'concurrency':c,'rounds':rounds})
        save(HERE/'bench'/f'laptop-{label}-speed.json',result)
    for target in (8000,27000,29000,30000):
        # Exact token count measured by the same chat tokenizer used for quality.
        lo,hi=1,target
        while lo<hi:
            mid=(lo+hi+1)//2
            messages=[{'role':'user','content':' data'*mid+'\nSummarize the text.'}]
            count=post('/v1/tokenize',{'model':'qwen3.5-9b','messages':messages})['count']
            if count<=target:lo=mid
            else:hi=mid-1
        messages=[{'role':'user','content':' data'*lo+'\nSummarize the text.'}]
        rec=runner.one(messages,128)
        result['contexts'].append({'target':target,'request':rec})
        save(HERE/'bench'/f'laptop-{label}-speed.json',result)
def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('model');p.add_argument('--draft');p.add_argument('--draft-tokens',type=int,default=2);p.add_argument('--speed-only',action='store_true');p.add_argument('--long-only',action='store_true');a=p.parse_args()
    bench=HERE/'bench'; label=a.label; log=bench/f'laptop-{label}-runtime.txt'
    env=os.environ.copy();env.update(MAX_JOBS='2',PATH=str(PY.parent)+':'+env['PATH'],HF_HOME=str(Path.home()/'qwen35-laptop-hf-cache'),HF_HUB_DISABLE_XET='1')
    args=[str(PY),'-m','sglang.launch_server','--model-path',a.model,'--host','localhost','--port','30935','--served-model-name','qwen3.5-9b','--context-length','32768','--max-running-requests','8','--max-total-tokens','49152','--chunked-prefill-size','2048','--cuda-graph-max-bs-decode','8','--mamba-ssm-dtype','bfloat16','--max-mamba-cache-size','48','--reasoning-parser','qwen3','--tool-call-parser','qwen3_coder']
    if a.draft:args+=['--speculative-algorithm','NEXTN','--speculative-draft-model-path',a.draft,'--speculative-num-steps','1','--speculative-eagle-topk','1','--speculative-num-draft-tokens',str(a.draft_tokens)]
    samples=[]; stop=threading.Event(); record={'args':args,'baseline_memory_mib':gpu(),'sampling_interval_seconds':0.5}
    def monitor():
        while not stop.is_set():
            samples.append([time.time(),*gpu()]);stop.wait(.5)
    thread=threading.Thread(target=monitor);thread.start()
    proc=None
    try:
        with log.open('w') as f:
            proc=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.monotonic()
            while time.monotonic()-start<1800:
                if proc.poll() is not None:raise RuntimeError('engine exited before ready')
                try:
                    with urllib.request.urlopen(BASE+'/health',timeout=2) as r:
                        if r.status==200:break
                except OSError:pass
                time.sleep(2)
            else:raise RuntimeError('startup timeout')
            record.update(ready_seconds=time.monotonic()-start,ready_memory_mib=gpu(),startup_peak_mib=max(x[1] for x in samples))
            save(bench/f'laptop-{label}-memory.json',record)
            speed(label,a.long_only)
            if not a.draft and not a.speed_only:
                with (bench/f'laptop-{label}-quality.txt').open('w') as qlog:
                    subprocess.run([str(PY),str(HERE/'quality.py'),'--endpoint',BASE,'--model','qwen3.5-9b','--out',str(bench/f'laptop-{label}-quality')],env=env,stdout=qlog,stderr=subprocess.STDOUT,check=True)
    except Exception as exc:
        record['failure']=str(exc)
        raise
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        stop.set();thread.join()
        record.update(peak_mib=max((x[1] for x in samples),default=0),samples=samples,final_memory_mib=gpu())
        record['margin_mib']=record['baseline_memory_mib'][1]-record['peak_mib']
        save(bench/f'laptop-{label}-memory.json',record)
if __name__=='__main__':main()
