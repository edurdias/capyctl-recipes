#!/usr/bin/env python3
"""Score published layouts against the unchanged saved teacher-forced reference."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM
from next_token_agree import teacher_force, summary

p = argparse.ArgumentParser()
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--answers', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise SystemExit('GPU occupied; no model loaded')
torch.set_num_threads(8)
torch.manual_seed(0)
report = {'candidate': str(a.candidate), 'reference': str(a.reference), 'answers': str(a.answers), 'method': 'unchanged next_token_agree.teacher_force, saved 200-record corpus and actual teacher-forced reference; eager BF16 graph'}
try:
    model, loading = AutoModelForCausalLM.from_pretrained(a.candidate, dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='eager', local_files_only=True, output_loading_info=True)
    report['loading_info'] = loading
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('checkpoint loading was not exact')
    model.to(dtype=torch.bfloat16).eval()
    answers = json.loads(a.answers.read_text())
    reference = json.loads(a.reference.read_text())
    with torch.inference_mode():
        scored, generated = teacher_force(model, answers['records'] if isinstance(answers, dict) else answers, reference['records'])
    report.update(status='scored', summary=summary(scored), records=scored, candidate_vs_generated_summary=summary(generated))
except Exception as exc:
    report.update(status='unsupported', error=type(exc).__name__+': '+str(exc))
a.out.write_text(json.dumps(report, indent=2, default=list)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'records'}, default=list), flush=True)
