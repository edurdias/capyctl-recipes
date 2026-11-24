#!/usr/bin/env python3
"""Summarize measured laptop gates without treating missing results as passes."""
import json, math, statistics
from pathlib import Path
B=Path(__file__).resolve().parent
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [mid-half,mid+half]
result={}
for label in ('incumbent','candidate-c','candidate-c-nextn2','candidate-c-incumbent-nextn2'):
    row={}
    for kind in ('memory','speed'):
        p=B/f'laptop-{label}-{kind}.json'
        if p.exists():row[kind]=json.loads(p.read_text())
    if label=='candidate-c':
        corrected=B/'laptop-candidate-c-corrected-speed.json'
        if corrected.exists():row['speed']=json.loads(corrected.read_text())
    row['quality']={}
    for task in ('gsm8k','tools','humaneval'):
        p=B/f'laptop-{label}-quality/{task}.json'
        if p.exists():row['quality'][task]=json.loads(p.read_text())['summary']
    if row.get('memory'):result[label]=row
inc=result.get('incumbent',{}).get('quality',{})
can=result.get('candidate-c',{}).get('quality',{})
gates={}
for task,n,field in [('gsm8k',250,'correct'),('tools',30,'pass'),('humaneval',164,'passed')]:
    if task not in inc or task not in can or inc[task]['n']!=n or can[task]['n']!=n:
        gates[task]={'pass':False,'reason':'full matched results unavailable'};continue
    i,c=inc[task],can[task];lower=wilson(i[field],n)[0];upper=wilson(i['truncated'],n)[1]
    gates[task]={'incumbent_accuracy_wilson95_lower':lower,'incumbent_truncation_wilson95_upper':upper,'candidate_accuracy':c[field]/n,'candidate_truncation':c['truncated']/n,'pass':c[field]/n>=lower and c['truncated']/n<=upper and c['errors']==i['errors']==0 and (task!='tools' or c[field]==30)}
result['gates']=gates
(B/'laptop-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('| Checkpoint | Startup / peak MiB | Margin MiB | c1 / c2 / c4 / c8 median tok/s | GSM8K | Tools | HumanEval |')
print('|---|---|---|---|---|---|---|')
for label,row in result.items():
    if label=='gates':continue
    m=row['memory'];s=row.get('speed',{});qs=row['quality'];speed=[]
    for lv in s.get('levels',[]):
        vals=[r.get('aggregate_tps') for r in lv['rounds'] if r.get('aggregate_tps') is not None]
        speed.append(f'{statistics.median(vals):.1f}' if vals else 'unavailable')
    scores=[str(qs[t].get(f))+'/'+str(qs[t]['n']) if t in qs else 'unavailable' for t,f in [('gsm8k','correct'),('tools','pass'),('humaneval','passed')]]
    print(f"| {label} | {m.get('startup_peak_mib','unavailable')} / {m['peak_mib']} | {m['margin_mib']} | {' / '.join(speed) or 'unavailable'} | {' | '.join(scores)} |")
print(json.dumps(gates,indent=2))
