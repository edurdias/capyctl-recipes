#!/usr/bin/env python3
"""Publish text-free evidence tables without promoting a failing checkpoint."""
import json
from pathlib import Path
import statistics
import sys

H = Path.home()
ROOT = H/'clef-build'
HERE = H/'clef-reference/comparisons/clef-flash-builds-gb10'
sys.path.insert(0, str(HERE))
import decision_eval as harness


def main():
    output = HERE/'results'
    spec = json.loads((ROOT/'candidate-spec.json').read_text())
    reference = json.loads((output/'gb10-fp8-vs-bf16-heldout.json').read_text())['overall']
    text = ['# Clef checkpoint experiment', '', 'Agreement is against BF16, not gold-label correctness. Wilson intervals are descriptive question-level intervals; questions within requests can be correlated.', '', 'Sensitivity and layout-selection numbers are BF16-arithmetic simulations with explicit activation QDQ, not quantized-kernel measurements. Final agreement and timing below use the unmodified stock loader. Hooks were confined to an experiment launcher outside the serving environment.', '', 'Calibration: 18 independent deterministic synthetic decision requests; exact disjointness from the held-out set checked. No held-out row was used for calibration, scale tuning or layout selection. Diagnostic selection uses the recovered 41-request / 104-question cohort.', '', 'ModelOpt 0.47.0 supplies calibration and weight rounding. Own compressed-tensors export is authorized because the approved ModelOpt FP8 loader supports per-tensor, not per-channel weights. The published per-channel FP8 layout was round-tripped before new tensors were built. Static versus dynamic activations were selected by diagnostic simulation. Both joint and lexical heads stay byte-identical BF16.', '', '| Build | Package GB | Heldout same top | Agreement, 95% Wilson | Mean KL | Mean TV | Low-margin flips | Gate |', '|---|---:|---:|---|---:|---:|---:|---|']
    summaries = {}
    for candidate in spec['candidates']:
        tag = candidate['tag']
        audit = json.loads((ROOT/f'{tag}-audit.json').read_text())
        metrics = json.loads((output/f'{tag}-vs-bf16-heldout.json').read_text())
        gate = metrics['agreement'] >= .99 and metrics['wilson95'][0] >= .98 and metrics['mean_kl'] <= reference['mean_kl'] and metrics['mean_tv'] <= reference['mean_tv'] and metrics['request_errors'] == 0
        text.append(f"| {tag} | {audit['package_bytes']/1e9:.6f} | {metrics['same_top']}/{metrics['questions']} | {metrics['agreement']*100:.3f}% ({metrics['wilson95'][0]*100:.3f}–{metrics['wilson95'][1]*100:.3f}%) | {metrics['mean_kl']:.8g} | {metrics['mean_tv']:.8g} | {metrics['low_margin_flips']}/{metrics['low_margin_questions']} | {'PASS' if gate else 'FAIL'} |")
        groups = {}
        for rnd in range(1,4):
            for point in json.loads((output/f'{tag}-latency-round{rnd}.json').read_text()):
                groups.setdefault((point['questions'],point['target_total_tokens']),[]).extend(point['requests'])
        latency = []
        for (qn,target), rows in sorted(groups.items()):
            assert len(rows) == 300 and not any(r['error'] for r in rows)
            tokens = [r['input_tokens'] for r in rows]
            latency.append({'questions':qn,'target':target,'token_median':statistics.median(tokens),'token_min':min(tokens),'token_max':max(tokens),'p50_ms':harness.bench.percentile([r['latency_s']*1000 for r in rows],50),'p95_ms':harness.bench.percentile([r['latency_s']*1000 for r in rows],95),'errors':0})
        levels = {}
        for rnd in range(1,4):
            for level in json.loads((output/f'{tag}-throughput-round{rnd}.json').read_text()):
                levels.setdefault(level['concurrency'],[]).append(level['summary'])
        throughput = [{'concurrency':c,'requests_per_s':statistics.median(r['requests_per_s'] for r in rs),'questions_per_s':statistics.median(r['questions_per_s'] for r in rs),'errors':sum(r['errors'] for r in rs)} for c,rs in levels.items()]
        summaries[tag] = {'audit':audit,'heldout':metrics,'gate_pass':gate,'latency':latency,'throughput':throughput,'run':json.loads((output/f'{tag}-run.json').read_text())}
        (output/f'{tag}-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    text += ['', '## Standard latency points', '', 'Three rounds; five warmups and 100 timed whole-request measurements per point per round. Inputs are approximate, not token-identical. The 3/5-question short rows are schema floors, not 200-token claims.', '', '| Build | Questions | Nominal input | Actual token median (range) | p50 ms | p95 ms |', '|---|---:|---:|---|---:|---:|']
    for tag,s in summaries.items():
        for p in s['latency']:
            text.append(f"| {tag} | {p['questions']} | {p['target']} | {p['token_median']} ({p['token_min']}–{p['token_max']}) | {p['p50_ms']:.2f} | {p['p95_ms']:.2f} |")
    text += ['', '## Throughput', '', 'Three questions near 2k input; 200 timed requests at each client-concurrency level in each of three rounds. One running request: higher client concurrency measures queueing, not parallel engine batching.', '', '| Build | Client concurrency | Requests/s | Questions/s | Errors |', '|---|---:|---:|---:|---:|']
    for tag,s in summaries.items():
        for p in s['throughput']: text.append(f"| {tag} | {p['concurrency']} | {p['requests_per_s']:.3f} | {p['questions_per_s']:.3f} | {p['errors']} |")
    text += ['', 'All final engines use SGLang 0.5.22, 16,384 context/token pool, one running request, FP8 E4M3 KV, BF16 Mamba state, and static memory fraction 0.50. NVML device-memory queries are unsupported: GPU peak is unmeasured; system MemAvailable is separately sampled. Ada 16-GB residency is unmeasured; checkpoint bytes alone are not a residency qualification. The optional BF16-restoration checkpoint was not exported without that qualification.', '', 'All per-suite/length slices, flips and distributions are retained in the comparison JSON/JSONL files. Low margin means BF16 top-two probability margin <=0.05. KL uses natural logarithms and candidate floor 1e-9.', '']
    units = json.loads((ROOT/'units.json').read_text())
    sensitivity = {'method':'BF16-arithmetic simulations only; one quantized family per layer block; numerical QDQ does not promise stock-kernel equivalence','units':{}}
    text += ['## Per-unit sensitivity (BF16-arithmetic simulations)', '', '| Unit | FP8 dynamic agreement / KL | FP8 static agreement / KL | NVFP4 agreement / KL |', '|---|---|---|---|']
    for unit in units:
        row = {}
        cells = []
        for key,name in [('fp8_dynamic',f'fp8-{unit}-dynamic'),('fp8_static',f'fp8-{unit}-static'),('nvfp4',f'nvfp4-{unit}-nvfp4')]:
            path=ROOT/f'{name}-comparison.json'
            result=json.loads(path.read_text()) if path.exists() else None
            row[key]=result
            cells.append(f"{result['agreement']*100:.3f}% / {result['mean_kl']:.8g}" if result else 'Not run: FP8-sensitive')
        sensitivity['units'][unit]=row
        text.append('| '+unit+' | '+' | '.join(cells)+' |')
    (output/'capyctl-sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    sensitive_fp8 = [u for u,r in sensitivity['units'].items() if r['fp8_dynamic']['flips']]
    sensitive_nvfp4 = [u for u,r in sensitivity['units'].items() if r['nvfp4'] and r['nvfp4']['flips']]
    text += ['', f"FP8 dynamic sensitive units by top flips: {sensitive_fp8}. Individual zero-flip results do not rule out cumulative interactions.", f"NVFP4 sensitive units by top flips: {sensitive_nvfp4}.", 'The published NVFP4 checkpoint already shares both global scales in every one of its 11 quantized DeltaNet qkv/z pairs. This audit does not identify a published fused-scale bug. New fused NVFP4 exports independently assert shared input and weight global scales.', '']
    for filename in ('calibration-manifest.json','roundtrip.json','fp8-tuning.json','layout-selection.json','candidate-spec.json','published-fused-scale-audit.json'):
        shutil_copy(ROOT/filename,output/('capyctl-'+filename))
    assert json.loads((output/'capyctl-gpu-release.json').read_text())['compute_inventory_empty']
    text += ['', 'Owned engines stopped after measurements.', '']
    (output/'capyctl-build-report.txt').write_text('\n'.join(text))
    (output/'capyctl-build-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('\n'.join(text[:18]),flush=True)


def shutil_copy(source,destination):
    import shutil
    shutil.copy2(source,destination)


if __name__ == '__main__':
    main()
