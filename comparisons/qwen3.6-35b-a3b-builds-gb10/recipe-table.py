#!/usr/bin/env python3
"""Render retained Qwen3.6 evidence; never run inference or modify evidence."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {
    'cd': 'CapyCTL DeltaNet · vLLM DFlash2 (elected)',
    'cc': 'CapyCTL combined · vLLM DFlash2',
    'cd0': 'CapyCTL DeltaNet · vLLM, no drafter (isolation)',
    'n1': 'NVIDIA published GB10 recipe · vLLM MTP4 (baseline)',
    'e1': 'NVIDIA · SGLang, no drafter (old capped screen)',
    'v0': 'NVIDIA · CapyCTL vLLM, no drafter',
    'v2': 'NVIDIA · CapyCTL vLLM MTP2',
    'v3': 'NVIDIA · CapyCTL vLLM MTP3',
    'v4': 'NVIDIA · CapyCTL vLLM MTP4',
    'vd': 'NVIDIA · CapyCTL vLLM DFlash2',
    'vs': 'NVIDIA · CapyCTL vLLM 0.33 DSpark8',
    'pr': 'RedHatAI · vLLM DFlash2',
    'pu': 'Unsloth standard · vLLM DFlash2',
    'puf': 'Unsloth Fast · vLLM DFlash2',
    'pq': 'Qwen official FP8 · vLLM DFlash2',
    'pi': 'Intel mixed · vLLM DFlash2',
    'pv1': 'Published vLLM-site recipe · NVIDIA MTP3',
    'ps': 'Published SGLang cookbook · NVIDIA, no drafter',
    'pg': 'Published MXFP4_MOE GGUF · llama.cpp v0.7.0',
    'pgm': 'Published MTP GGUF · llama.cpp b12242',
}


def rows(matrix):
    return matrix['engines']['rows']


def points(row, field='levels', label='short'):
    return {p['streams'] if field == 'levels' else p['target_prompt_tokens']: p
            for p in row[field] if p['label'] == label}


def number(value, digits=1):
    return '—' if value is None else f'{value:.{digits}f}'


def gate(row):
    q = row.get('quality') or {}
    if row.get('quality_cap') != 32768:
        return 'old capped screen' if row.get('quality_cap') else 'not evaluated'
    if row.get('quality_scope') == 'screen-50' or q.get('scope') == 'screen':
        return 'screen pass only' if q.get('screen_passed') else 'screen fail'
    return 'full pass' if q.get('approved') else 'full fail'


def markdown(matrix):
    lines = ['## Every measured setup', '',
             'Aggregate completion tokens / round wall time, median of three rounds. All columns use aggregate throughput, including one stream. Screens are not qualification. Ready memory is the system-used-memory delta, not total footprint.', '',
             '| Setup (run) | Gate | 1 stream | 2 streams | 4 streams | 8 streams | Ready s | Ready Δ GiB | GSM8K | Tools | HumanEval |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows(matrix):
        p = points(r)
        scores = []
        for name, field in [('gsm8k', 'correct'), ('tools', 'pass'), ('humaneval', 'passed')]:
            t = r['tasks'].get(name)
            scores.append(f'{t[field]}/{t["n"]} ({t["truncated"]} trunc)' if t else '—')
        lines.append('| ' + ' | '.join([f'{LABELS.get(r["run"], r["key"])} (`{r["run"]}`)', gate(r)] +
            [number(p.get(c, {}).get('aggregate_tps')) for c in (1, 2, 4, 8)] +
            [number(r['ready_s']), number(r['ready_memory_delta_bytes'] / 2**30)] + scores) + ' |')
    lines += ['', '## TTFT, draft acceptance and context', '',
              'TTFT includes reasoning tokens. Accepted length is accepted draft tokens per verification call plus its bonus token; it is not a probability. Context columns are post-first-token decode at exact native input lengths, not aggregate throughput.', '',
              '| Setup (run) | Short TTFT 1/2/4/8, s | 24k TTFT 1/4, s | Accepted length 1/2/4/8 | Decode 8k/32k/64k/110k, tokens/s |',
              '|---|---|---|---|---|']
    for r in rows(matrix):
        p, t, long = points(r), points(r, label='24k'), points(r, 'long', 'long')
        cells = [f'{LABELS.get(r["run"], r["key"])} (`{r["run"]}`)',
                 ' / '.join(number(p.get(c, {}).get('ttft_s'), 3) for c in (1, 2, 4, 8)),
                 ' / '.join(number(t.get(c, {}).get('ttft_s'), 3) for c in (1, 4)),
                 ' / '.join(number((p.get(c, {}).get('acceptance') or {}).get('mean_accepted_length'), 3) for c in (1, 2, 4, 8)),
                 ' / '.join(number(long.get(c, {}).get('decode_tps')) for c in (8000, 32000, 64000, 110000))]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix', type=Path, default=HERE / 'bench/final-matrix.json')
    args = ap.parse_args()
    print(markdown(json.loads(args.matrix.read_text())))


if __name__ == '__main__':
    main()
