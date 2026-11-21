#!/usr/bin/env python3
"""Render every retained Inkling launch; never run inference or mutate evidence."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {
    'tv-std': 'Published community standard (baseline)',
    'tv-fp8': 'Published community max-FP8 (screen)',
    'sg-plain': 'Original plain, cold safety stop',
    'sg-plain-jit-warmup': 'Original plain, 0.45 JIT preparation',
    'sg-plain-warm': 'Original plain, initial transport failures',
    'sg-plain-transport': 'Original plain, two transport repairs only',
    'sg-plain-screen-completed': 'Original plain, composite screen',
    'sg-dspark-host-a': 'Original DSpark, screen',
    'sg-dspark-code-full': 'Original DSpark, full code only',
    'sg-fp8-dspark-3-tune': 'FP8 DSpark, unresolved 3-token option rejected',
    'sg-fp8-dspark-3-configured': 'FP8 DSpark, gamma 2 rejected by model',
    'sg-fp8-dspark-8-tune': 'FP8 DSpark gamma 7, tuning only',
    'sg-fp8-dspark-host-a': 'FP8 DSpark gamma 7 (elected speed)',
    'sg-fp8-plain-host-a': 'FP8 plain, failed screen',
    'sg-fp8-index-load-only': 'FP8 DSpark, repaired-index load only',
    'sg-fp8-dspark-full': 'FP8 DSpark, full quality continuation',
    'cold-start-elected-host-a': 'FP8 DSpark, elected cold-JIT start',
}


def rows(matrix):
    return matrix['rows']


def points(row, field='measure', label='short'):
    return {p['streams'] if label != 'long' else p['target_prompt_tokens']: p
            for p in (row.get(field) or {}).get('levels', []) if p['label'] == label}


def number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def gate(row):
    quality = row.get('quality') or {}
    if row['status'] == 'failed':
        return 'failed before ready'
    if quality.get('scope') == 'full-humaneval':
        return 'code only pass' if quality['humaneval']['passed'] else 'code only fail'
    if quality.get('scope') == 'full':
        return 'full pass' if quality.get('approved') else 'full fail'
    if quality.get('scope') == 'screen-50':
        return 'screen pass only' if quality.get('screen_passed') else 'screen fail'
    return 'not evaluated'


def markdown(matrix):
    lines = ['## Every retained launch', '',
             'Tokens/s are aggregate completion tokens / round wall span, median of three rounds. Quality cells show score and truncations; screens are not full qualification. Ready/minimum memory is absolute system MemAvailable, not an isolated allocation.', '',
             '| Setup (run) | Gate | 1 stream | 2 streams | 4 streams | 8 streams | Ready s | Ready / min GiB available | GSM8K | Tools | HumanEval |',
             '|---|---|---:|---:|---:|---:|---:|---|---|---|---|']
    for row in rows(matrix):
        p = points(row)
        scores = []
        for name, field in [('gsm8k', 'correct'), ('tools', 'pass'), ('humaneval', 'passed')]:
            score = row.get('accuracy', {}).get(name)
            scores.append(f'{score[field]}/{score["n"]} ({score["truncated"]} trunc' +
                          (f'; {score["errors"]} errors' if score.get('errors') else '') + ')' if score else '—')
        ready = (row.get('memory_ready') or {}).get('MemAvailable')
        minimum = row.get('minimum_memavailable_bytes')
        memory = ' / '.join(number(v / 2**30) if v is not None else '—' for v in (ready, minimum))
        cells = [f'{LABELS.get(row["run"], row["key"])} (`{row["run"]}`)', gate(row)]
        cells += [number(p.get(c, {}).get('aggregate_tps')) for c in (1, 2, 4, 8)]
        cells += [number(row.get('ready_s')), memory] + scores
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## TTFT, acceptance and exact context', '',
              'TTFT includes reasoning. Context decode excludes TTFT. Accepted length includes the target/bonus token; SGLang span lengths include prefill-emitted tokens and are not exact accepted-draft fractions. The tuning row\'s 32k point used 128 output tokens and is excluded from the 512-token ranking.', '',
              '| Setup (run) | Short TTFT 1/2/4/8, s | 24k TTFT 1/4, s | Accepted length 1/2/4/8 | Decode 8k/32k/64k/110k, tokens/s |',
              '|---|---|---|---|---|']
    for row in rows(matrix):
        short, ttft, long = points(row), points(row, label='24k'), points(row, 'long', 'long')
        if not short and not ttft and not long:
            continue
        cells = [f'{LABELS.get(row["run"], row["key"])} (`{row["run"]}`)',
                 ' / '.join(number(short.get(c, {}).get('ttft_s'), 3) for c in (1, 2, 4, 8)),
                 ' / '.join(number(ttft.get(c, {}).get('ttft_s'), 3) for c in (1, 4)),
                 ' / '.join(number((short.get(c, {}).get('acceptance') or {}).get('mean_accepted_length'), 3) for c in (1, 2, 4, 8)),
                 ' / '.join(number(long.get(c, {}).get('decode_tps')) for c in (8000, 32000, 64000, 110000))]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=HERE / 'bench/final-matrix.json')
    args = parser.parse_args()
    print(markdown(json.loads(args.matrix.read_text())))


if __name__ == '__main__':
    main()
