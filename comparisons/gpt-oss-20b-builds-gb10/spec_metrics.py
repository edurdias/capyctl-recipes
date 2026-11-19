#!/usr/bin/env python3
"""Label-free native speculative metrics, sampled outside timed workload spans."""
import math
import re
import time
import urllib.request

LINE = re.compile(r'^(vllm|llamacpp|sglang):([A-Za-z0-9_]+)(?:\{[^\n]*\})?\s+([^\s]+)(?:\s+[^\s]+)?$')


def snapshot(endpoint):
    result = {'sampled_at': time.time(), 'status': 'unmeasured', 'counters': {}, 'gauges': {}}
    try:
        with urllib.request.urlopen(endpoint.rstrip('/')+'/metrics', timeout=10) as response:
            text = response.read().decode()
        for line in text.splitlines():
            match = LINE.fullmatch(line)
            if not match:
                continue
            engine, name, raw = match.groups()
            if name.endswith('_created'):
                continue
            counter = (name.startswith('spec_decode_num_') or
                       name in ('spec_verify_calls_total', 'generation_tokens_total'))
            gauge = name in ('spec_accept_length', 'spec_accept_rate')
            if not (counter or gauge):
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid native metric')
            target = result['counters' if counter else 'gauges']
            target[engine+':'+name] = target.get(engine+':'+name, 0)+value
        if result['counters'] or result['gauges']:
            result['status'] = 'measured'
    except (OSError, UnicodeError, ValueError) as exc:
        result['error'] = type(exc).__name__
    return result


def span(before, after):
    result = {'before': before, 'after': after, 'deltas': {}, 'mean_accepted_length': None,
              'token_acceptance': None, 'status': 'unmeasured'}
    if before['status'] != 'measured' or after['status'] != 'measured':
        return result
    for key in before['counters'].keys() | after['counters'].keys():
        delta = after['counters'].get(key, 0)-before['counters'].get(key, 0)
        if delta < 0:
            result['error'] = 'native counter reset'
            return result
        result['deltas'][key] = delta
    for engine in ('vllm', 'llamacpp'):
        def count(name):
            return result['deltas'].get(engine+':'+name+'_total', result['deltas'].get(engine+':'+name))
        drafts, drafted, accepted = [count('spec_decode_num_'+name) for name in ('drafts', 'draft_tokens', 'accepted_tokens')]
        if drafts and drafted and accepted is not None:
            result.update(mean_accepted_length=1+accepted/drafts, token_acceptance=accepted/drafted,
                          method='native accepted draft tokens / verification calls + bonus token', status='measured')
            return result
    verifies = result['deltas'].get('sglang:spec_verify_calls_total')
    generated = result['deltas'].get('sglang:generation_tokens_total')
    if verifies and generated is not None:
        result.update(mean_accepted_length=generated/verifies, status='measured',
                      method='native completed generation tokens / verification calls; includes prefill-emitted tokens',
                      native_last_accept_length=after['gauges'].get('sglang:spec_accept_length'),
                      native_last_accept_rate=after['gauges'].get('sglang:spec_accept_rate'))
    return result
