#!/usr/bin/env python3
"""GPT-OSS BF16 reconstructed-weight agreement; Harmony, fixed Gemma corpus.

Source experts are reconstructed by the installed Transformers MXFP4 loader.
After measuring the teacher, replace only the audited candidate's attention
weights in that SAME BF16 graph. This isolates weight error; it does not qualify
Quark activation quantization or the native inference engine.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Mxfp4Config

HERE = Path(__file__).resolve().parent
GUARD_ROOT = HERE.parents[2] / 'vllm/gemma-4-26b-a4b-capyctl-nvfp4-gb10/convert'
sys.path.insert(0, str(GUARD_ROOT if GUARD_ROOT.exists() else HERE))
from memory_guard import MemoryGuard


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    p.write_text(json.dumps(data, indent=2) + '\n')


def score_record(record, picked):
    agree = [x == y for x, y in zip(record['answer_ids'], picked, strict=True)]
    return {'id': record['id'], 'category': record['category'], 'n': len(agree),
            'agree': sum(agree), 'first_disagreement': next((i for i, ok in enumerate(agree) if not ok), len(agree)),
            'picked_ids': picked, 'agreement_mask': agree}


def summary(records):
    positions = sum(r['n'] for r in records)
    agree = sum(r['agree'] for r in records)
    return {'positions': positions, 'agree': agree, 'agreement': agree / positions,
            'median_first_disagreement': statistics.median(r['first_disagreement'] for r in records),
            'by_category': {c: sum(r['agree'] for r in records if r['category'] == c) /
                            sum(r['n'] for r in records if r['category'] == c)
                            for c in sorted({r['category'] for r in records})}}


def teacher_force(model, records, reference=None):
    scored, generated = [], []
    for i, record in enumerate(records):
        ids = torch.tensor([record['prompt_ids'] + record['answer_ids'][:-1]], device='cuda')
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(record['answer_ids'])).logits[0].float()
        assert torch.isfinite(logits).all(), record['id']
        top = logits.topk(2, dim=-1)
        picked = top.indices[:, 0].tolist()
        generated.append(score_record(record, picked))
        target = record if reference is None else {**record, 'answer_ids': reference[i]['picked_ids']}
        row = score_record(target, picked)
        margins = (top.values[:, 0] - top.values[:, 1]).tolist()
        row['top1_top2_margins'] = margins
        row['disagreement_top1_top2_margins'] = [{'position': j, 'margin': margins[j], 'reference_id': target['answer_ids'][j], 'candidate_id': picked[j]} for j, ok in enumerate(row['agreement_mask']) if not ok]
        scored.append(row)
        del ids, logits, top
        print(f'teacher-forced {i+1}/{len(records)}', flush=True)
    return scored, generated


def main(memory):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    a = ap.parse_args()
    # Every launch requires an empty GPU. No signals are sent by this program.
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError('GPU occupied; stop: ' + processes)
    audit = json.loads(a.audit.read_text())
    assert audit['passed'] and audit['expert_tensors'] == 144 and audit['attention_tensors'] == 96
    for name in ['config.json', 'model.safetensors.index.json']:
        assert sha(a.candidate / name) == audit['files'][name]['sha256']
    prompts = json.loads(a.prompts.read_text())['prompts']
    assert len(prompts) == 200 and len({p['id'] for p in prompts}) == 200
    a.results.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(a.source, local_files_only=True)
    tokenizer.padding_side = 'left'
    model, loading = AutoModelForCausalLM.from_pretrained(a.source, quantization_config=Mxfp4Config(dequantize=True), dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='eager', local_files_only=True, output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    # Transformers keeps layernorm state FP32 by default; the protocol uses one
    # ordinary BF16 graph for both checkpoints, including that state.
    model.to(dtype=torch.bfloat16)
    model.eval()
    assert not model.config.tie_word_embeddings
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert not any(p.is_meta for p in model.parameters())
    coverage = {'parameter_tensors': len(list(model.parameters())), 'parameter_elements': sum(p.numel() for p in model.parameters()), 'parameter_dtypes': dict(Counter(str(p.dtype) for p in model.parameters())), 'expert_alpha': model.model.layers[0].mlp.experts.alpha, 'expert_limit': model.model.layers[0].mlp.experts.limit}
    generation = GenerationConfig.from_pretrained(a.source, local_files_only=True)
    generation.do_sample = False
    generation.temperature = generation.top_p = generation.top_k = None
    eos = generation.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    records = []
    with torch.inference_mode():
        for start in range(0, len(prompts), a.batch_size):
            batch = prompts[start:start+a.batch_size]
            texts = [tokenizer.apply_chat_template(p['messages'], tools=p.get('tools'), tokenize=False, add_generation_prompt=True, reasoning_effort='low') for p in batch]
            inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
            outputs = model.generate(**inputs, generation_config=generation, max_new_tokens=512, use_cache=True)
            for p, ids, mask, out in zip(batch, inputs['input_ids'], inputs['attention_mask'], outputs):
                answer = out[inputs['input_ids'].shape[1]:].tolist()
                end = next((i+1 for i, token in enumerate(answer) if token in eos), len(answer))
                answer = answer[:end]
                records.append({'id': p['id'], 'category': p['category'], 'prompt_ids': ids[mask.bool()].tolist(), 'answer_ids': answer, 'hit_cap': len(answer) == 512 and answer[-1] not in eos})
            save(a.results / 'bf16-answers.partial.json', {'records': records})
            print(f'generated {len(records)}/{len(prompts)}', flush=True)
            del inputs, outputs
        metadata = {'source_revision': '6cee5e81ee83917806bbde320786a8fb61efebee', 'prompt_sha256': sha(a.prompts), 'source_config_sha256': sha(a.source / 'config.json'), 'source_index_sha256': sha(a.source / 'model.safetensors.index.json'), 'tokenizer_sha256': sha(a.source / 'tokenizer.json'), 'chat_template_sha256': sha(a.source / 'chat_template.jinja'), 'audit_sha256': sha(a.audit), 'reasoning_effort': 'low', 'max_tokens': 512, 'generation_batch_size': a.batch_size, 'eos_ids': sorted(eos), 'coverage': coverage, 'software': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__}}
        save(a.results / 'bf16-answers.json', {**metadata, 'records': records})
        (a.results / 'bf16-answers.partial.json').unlink()
        teacher, _ = teacher_force(model, records)
        save(a.results / 'score-bf16.json', {**metadata, 'summary': summary(teacher), 'records': teacher})
        index = json.loads((a.candidate / 'model.safetensors.index.json').read_text())['weight_map']
        replaced = []
        for shard in sorted(set(index.values())):
            with safe_open(str(a.candidate / shard), framework='pt', device='cpu') as reader:
                for row in audit['tensors']:
                    name = row['name']
                    if row['kind'] != 'attention' or index[name] != shard:
                        continue
                    packed = reader.get_tensor(name)
                    scale = reader.get_tensor(name.removesuffix('.weight') + '.weight_scale')
                    parameter = model.get_parameter(name)
                    value = (packed.float() * scale[:, None]).to(torch.bfloat16)
                    assert value.shape == parameter.shape and torch.isfinite(value).all(), name
                    parameter.copy_(value)
                    replaced.append(name)
        assert len(replaced) == 96
        candidate, generated = teacher_force(model, records, teacher)
    report = {**metadata, 'method': 'BF16 reconstructed OpenAI teacher; same graph with FP8-attention weights dequantized to BF16; unchanged experts; no activation quantization', 'summary': summary(candidate), 'candidate_vs_generated': summary(generated), 'teacher_vs_generated': summary(teacher), 'replaced_attention_weights': replaced, 'truncated_records': [r['id'] for r in records if r['hit_cap']], 'completed_records': sum(not r['hit_cap'] for r in records), 'records': candidate, 'candidate_vs_generated_records': generated, 'wall_seconds': time.monotonic()-started, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'gate': {'minimum_agreement': 0.995, 'passed': summary(candidate)['agreement'] >= 0.995}}
    report['memory_guard'] = memory.metrics()
    save(a.results / 'score-fp8attn.json', report)
    print(json.dumps({k: v for k,v in report.items() if k not in ('records', 'candidate_vs_generated_records', 'replaced_attention_weights')}), flush=True)

if __name__ == '__main__':
    with MemoryGuard() as guard:
        main(guard)
