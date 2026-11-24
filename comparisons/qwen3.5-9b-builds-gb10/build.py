#!/usr/bin/env python3
"""Independent GPTQ W4A16 builds; preserves every excluded source tensor, including MTP."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import struct
from pathlib import Path

SOURCE_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
TARGET = re.compile(r'model\.language_model\.layers\.\d+\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\.weight$')

def targeted(name):
    return name == 'lm_head.weight' or bool(TARGET.fullmatch(name))

def inventory(directory):
    result = {}
    for path in sorted(directory.glob('*.safetensors')):
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(length))
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                assert name not in result, name
                start, end = info['data_offsets']
                stream.seek(8 + length + start)
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    block = stream.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise EOFError(name)
                    digest.update(block)
                    remaining -= len(block)
                result[name] = {'file': path.name, 'dtype': info['dtype'], 'shape': info['shape'], 'bytes': end - start, 'sha256': digest.hexdigest()}
    return result

def audit(source, output, group, head_bits=4):
    import torch
    from safetensors import safe_open
    src, dst = inventory(source), inventory(output)
    retained = [n for n in src if not targeted(n)]
    for name in retained:
        assert name in dst, f'missing retained tensor: {name}'
        assert all(src[name][key] == dst[name][key] for key in ('dtype', 'shape', 'sha256')), name
    mtp = [n for n in src if n.startswith('mtp.') or '.mtp.' in n]
    assert mtp and all(src[n]['dtype'] == 'BF16' for n in mtp)
    quantized = [n for n in src if targeted(n)]
    assert 'lm_head.weight' in quantized
    for name in quantized:
        prefix = name.removesuffix('.weight')
        assert name not in dst, f'uncompressed target: {name}'
        assert prefix + '.weight_packed' in dst, name
        assert prefix + '.weight_scale' in dst, name
        bits = head_bits if name == 'lm_head.weight' else 4
        assert math.prod(src[name]['shape']) * bits // 8 == dst[prefix + '.weight_packed']['bytes'], name
        scales = 1 if name == 'lm_head.weight' and head_bits == 8 else src[name]['shape'][1] // group
        assert dst[prefix + '.weight_scale']['shape'] == [src[name]['shape'][0], scales], name
    expected_names = set(retained) | {name.removesuffix('.weight') + suffix for name in quantized for suffix in ('.weight_packed', '.weight_scale', '.weight_shape')}
    assert set(dst) == expected_names, {'unexpected_names': sorted(set(dst) - expected_names), 'missing_names': sorted(expected_names - set(dst))}
    for path in output.glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as reader:
            for name in reader.keys():
                if name.endswith('weight_scale'):
                    value = reader.get_tensor(name)
                    assert torch.isfinite(value).all() and (value > 0).all(), name
    config = json.loads((output / 'config.json').read_text())['quantization_config']
    assert config['quant_method'] == 'compressed-tensors'
    assert any('lm_head' in scheme['targets'] for scheme in config['config_groups'].values())
    head = next(scheme for scheme in config['config_groups'].values() if 'lm_head' in scheme['targets'])
    assert head['weights']['num_bits'] == head_bits and head['weights']['symmetric']
    assert head['weights']['strategy'] == ('channel' if head_bits == 8 else 'group')
    assert head.get('input_activations') is None
    # Target-only decode weight-read model: decoder tensors plus head, excluding
    # embeddings, vision and MTP. This is not measured DRAM traffic or speed.
    decode = sum(info['bytes'] for name, info in dst.items() if name.startswith('model.language_model.layers.') or name.startswith('lm_head.'))
    report = {'source_revision': SOURCE_REVISION, 'group_size': group, 'source_inventory': src, 'output_inventory': dst, 'retained_identity_count': len(retained), 'mtp_tensors': mtp, 'quantized_source_tensors': quantized, 'estimated_weight_bytes_per_generated_token': decode, 'traffic_label': 'architecture-based estimate, excludes caches/state/activations/workspace', 'checkpoint_weight_bytes': sum(p.stat().st_size for p in output.glob('*.safetensors'))}
    report['head_bits'] = head_bits
    report['head_strategy'] = head['weights']['strategy']
    (output / 'tensor-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group-size', type=int, choices=(32, 128), required=True)
    p.add_argument('--head-bits', type=int, choices=(4, 8), default=4)
    p.add_argument('--source', type=Path, default=Path.home() / 'models' / ('qwen35-source-' + SOURCE_REVISION[:12]))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--gpu-release', required=True, choices=['GPU owner GPU free'])
    a = p.parse_args()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from safetensors import safe_open
    from safetensors.torch import save_file
    assert torch.cuda.is_available(), 'GPU calibration requires released GPU'
    assert not a.output.exists(), 'build into a new directory'
    tokenizer = AutoTokenizer.from_pretrained(a.source, local_files_only=True)
    records = [json.loads(line) for line in a.calibration.read_text().splitlines() if line.strip()]
    assert len(records) >= 128
    samples = []
    for row in records:
        text = tokenizer.apply_chat_template(row['messages'], tools=row.get('tools'), tokenize=False, add_generation_prompt=False)
        samples.append(tokenizer(text, truncation=True, max_length=2048, add_special_tokens=False))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.source, dtype=torch.bfloat16, device_map='auto', local_files_only=True)
    selected = [name for name, module in model.named_modules() if targeted(name + '.weight')]
    assert 'lm_head' in selected and len(selected) == 201, (len(selected), selected)
    groups = {'int4': {'targets': [n for n in selected if n != 'lm_head'], 'weights': {'num_bits': 4, 'type': 'int', 'symmetric': True, 'strategy': 'group', 'group_size': a.group_size}, 'input_activations': None, 'output_activations': None}}
    groups['head'] = {'targets': ['lm_head'], 'weights': {'num_bits': a.head_bits, 'type': 'int', 'symmetric': True, 'strategy': 'channel' if a.head_bits == 8 else 'group', 'group_size': None if a.head_bits == 8 else a.group_size}, 'input_activations': None, 'output_activations': None}
    recipe = GPTQModifier(config_groups=groups, actorder=None, dampening_frac=0.01, offload_hessians=True)
    oneshot(model=model, processor=tokenizer, dataset=Dataset.from_list(samples), recipe=recipe, max_seq_length=2048, num_calibration_samples=len(samples))
    # Persist calibrated, packed state before the Transformers serializer. This
    # recovery artifact can be re-sharded without ever repeating calibration.
    from compressed_tensors import ModelCompressor
    from compressed_tensors.offload import disable_onloading
    compressor = ModelCompressor.from_pretrained_model(model)
    compressor.compress_model(model, skip_compressed=True)
    recovery = a.output.with_name(a.output.name + '-calibrated-state')
    recovery.mkdir()
    with disable_onloading():
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    assert all(not value.is_meta for value in state.values())
    torch.save(state, recovery / 'calibrated-state.pt')
    model.config.save_pretrained(recovery)
    compressor.update_config(recovery)
    del state
    model.save_pretrained(a.output, save_compressed=True, max_shard_size='4GB', save_original_format=False)
    # Restore all unquantized data directly from source bytes, not model state:
    # Transformers may omit MTP entirely or adapt numerical state on load.
    index = json.loads((a.output / 'model.safetensors.index.json').read_text())
    preserved = {}
    for shard in sorted(a.source.glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as reader:
            for name in reader.keys():
                if not targeted(name):
                    preserved[name] = reader.get_tensor(name)
    for shard in sorted(a.output.glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as reader:
            tensors = {n: reader.get_tensor(n) for n in reader.keys()}
        for name in tensors.keys() & preserved.keys():
            tensors[name] = preserved.pop(name)
        save_file(tensors, shard, metadata={'format': 'pt'})
    if preserved:
        save_file(preserved, a.output / 'retained-source.safetensors', metadata={'format': 'pt'})
        index['weight_map'].update({n: 'retained-source.safetensors' for n in preserved})
    index['metadata']['total_size'] = sum(v['bytes'] for v in inventory(a.output).values())
    (a.output / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    for path in a.source.iterdir():
        if path.is_file() and path.suffix in ('.json', '.jinja', '.txt', '.model') and path.name not in ('config.json', 'model.safetensors.index.json'):
            shutil.copy2(path, a.output / path.name)
    audit(a.source, a.output, a.group_size, a.head_bits)
    def file_sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    manifest = {'source': 'Qwen/Qwen3.5-9B', 'revision': SOURCE_REVISION, 'group_size': a.group_size, 'calibration_sha256': file_sha256(a.calibration), 'calibration_samples': len(samples), 'max_sequence_length': 2048, 'algorithm': 'GPTQ', 'activation_ordering': None, 'explicit_targets': selected, 'versions': {n: importlib.metadata.version(n) for n in ('llmcompressor', 'compressed-tensors', 'transformers', 'torch', 'datasets', 'safetensors')}, 'files': {p.name: file_sha256(p) for p in a.output.iterdir() if p.is_file()}}
    manifest['head_bits'] = a.head_bits
    manifest['head_strategy'] = groups['head']['weights']['strategy']
    (a.output / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
