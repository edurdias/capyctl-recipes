#!/usr/bin/env python3
"""Export channel-scaled E4M3 attention, copying every other tensor verbatim."""
import argparse
import hashlib
import json
import math
import mmap
import re
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open

SELECTED = re.compile(r"model\.llm\.layers\.(\d+)\.attn\.(wq_du|wk_dv|wv_dv|wr_du|gate_attn|wo_ud)\.weight$")

def header(path):
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(length)), length + 8

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

def write_index(shard, source_index, destination):
    actual, _ = header(shard)
    names = sorted(set(actual)-{'__metadata__'})
    total_size = sum(actual[name]['data_offsets'][1]-actual[name]['data_offsets'][0]
                     for name in names)
    index = json.loads(source_index.read_text())
    index.setdefault('metadata', {})['total_size'] = total_size
    index['weight_map'] = {name: shard.name for name in names}
    destination.write_text(json.dumps(index, indent=2)+'\n')
    written = json.loads(destination.read_text())
    if written['weight_map'] != index['weight_map'] or written['metadata'] != index['metadata']:
        raise ValueError('Written index does not match the actual tensor header')
    return {'bytes': destination.stat().st_size, 'sha256': digest_file(destination),
            'tensor_count': len(names), 'total_size': total_size,
            'header_coverage_exact': True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path.home()/'models/Inkling-2-Mini-ed40e7da')
    parser.add_argument('--output', type=Path, default=Path.home()/'models/Inkling-2-Mini-capyctl-fp8attn')
    parser.add_argument('--rebuild-index-only', action='store_true',
                        help='Repair an existing export index and manifest without rewriting tensors')
    args = parser.parse_args()
    source, output = args.source, args.output
    revision = (source/'REVISION').read_text().strip()
    if not revision.startswith('ed40e7da'):
        raise ValueError('Expected pinned original revision ed40e7da')
    if args.rebuild_index_only:
        manifest_path = output/'fp8-attention-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['index'] = write_index(
            output/'model-00001-of-00001.safetensors',
            source/'model.safetensors.index.json', output/'model.safetensors.index.json')
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(manifest['index'], indent=2), flush=True)
        return
    output.mkdir(exist_ok=False)
    shard = next(source.glob('model-*.safetensors'))
    old, old_start = header(shard)
    names = sorted(set(old)-{'__metadata__'})
    chosen = [n for n in names if SELECTED.fullmatch(n)]
    if len(chosen) != 24 * 6:
        raise ValueError(f'Expected 144 selected weights, got {len(chosen)}')
    config = json.loads((source/'config.json').read_text())
    text = config.get('text_config', config)
    loop = text['loop_config']
    print('Loop configuration:', json.dumps(loop), flush=True)
    # The Mini core is the middle twelve of twenty-four physical layers.
    core_start = loop['core_start']
    core_end = loop['core_end']
    passes = loop['num_passes']
    quantized = {}
    tensors = {}
    manifest = {'source_revision': revision, 'scheme': 'E4M3FN per-output-channel FP32 dequantization scales; dynamic per-token FP8 activation', 'selected_pattern': SELECTED.pattern, 'tensors': [], 'files': {}}
    with safe_open(shard, framework='pt', device='cpu') as sf:
        for name in chosen:
            weight = sf.get_tensor(name)
            if weight.dtype != torch.bfloat16 or weight.ndim != 2:
                raise ValueError(f'{name}: expected BF16 matrix')
            fp32 = weight.float()
            scale = fp32.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny) / 448.0
            q = (fp32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            scale_name = name.removesuffix('.weight') + '.weight_scale'
            quantized[name] = q.view(torch.uint8).numpy().tobytes()
            quantized[scale_name] = scale.numpy().tobytes()
            tensors[name] = {'dtype': 'F8_E4M3', 'shape': list(q.shape)}
            tensors[scale_name] = {'dtype': 'F32', 'shape': list(scale.shape)}
            print(name, list(weight.shape), flush=True)
    for name in names:
        if name not in tensors:
            tensors[name] = {'dtype': old[name]['dtype'], 'shape': old[name]['shape']}
    cursor = 0
    for name, info in tensors.items():
        nbytes = len(quantized[name]) if name in quantized else old[name]['data_offsets'][1]-old[name]['data_offsets'][0]
        info['data_offsets'] = [cursor, cursor+nbytes]
        cursor += nbytes
    new_header = {'__metadata__': old.get('__metadata__', {'format': 'pt'}), **tensors}
    encoded = json.dumps(new_header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    target = output/shard.name
    with shard.open('rb') as src, target.open('wb') as dst:
        mapped = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
        dst.write(struct.pack('<Q', len(encoded)))
        dst.write(encoded)
        for name, info in tensors.items():
            h = hashlib.sha256()
            if name in quantized:
                data = quantized.pop(name)
                dst.write(data)
                h.update(data)
            else:
                start, end = old[name]['data_offsets']
                for offset in range(start, end, 8 << 20):
                    chunk = mapped[old_start+offset:old_start+min(end, offset+(8 << 20))]
                    dst.write(chunk)
                    h.update(chunk)
            record = {'name': name, 'dtype': info['dtype'], 'shape': info['shape'], 'bytes': info['data_offsets'][1]-info['data_offsets'][0], 'sha256': h.hexdigest(), 'converted': bool(SELECTED.fullmatch(name)), 'added_scale': name.endswith('.weight_scale')}
            if name in old:
                start, end = old[name]['data_offsets']
                original_hash = hashlib.sha256()
                for offset in range(start, end, 8 << 20):
                    original_hash.update(mapped[old_start+offset:old_start+min(end, offset+(8 << 20))])
                record['source_sha256'] = original_hash.hexdigest()
                if not record['converted'] and record['source_sha256'] != record['sha256']:
                    raise ValueError(f'Untouched tensor mismatch: {name}')
            manifest['tensors'].append(record)
        mapped.close()
    # Read back the written bytes, not only the buffers passed to write().
    actual_header, actual_start = header(target)
    with target.open('rb') as f:
        for record in manifest['tensors']:
            start, end = actual_header[record['name']]['data_offsets']
            f.seek(actual_start+start)
            h = hashlib.sha256()
            remaining = end-start
            while remaining:
                chunk = f.read(min(remaining, 8 << 20))
                if not chunk:
                    raise ValueError('Truncated output checkpoint')
                h.update(chunk)
                remaining -= len(chunk)
            if h.hexdigest() != record['sha256']:
                raise ValueError(f'Output byte audit failed: {record["name"]}')
    for path in source.iterdir():
        if path.name in {shard.name, 'config.json', 'hf_quant_config.json', 'model.safetensors.index.json'} or path.name.startswith('download-') or path.suffix == '.lock':
            continue
        destination = output/path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
            if path.suffix == '.safetensors':
                original_hash = digest_file(path)
                actual_hash = digest_file(destination)
                if original_hash != actual_hash:
                    raise ValueError(f'File copy mismatch: {path.name}')
                manifest['files'][path.name] = {'sha256': actual_hash, 'source_sha256': original_hash, 'bytes': path.stat().st_size}
    manifest['index'] = write_index(
        target, source/'model.safetensors.index.json', output/'model.safetensors.index.json')
    precision = json.loads((source/'hf_quant_config.json').read_text())
    for group in precision['model']['parameter_groups']:
        group['tensors'] = [n for n in group['tensors'] if n not in chosen]
    precision['model']['parameter_groups'].append({'compute_dtype': 'fp8', 'output_dtype': 'float8_e4m3fn', 'required_activation_quantization': {'format': 'fp8_e4m3', 'strategy': 'token', 'dynamic': True}, 'tensors': chosen, 'weight_block_shape': [], 'weight_transform': 'fp8_per_channel'})
    # Precision is authoritative in hf_quant_config.json, not a top-level stub.
    config.pop('quantization_config', None)
    config['capyctl_fp8_attention'] = {'source_revision': revision, 'weight_scale': 'per_output_channel_float32', 'projections': ['wq_du', 'wk_dv', 'wv_dv', 'wr_du', 'gate_attn', 'wo_ud']}
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'hf_quant_config.json').write_text(json.dumps(precision, indent=2)+'\n')
    before = after = 0
    for name in chosen:
        layer = int(SELECTED.fullmatch(name)[1])
        visits = passes if core_start <= layer < core_end else 1
        elements = math.prod(old[name]['shape'])
        rows = old[name]['shape'][0]
        before += 2 * elements * visits
        after += (elements + 4 * rows) * visits
    manifest['logical_attention_bytes_per_decoded_token'] = {'label': '[INFERENCE] logical weight streams; excludes KV/activations/kernel rereads/DSpark amortization', 'core_start': core_start, 'core_end': core_end, 'passes': passes, 'before': before, 'after': after, 'saving': before-after}
    manifest['main_checkpoint'] = {'bytes': target.stat().st_size, 'source_bytes': shard.stat().st_size, 'sha256': digest_file(target)}
    manifest['audit'] = {'all_output_tensor_bytes_verified': True, 'all_untouched_source_tensor_bytes_identical': True, 'experts_unchanged': True, 'selected_weights': len(chosen)}
    (output/'fp8-attention-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'tensors'}, indent=2), flush=True)

if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
