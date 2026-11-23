#!/usr/bin/env python3
"""Export at most three prespecified layouts; never select using held-out results."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

H = Path.home()
ROOT = H / 'clef-build'
SOURCE = H / 'models/clef-flash'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def export(tag, layout, activation, tuned=False):
    out = H / 'models' / tag
    assert not out.exists(), f'Refusing to overwrite {out}'
    out.mkdir()
    mapping = json.loads((ROOT / 'mapping.json').read_text())['modules']
    index = json.loads((SOURCE / 'model.safetensors.index.json').read_text())['weight_map']
    groups = json.loads((H / 'models/clef-flash-nvfp4-e676d8e6/config.json').read_text())['quantization_config']
    fp8 = copy.deepcopy(groups['config_groups']['group_0'])
    nvfp4 = copy.deepcopy(groups['config_groups']['group_1'])
    groups['config_groups'] = {}
    source_modules = set()
    inventory = []
    target_config = {}
    weight_map = {}
    buffer = {}
    shard_bytes = 0
    shard_id = 0
    total_bytes = 0
    def flush():
        nonlocal buffer, shard_bytes, shard_id
        if not buffer: return
        shard_id += 1
        name = f'model-{shard_id:05d}.safetensors'
        save_file(buffer, str(out / name), metadata={'format': 'pt'})
        for key in buffer: weight_map[key] = name
        buffer = {}; shard_bytes = 0
    def put(key, tensor):
        nonlocal shard_bytes, total_bytes
        tensor = tensor.cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        if shard_bytes + size > 3 * 1024**3: flush()
        buffer[key] = tensor
        shard_bytes += size; total_bytes += size
        inventory.append({'name': key, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'bytes': size})
    for fused, names in mapping.items():
        layer = int(fused.split('.layers.')[1].split('.')[0])
        family = 'mlp' if '.mlp.' in fused else 'attention' if '.self_attn.' in fused else 'deltanet'
        mode = layout[f'{layer:02d}-{family}']
        for name in names:
            if mode == 'bf16': continue
            source_modules.add(name)
            file_mode = 'fp8t' if mode == 'fp8' and tuned else mode
            tensors = load_file(ROOT / 'bank' / (name + '.' + file_mode + '.export.safetensors'))
            if mode == 'fp8' and activation == 'dynamic': tensors.pop('input_scale', None)
            for key, tensor in tensors.items(): put(name + '.' + key, tensor)
            target_config[name] = mode
    for filename in sorted(set(index.values())):
        with safe_open(SOURCE / filename, framework='pt') as stream:
            for name in stream.keys():
                if name.endswith('.weight') and name[:-7] in source_modules: continue
                put(name, stream.get_tensor(name))
    flush()
    fp8['targets'] = ['re:.*layers\\.' + re.escape(n.split('.layers.')[1]) + '$' for n,m in target_config.items() if m == 'fp8']
    nvfp4['targets'] = ['re:.*layers\\.' + re.escape(n.split('.layers.')[1]) + '$' for n,m in target_config.items() if m == 'nvfp4']
    if activation == 'static':
        fp8['input_activations'].update(dynamic=False, strategy='tensor', observer='minmax')
    if fp8['targets']: groups['config_groups']['fp8'] = fp8
    if nvfp4['targets']: groups['config_groups']['nvfp4'] = nvfp4
    groups['format'] = 'mixed-precision' if nvfp4['targets'] else 'float-quantized'
    config = json.loads((SOURCE / 'config.json').read_text())
    config['quantization_config'] = groups
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (out / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}, indent=2) + '\n')
    for path in SOURCE.iterdir():
        if path.name in ('config.json', 'model.safetensors.index.json') or path.suffix == '.safetensors': continue
        if path.is_file(): shutil.copy2(path, out / path.name)
    shutil.copy2(SOURCE / 'joint_head.safetensors', out / 'joint_head.safetensors')
    head = sha(out / 'joint_head.safetensors')
    assert head == sha(SOURCE / 'joint_head.safetensors') == '19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0'
    def tensor_hash(directory, name):
        idx = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
        with safe_open(directory / idx[name], framework='pt') as stream:
            value = stream.get_tensor(name)
            assert value.dtype == torch.bfloat16
            return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    lexical = tensor_hash(SOURCE, 'lm_head.weight')
    assert lexical == tensor_hash(out, 'lm_head.weight')
    # Fused NVFP4 projections must share both activation and global weight scales.
    fused_audit = []
    for fused, names in mapping.items():
        if not all(target_config.get(n) == 'nvfp4' for n in names): continue
        for key in ('weight_global_scale', 'input_global_scale'):
            values = [load_file(ROOT / 'bank' / (n + '.nvfp4.export.safetensors'))[key] for n in names]
            assert all(torch.equal(values[0], v) for v in values[1:]), (fused, key)
        fused_audit.append(fused)
    audit = {'tag': tag, 'layout': layout, 'activation': activation, 'head_sha256': head, 'lexical_head_sha256': lexical, 'head_byte_identity': True, 'lexical_head_byte_identity': True, 'weight_tensor_bytes': total_bytes, 'package_bytes': sum(p.stat().st_size for p in out.iterdir() if p.is_file()), 'fused_nvfp4_shared_scale_modules': fused_audit, 'tensor_inventory': inventory, 'provenance': 'ModelOpt 0.47.0 calibration and rounding, own compressed-tensors exporter; sensitivity selection excludes heldout'}
    (ROOT / f'{tag}-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


if __name__ == '__main__':
    import sys
    spec = json.loads(Path(sys.argv[1]).read_text())
    assert len(spec['candidates']) <= 3
    for candidate in spec['candidates']:
        audit = export(**candidate)
        print(json.dumps({k:v for k,v in audit.items() if k != 'tensor_inventory'}), flush=True)
