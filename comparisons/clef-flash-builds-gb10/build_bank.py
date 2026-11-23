#!/usr/bin/env python3
"""ModelOpt calibration/rounding bank for controlled numerical ablations and exports."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import modelopt
import modelopt.torch.quantization as mtq
from modelopt.torch.export import quant_utils as quant

H = Path.home()
ROOT = H / 'clef-build'
SOURCE = H / 'models/clef-flash'


def converted(weight, samples, amax, mode):
    module = torch.nn.Sequential(torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype))
    module[0].weight.data = weight
    cfg = copy.deepcopy(mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG if mode == 'fp8' else mtq.NVFP4_DEFAULT_CFG)
    if mode == 'fp8':
        cfg['quant_cfg'][2]['cfg'] = {'num_bits': (4, 3), 'axis': None}
    calibration = samples.to(device=weight.device, dtype=weight.dtype).clone()
    calibration[0, 0] = amax
    mtq.quantize(module, cfg, lambda m: m(calibration))
    layer = module[0]
    scale = quant.get_weight_scaling_factor(layer).float()
    if mode == 'fp8':
        scale = scale.bfloat16().float()
    scale2 = quant.get_weight_scaling_factor_2(layer)
    act = quant.get_activation_scaling_factor(layer).float()
    fmt = quant.get_quantization_format(layer)
    packed = quant.to_quantized_weight(weight, scale, fmt, scale2, block_size=16)
    qdq = layer.weight_quantizer(weight).to(torch.bfloat16)
    if mode == 'fp8':
        result = {'weight': packed, 'weight_scale': scale.bfloat16().reshape(-1, 1), 'input_scale': act.reshape(1)}
    else:
        result = {'weight_packed': packed.reshape(weight.shape[0], -1), 'weight_scale': scale.to(torch.float8_e4m3fn).reshape(weight.shape[0], -1), 'weight_global_scale': scale2.float().reciprocal().reshape(1), 'input_global_scale': act.reciprocal().reshape(1)}
    if mode == 'fp8':
        qdq = (result['weight'].float() * result['weight_scale']).to(torch.bfloat16)
    else:
        packed_bytes = result['weight_packed']
        nibbles = torch.stack((packed_bytes & 15, packed_bytes >> 4), dim=-1).reshape_as(weight).long()
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=weight.device)
        qdq = (lut[nibbles].reshape(weight.shape[0], -1, 16) * result['weight_scale'].float().unsqueeze(-1) / result['weight_global_scale']).reshape_as(weight).to(torch.bfloat16)
    return result, qdq, act


def main():
    assert modelopt.__version__ == '0.47.0'
    torch.set_num_threads(2)
    (ROOT / 'bank').mkdir(exist_ok=True)
    am = json.loads((ROOT / 'activation-amax.json').read_text())
    samples = load_file(ROOT / 'activation-samples.safetensors')
    index = json.loads((SOURCE / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {f: safe_open(SOURCE / f, framework='pt', device='cuda') for f in set(index.values())}
    source = lambda name: handles[index[name]].get_tensor(name)
    mapping = {}
    for layer in range(32):
        p = f'model.language_model.layers.{layer}'
        mapping[p + '.mlp.gate_up_proj'] = [p + '.mlp.gate_proj', p + '.mlp.up_proj']
        mapping[p + '.mlp.down_proj'] = [p + '.mlp.down_proj']
        if layer % 4 == 3:
            mapping[p + '.self_attn.qkv_proj'] = [p + '.self_attn.' + q + '_proj' for q in ('q', 'k', 'v')]
            mapping[p + '.self_attn.o_proj'] = [p + '.self_attn.o_proj']
        else:
            mapping[p + '.linear_attn.in_proj_qkvz'] = [p + '.linear_attn.in_proj_qkv', p + '.linear_attn.in_proj_z']
            mapping[p + '.linear_attn.out_proj'] = [p + '.linear_attn.out_proj']
    # Runtime prefixes may omit language_model; identify them from observed hooks.
    runtime = {}
    for p in mapping:
        options = [p, p.replace('model.language_model.', 'model.')]
        found = [v for v in options if v in am]
        assert len(found) == 1, (p, found, list(am)[:4])
        runtime[p] = found[0]
    (ROOT / 'mapping.json').write_text(json.dumps({'modules': mapping, 'runtime': runtime}, indent=2))
    units = {}
    roundtrip = []
    published = safe_open(H / 'clef-flash-fp8-7b43c2d7/model.safetensors', framework='pt', device='cuda')
    for fused, names in mapping.items():
        run = runtime[fused]
        weights = [source(n + '.weight') for n in names]
        weight = torch.cat(weights)
        layer = int(fused.split('.layers.')[1].split('.')[0])
        family = 'mlp' if '.mlp.' in fused else 'attention' if '.self_attn.' in fused else 'deltanet'
        units.setdefault(f'{layer:02d}-{family}', []).append(run)
        # Match the existing serialized format before building new tensors.
        for name in names:
            pw = published.get_tensor(name + '.weight')
            ps = published.get_tensor(name + '.weight_scale')
            dequant = pw.float() * ps.float()
            result, _, _ = converted(dequant, samples[run], am[run], 'fp8')
            assert result['weight'].shape == pw.shape and result['weight'].dtype == pw.dtype
            assert result['weight_scale'].shape == ps.shape and result['weight_scale'].dtype == ps.dtype
            roundtrip.append({'module': name, 'format_match': True, 'weight_equal_fraction': float((result['weight'].float() == pw.float()).float().mean()), 'scales_max_absolute_difference': float((result['weight_scale']-ps).abs().max())})
            del pw, ps, dequant, result
        for mode in ('fp8', 'nvfp4'):
            result, qdq, act = converted(weight, samples[run], am[run], mode)
            save_file({**{k:v.detach().cpu().contiguous() for k,v in result.items()}, 'qdq': qdq.cpu(), 'input_scale': torch.tensor([am[run]/448], dtype=torch.float32), 'nvfp4_input_scale': torch.tensor([am[run]/2688], dtype=torch.float32)}, str(ROOT / 'bank' / (run + '.' + mode + '.safetensors')))
            offset = 0
            for name, w in zip(names, weights):
                tensors = {}
                for key, value in result.items():
                    tensors[key] = (value[offset:offset+w.shape[0]] if value.ndim > 1 else value).detach().cpu().contiguous()
                save_file(tensors, str(ROOT / 'bank' / (name + '.' + mode + '.export.safetensors')))
                offset += w.shape[0]
            del result, qdq
        print('converted', fused, flush=True)
        del weights, weight
    (ROOT / 'units.json').write_text(json.dumps(units, indent=2))
    (ROOT / 'roundtrip.json').write_text(json.dumps({'modelopt': modelopt.__version__, 'published_format': 'compressed-tensors float-quantized, static channel FP8 weights', 'modules': roundtrip}, indent=2))


if __name__ == '__main__':
    main()
