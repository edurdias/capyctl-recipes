#!/usr/bin/env python3
"""CPU check of captured reference inputs, before full conversion or candidate GPU use."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from capture_hook import CASES, LAYERS, REVISION, TOKENS
from checkpoint import digest, headers, read_tensor
from nvfp4 import LEVELS, codes, search, unpack


@torch.inference_mode()
def activation_quantize(values, fp8_scale):
    """Return reference FP8 and NVFP4 using the same calibrated absolute bound."""
    original = values.float()
    fp8 = (original / fp8_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * fp8_scale
    global_scale = fp8_scale / 6
    blocks = original.reshape(-1, 16)
    scales = (blocks.abs().amax(-1) / (6 * global_scale)).clamp(2**-9, 448).to(torch.float8_e4m3fn).float()
    denominator = scales[:, None] * global_scale
    ordinal = codes(blocks / denominator)
    fp4 = LEVELS[(ordinal & 7).long()] * torch.where(ordinal >= 8, -1., 1.) * denominator
    return fp8, fp4.reshape_as(original)


def relative_l2(actual, reference):
    error = (actual.double() - reference.double()).square().sum().item()
    energy = reference.double().square().sum().item()
    return (error / energy) ** .5 if energy else 0.


@torch.inference_mode()
def check(source, captures):
    source, captures = Path(source), Path(captures)
    metadata = headers(source)
    evidence = []
    for layer in LAYERS:
        base = f'language_model.backbone.layers.{layer}.mixer.in_proj'
        weight = read_tensor(source, metadata[base + '.weight']).float()
        weight.mul_(read_tensor(source, metadata[base + '.weight_scale']))
        packed, scales, global_scale, weight_error = search(weight)
        candidate = (unpack(packed).reshape(-1, 16) * scales.float().reshape(-1, 1) * global_scale).reshape_as(weight)
        del packed, scales, global_scale
        fp8_scale = read_tensor(source, metadata[base + '.input_scale']).float().reshape(())
        for case in CASES:
            path = captures / f'{case}-layer-{layer}.safetensors'
            with safe_open(path, framework='pt', device='cpu') as handle:
                attributes = handle.metadata()
                if attributes.get('revision') != REVISION or attributes.get('case') != case or attributes.get('layer') != str(layer):
                    raise ValueError('Capture provenance mismatch: ' + str(path))
                values = handle.get_tensor('input').float()
            if values.shape != (TOKENS, weight.shape[1]) or not torch.isfinite(values).all():
                raise ValueError('Capture must contain 128 finite reference tokens per case/layer')
            fp8, fp4 = activation_quantize(values, fp8_scale)
            activation_global_scale = fp8_scale / 6
            blocks = values.reshape(-1, 16)
            block_scales = (blocks.abs().amax(-1) / (6 * activation_global_scale)).clamp(
                2**-9, 448).to(torch.float8_e4m3fn).float()
            rounding_threshold = .25 * block_scales[:, None] * activation_global_scale
            # Use all 128 tokens for activation statistics and 32 evenly spaced
            # tokens for the more expensive CPU projection comparisons.
            positions = torch.linspace(0, TOKENS - 1, 32).long()
            output_stats = {'activation_only': [0., 0.], 'weight_only': [0., 0.], 'combined': [0., 0.]}
            for positions_chunk in positions.split(8):
                reference = fp8[positions_chunk] @ weight.T
                alternatives = {'activation_only': fp4[positions_chunk] @ weight.T,
                                'weight_only': fp8[positions_chunk] @ candidate.T,
                                'combined': fp4[positions_chunk] @ candidate.T}
                energy = reference.double().square().sum().item()
                for name, value in alternatives.items():
                    output_stats[name][0] += (value.double() - reference.double()).square().sum().item()
                    output_stats[name][1] += energy
            evidence.append({'layer': layer, 'case': case, 'capture_sha256': digest(path),
                'tokens': TOKENS, 'projection_tokens': 32, 'calibrated_amax': (fp8_scale * 448).item(),
                'fp8_input_scale': fp8_scale.item(),
                'nvfp4_input_scale': activation_global_scale.item(),
                'below_fp4_rounding_threshold_fraction': (
                    blocks.abs() < rounding_threshold).float().mean().item(),
                'captured_absolute_max': values.abs().max().item(),
                'outside_calibrated_bound_fraction': (values.abs() > fp8_scale * 448).float().mean().item(),
                'fp8_activation_relative_l2': relative_l2(fp8, values),
                'fp4_activation_relative_l2': relative_l2(fp4, values),
                'fp8_zero_fraction': (fp8 == 0).float().mean().item(),
                'fp4_zero_fraction': (fp4 == 0).float().mean().item(),
                'projection_relative_l2': {name: (error / energy) ** .5 if energy else 0.
                                           for name, (error, energy) in output_stats.items()},
                'weight_search': weight_error})
        del weight, candidate
    return {'revision': REVISION, 'checked_layers': list(LAYERS), 'cases': list(CASES),
            'activation_bound': 'NVIDIA FP8 scale / 6, no new calibration',
            'rounding_threshold': 'abs(x) < 0.25 * rounded dynamic block-16 scale * (NVIDIA input_scale / 6); ties round to even zero',
            'status': 'measured CPU sensitivity; owner review required before conversion',
            'qualification': 'Not a task-quality, stopping, MTP-acceptance or GPU-kernel qualification',
            'results': evidence}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--captures', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = check(args.source, args.captures)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
