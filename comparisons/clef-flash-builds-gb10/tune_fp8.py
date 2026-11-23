#!/usr/bin/env python3
"""Calibration-only channel-scale search, using ModelOpt's FP8 rounding/export."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from modelopt.torch.export.quant_utils import to_quantized_weight, QUANTIZATION_FP8_PC_PT

H = Path.home()
ROOT = H / 'clef-build'
SOURCE = H / 'models/clef-flash'


def main():
    torch.set_num_threads(2)
    mapping = json.loads((ROOT / 'mapping.json').read_text())
    index = json.loads((SOURCE / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {f: safe_open(SOURCE / f, framework='pt', device='cuda') for f in set(index.values())}
    samples = load_file(ROOT / 'activation-samples.safetensors')
    receipts = []
    for fused, names in mapping['modules'].items():
        runtime = mapping['runtime'][fused]
        bank = load_file(ROOT / 'bank' / (runtime + '.fp8.safetensors'), device='cuda')
        weight = torch.cat([handles[index[n+'.weight']].get_tensor(n+'.weight') for n in names]).float()
        x = samples[runtime][:128].cuda().float()
        diagonal = x.square().mean(dim=0).clamp_min(1e-10)
        scales = bank['weight_scale'].flatten().float()
        best_scales = scales.clone()
        def loss(q):
            residual = q.float() - weight
            diagonal_loss = (residual.square() * diagonal).sum(dim=1)
            reconstruction = torch.nn.functional.linear(x, residual).square().mean(dim=0)
            return .75 * diagonal_loss + .25 * reconstruction
        best_loss = loss(bank['weight'].float() * scales[:, None])
        before = best_loss.sum().item()
        for ratio in (.8, .825, .85, .875, .9, .925, .95, .975, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2):
            candidate = (scales * ratio).bfloat16().float()
            # Clamp deliberately before the ModelOpt pack operation: candidate scales
            # below minmax represent a measured clipping choice, not FP8 overflow.
            clipped = torch.maximum(torch.minimum(weight, candidate[:, None]*448), -candidate[:, None]*448)
            packed = to_quantized_weight(clipped, candidate, QUANTIZATION_FP8_PC_PT)
            cost = loss(packed.float() * candidate[:, None])
            improve = cost < best_loss
            best_scales = torch.where(improve, candidate, best_scales)
            best_loss = torch.minimum(cost, best_loss)
        clipped = torch.maximum(torch.minimum(weight, best_scales[:, None]*448), -best_scales[:, None]*448)
        packed = to_quantized_weight(clipped, best_scales, QUANTIZATION_FP8_PC_PT)
        bank.update(weight=packed, weight_scale=best_scales[:, None].bfloat16(), qdq=(packed.float()*best_scales[:, None]).bfloat16())
        save_file({k:v.cpu().contiguous() for k,v in bank.items()}, str(ROOT / 'bank' / (runtime + '.fp8t.safetensors')))
        offset = 0
        for name in names:
            rows = handles[index[name+'.weight']].get_slice(name+'.weight').get_shape()[0]
            save_file({'weight': packed[offset:offset+rows].cpu().contiguous(), 'weight_scale': best_scales[offset:offset+rows, None].bfloat16().cpu().contiguous(), 'input_scale': bank['input_scale'].cpu()}, str(ROOT / 'bank' / (name + '.fp8t.export.safetensors')))
            offset += rows
        receipts.append({'module': fused, 'calibration_loss_before': before, 'calibration_loss_after': best_loss.sum().item(), 'changed_channel_fraction': float((best_scales != scales).float().mean())})
        print('tuned', fused, receipts[-1], flush=True)
    (ROOT / 'fp8-tuning.json').write_text(json.dumps({'method': 'ModelOpt FP8 pack; per-channel grid 0.80..1.20 x minmax scale; 75% diagonal-weighted error plus 25% calibration output error; heldout excluded', 'modules': receipts}, indent=2))


if __name__ == '__main__':
    main()
