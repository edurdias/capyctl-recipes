"""ModelOpt-compatible block-16 NVFP4, with CPU-only searched E4M3 scales.

Reference: NVIDIA Model-Optimizer 0.47.0 nvfp4_tensor.py (_cast_fp4 and
quantize). This implements the storage format, not a ModelOpt calibration run.
"""
import math
import torch

MULTIPLIERS = [0.60 + 0.025 * i for i in range(21)]
LEVELS = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
BOUNDS = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])


def codes(values):
    """Saturating E2M1 with round-to-nearest-even, including midpoint ties."""
    absolute = values.abs().contiguous()
    ordinal = torch.bucketize(absolute, BOUNDS).to(torch.uint8)
    # Odd lower-code midpoints round up; the other four round down.
    ordinal += ((absolute == .75) | (absolute == 1.75) | (absolute == 3.5)).to(torch.uint8)
    return ordinal | ((values < 0).to(torch.uint8) << 3)


def pack(values):
    quantized = codes(values)
    return quantized[..., 0::2] | (quantized[..., 1::2] << 4)


def unpack(packed):
    result = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.float32)
    for offset, nibble in ((0, packed & 15), (1, packed >> 4)):
        result[..., offset::2] = LEVELS[(nibble & 7).long()] * torch.where(nibble >= 8, -1., 1.)
    return result


@torch.inference_mode()
def search(weights, chunk_blocks=16384):
    """One FP32 in_proj plus output buffers; search temporaries are chunk bounded.

    One global scale spans the entire already-concatenated Mamba projection,
    including z/x/B/C/dt. All five runtime partitions therefore share it.
    """
    if weights.device.type != 'cpu' or weights.ndim != 2 or weights.shape[1] % 16:
        raise ValueError('Expected CPU matrix with input width divisible by 16')
    if chunk_blocks < 1 or not torch.isfinite(weights).all():
        raise ValueError('Invalid chunk size or nonfinite source weights')
    weights = weights.float()
    maximum = weights.abs().max().item()
    global_scale = torch.tensor(maximum / (6 * 448) if maximum else 1., dtype=torch.float32)
    if not torch.isfinite(global_scale) or global_scale <= 0:
        raise ValueError('Invalid global weight scale')
    blocks = weights.reshape(-1, 16)
    packed = torch.empty((len(blocks), 8), dtype=torch.uint8)
    scales = torch.empty(len(blocks), dtype=torch.float8_e4m3fn)
    squared_error = amax_error = energy = 0.
    for start in range(0, len(blocks), chunk_blocks):
        block = blocks[start:start + chunk_blocks]
        initial = block.abs().amax(-1) / (6 * global_scale)
        best_error = torch.full_like(initial, math.inf)
        best_scale = torch.empty_like(initial)
        for multiplier in MULTIPLIERS:
            scale = (initial * multiplier).clamp(2**-9, 448).to(torch.float8_e4m3fn).float()
            denominator = (scale * global_scale)[:, None]
            normalized = block / denominator
            quantized = codes(normalized)
            decoded = LEVELS[(quantized & 7).long()] * torch.where(quantized >= 8, -1., 1.)
            error = (decoded * denominator - block).square().sum(-1)
            take = error < best_error
            best_error = torch.where(take, error, best_error)
            best_scale = torch.where(take, scale, best_scale)
            if multiplier == MULTIPLIERS[-1]:
                amax_error += error.double().sum().item()
        scales[start:start + len(block)] = best_scale.to(torch.float8_e4m3fn)
        packed[start:start + len(block)] = pack(block / (best_scale * global_scale)[:, None])
        squared_error += best_error.double().sum().item()
        energy += block.double().square().sum().item()
    return (packed.reshape(weights.shape[0], weights.shape[1] // 2),
            scales.reshape(weights.shape[0], weights.shape[1] // 16), global_scale,
            {'relative_l2_error': math.sqrt(squared_error / energy) if energy else 0.,
             'squared_error': squared_error, 'amax_squared_error': amax_error})
