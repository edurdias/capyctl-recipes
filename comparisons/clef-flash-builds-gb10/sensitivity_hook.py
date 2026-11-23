"""Private experiment hook; never enabled for candidate serving or latency."""
import json
import os
from pathlib import Path
import threading
import time


def install():
    import torch
    from safetensors.torch import load_file
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    root = Path(os.environ['CLEF_SENSITIVITY_ROOT'])
    original_init = LinearBase.__init__
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
    # Keep input projections observable; packed BF16 otherwise bypasses apply().
    Qwen3_5GatedDeltaNet.finalize_fused_in_proj = lambda self: None
    original_apply = UnquantizedLinearMethod.apply
    stats = {}
    cached = {}
    control = {'generation': -1}
    stamp = None

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.clef_prefix = kwargs.get('prefix', args[5] if len(args) > 5 else '')

    def apply(self, layer, x, bias=None):
        nonlocal stamp, control
        p = getattr(layer, 'clef_prefix', '')
        if 'layers.' not in p or not any(s in p for s in ('.mlp.', '.self_attn.', '.linear_attn.')):
            return original_apply(self, layer, x, bias)
        if any(s in p for s in ('in_proj_ba', 'in_proj_a', 'in_proj_b')):
            return original_apply(self, layer, x, bias)
        now = (root / 'control.json').stat().st_mtime_ns
        if now != stamp:
            control = json.loads((root / 'control.json').read_text())
            stamp = now
        if control.get('calibrate'):
            flat = x.detach().reshape(-1, x.shape[-1])
            amax = float(flat.abs().max())
            if p not in stats:
                stats[p] = {'amax': amax, 'samples': flat[::max(1, flat.shape[0] // 8)][:8].float().cpu()}
            else:
                stats[p]['amax'] = max(stats[p]['amax'], amax)
                if stats[p]['samples'].shape[0] < 256:
                    stats[p]['samples'] = torch.cat((stats[p]['samples'], flat[::max(1, flat.shape[0] // 8)][:8].float().cpu()))
        mode = control.get('layout', {}).get(p, control.get('mode', 'bf16'))
        active = control.get('prefixes', []) if 'layout' not in control else control['layout']
        if mode == 'bf16' or p not in active:
            return original_apply(self, layer, x, bias)
        key = (control['generation'], p)
        if key not in cached:
            if not cached or next(iter(cached))[0] != control['generation']:
                cached.clear()
            bank = load_file(str(root / 'bank' / (p + '.' + mode + '.safetensors')), device=str(x.device))
            cached[key] = bank
        bank = cached[key]
        activation = 'nvfp4' if mode == 'nvfp4' else control.get('activation')
        if activation == 'dynamic':
            scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448
            qx = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
        elif activation == 'static':
            scale = bank['input_scale']
            qx = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
        elif activation == 'nvfp4':
            global_scale = bank['nvfp4_input_scale']
            groups = x.float().reshape(*x.shape[:-1], -1, 16)
            scale = (groups.abs().amax(dim=-1, keepdim=True) / (6 * global_scale)).clamp(max=448).to(torch.float8_e4m3fn).float() * global_scale
            scale = scale.clamp_min(1e-20)
            z = groups / scale
            lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
            edges = (lut[:-1] + lut[1:]) / 2
            indices = torch.bucketize(z.abs().contiguous(), edges)
            qx = (lut[indices] * z.sign() * scale).reshape_as(x)
        else:
            qx = x
        return torch.nn.functional.linear(qx.to(x.dtype), bank['qdq'].to(x.dtype), bias)

    def flush():
        from safetensors.torch import save_file
        while True:
            time.sleep(3)
            if stats and control.get('calibrate'):
                snapshot = dict(stats)
                (root / 'activation-amax.json').write_text(json.dumps({p: s['amax'] for p, s in snapshot.items()}, indent=2))
                save_file({p: s['samples'] for p, s in snapshot.items()}, str(root / 'activation-samples.tmp'))
                (root / 'activation-samples.tmp').replace(root / 'activation-samples.safetensors')

    LinearBase.__init__ = init
    UnquantizedLinearMethod.apply = apply
    threading.Thread(target=flush, daemon=True).start()
