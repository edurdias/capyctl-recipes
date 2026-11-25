"""In-process reference capture; imported only by the owned capture launcher."""
import functools
import json
import os
from pathlib import Path
import re

REVISION = '9838c91537e6be21dec2f5276b784211a06fbeea'
LAYERS = (0, 18, 38, 60, 86)
CASES = ('chat', 'code', 'tool')
TOKENS = 128


def install(module):
    original = module.MambaMixer2.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        match = re.search(r'\.layers\.(\d+)\.mixer$', kwargs.get('prefix', ''))
        if match is None or int(match[1]) not in LAYERS:
            return
        layer = int(match[1])
        captured = {}
        directory = Path(os.environ['NEMOTRON_CAPTURE_DIR'])

        def capture(linear, inputs):
            import torch
            from safetensors.torch import save_file
            control = directory / 'case.json'
            if not control.exists():
                return
            case = json.loads(control.read_text())['case']
            if case not in CASES:
                raise ValueError('Unknown capture case')
            parts = captured.setdefault(case, [])
            count = sum(value.shape[0] for value in parts)
            remaining = TOKENS - count
            if remaining <= 0:
                return
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:remaining]
            if not values.numel():
                return
            parts.append(values.to(device='cpu', dtype=torch.bfloat16).contiguous())
            destination = directory / f'{case}-layer-{layer}.safetensors'
            temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
            save_file({'input': torch.cat(parts)}, str(temporary),
                      metadata={'revision': REVISION, 'case': case, 'layer': str(layer),
                                'kind': 'unquantized in_proj forward input'})
            temporary.replace(destination)

        self.in_proj.register_forward_pre_hook(capture)

    module.MambaMixer2.__init__ = initialize
