#!/usr/bin/env python3
"""Read-only quantized-head instrumentation, including spawned engine workers.

After GPU release, put this directory on PYTHONPATH and run the approved
SGLang 0.5.22 launch_server normally. Require the confirmation marker in logs;
missing instrumentation or a startup import error is a failed load check.
"""
import importlib.metadata
import json
import os

assert os.environ.get('QWEN35_GPU_RELEASE') == 'GPU owner GPU free'
assert importlib.metadata.version('sglang') == '0.5.22'
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import CompressedTensorsWNA16

original = CompressedTensorsWNA16.process_weights_after_loading

def inspected(self, layer):
    result = original(self, layer)
    config = self.kernel_config
    is_head = config.full_weight_shape == (4096, 248320)
    print('QWEN35_KERNEL_AUDIT ' + json.dumps({'class': type(self).__name__, 'kernel': 'gptq_marlin', 'weight_type': str(config.weight_type), 'pack_factor': self.pack_factor, 'layer_class': type(layer).__name__, 'full_weight_shape': list(config.full_weight_shape), 'group_size': self.group_size, 'symmetric': self.symmetric, 'is_output_head': is_head}), flush=True)
    if is_head:
        assert self.symmetric and ((self.group_size in (32, 128) and self.pack_factor == 8) or (self.group_size == -1 and self.pack_factor == 4))
        assert hasattr(layer, 'weight_packed') and hasattr(layer, 'weight_scale')
        print('QWEN35_QUANTIZED_HEAD_MARLIN_CONFIRMED', flush=True)
    return result

CompressedTensorsWNA16.process_weights_after_loading = inspected

original_apply = CompressedTensorsWNA16.apply_weights

def inspected_apply(self, layer, x, bias):
    result = original_apply(self, layer, x, bias)
    if self.kernel_config.full_weight_shape == (4096, 248320) and not getattr(self, '_head_execution_logged', False):
        print('QWEN35_HEAD_KERNEL_EXECUTED ' + json.dumps({'kernel': 'gptq_marlin', 'weight_type': str(self.kernel_config.weight_type), 'pack_factor': self.pack_factor, 'group_size': self.group_size}), flush=True)
        self._head_execution_logged = True
    return result

CompressedTensorsWNA16.apply_weights = inspected_apply
