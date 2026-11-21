#!/usr/bin/env python3
"""Precision and export metadata tests; full linear/loop behavior uses the engine."""
import copy
import json
import hashlib
from pathlib import Path
import unittest
import struct
import subprocess
import sys
import tempfile

import torch
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig

ROOT = Path.home()/'models'

class PrecisionContract(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT/'Inkling-2-Mini-ed40e7da/hf_quant_config.json').read_text())
        self.exported = json.loads((ROOT/'Inkling-2-Mini-capyctl-fp8attn/hf_quant_config.json').read_text())

    def test_native_precision_is_unchanged(self):
        before = Inkling2QuantConfig(self.original)
        after = Inkling2QuantConfig(self.exported)
        selected = after.fp8_weights
        self.assertEqual(len(selected), 144)
        self.assertEqual(before.mxfp8_weights, after.mxfp8_weights)
        self.assertEqual(before.nvfp4_weights, after.nvfp4_weights)
        self.assertEqual(before.unembed_output_dtype, after.unembed_output_dtype)
        for name, dtype in before.tensor_dtypes.items():
            if name not in selected:
                self.assertEqual(dtype, after.tensor_dtypes[name], name)
        for name in selected:
            self.assertEqual(after.tensor_dtypes[name], torch.float8_e4m3fn)
            self.assertEqual(after.tensor_dtypes[name+'_scale'], torch.float32)

    def test_head_is_not_an_allowed_dense_target(self):
        changed = copy.deepcopy(self.exported)
        changed['model']['parameter_groups'][-1]['tensors'][0] = 'model.llm.unembed.weight'
        # Remove the original head group so this exercises the dense-target guard,
        # not the independent duplicate-precision guard.
        changed['model']['parameter_groups'] = [group for group in changed['model']['parameter_groups'] if group['tensors'] != ['model.llm.unembed.weight']]
        with self.assertRaisesRegex(ValueError, 'Unsupported dense FP8 tensor'):
            Inkling2QuantConfig(changed)

    def test_sink_is_not_an_allowed_dense_target(self):
        changed = copy.deepcopy(self.exported)
        target = 'model.llm.layers.0.attn.attn_sink_proj.weight'
        for group in changed['model']['parameter_groups'][:-1]:
            group['tensors'] = [name for name in group['tensors'] if name != target]
        changed['model']['parameter_groups'][-1]['tensors'][0] = target
        with self.assertRaisesRegex(ValueError, 'Unsupported dense FP8 tensor'):
            Inkling2QuantConfig(changed)

    def test_scale_layout_contract_is_explicit(self):
        changed = copy.deepcopy(self.exported)
        changed['model']['parameter_groups'][-1]['weight_block_shape'] = [128, 128]
        with self.assertRaisesRegex(ValueError, 'Unsupported dense FP8 precision'):
            Inkling2QuantConfig(changed)

    def test_model_config_cannot_shadow_precision_file(self):
        from sglang.srt.model_loader.checkpoint_quantization import resolve_checkpoint_quant_spec
        config = json.loads((ROOT/'Inkling-2-Mini-capyctl-fp8attn/config.json').read_text())
        self.assertIsNone(resolve_checkpoint_quant_spec(config))
        self.assertEqual(Inkling2QuantConfig.get_config_filenames(), ['hf_quant_config.json'])

class IndexExportContract(unittest.TestCase):
    def test_index_matches_header_for_export_and_metadata_only_repair(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory(prefix='inkling-index-test-') as temporary:
            source = Path(temporary)/'source'
            source.mkdir()
            output = Path(temporary)/'output'
            shard_name = 'model-00001-of-00001.safetensors'
            tensors = {
                f'model.llm.layers.{layer}.attn.{projection}.weight':
                    torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
                for layer in range(24)
                for projection in ('wq_du', 'wk_dv', 'wv_dv', 'wr_du', 'gate_attn', 'wo_ud')
            }
            tensors['native.weight'] = torch.tensor([1, 2], dtype=torch.bfloat16)
            save_file(tensors, str(source/shard_name), metadata={'format': 'pt'})
            save_file({'draft.weight': torch.ones(1)}, str(source/'mtp.safetensors'))
            (source/'REVISION').write_text('ed40e7da625f594fc9c4990f5f66be24873c22d7\n')
            (source/'config.json').write_text(json.dumps({'text_config': {
                'loop_config': {'core_start': 6, 'core_end': 18, 'num_passes': 3}}}))
            (source/'hf_quant_config.json').write_text(json.dumps({
                'model': {'parameter_groups': [{'tensors': list(tensors)}]}}))
            (source/'model.safetensors.index.json').write_text(json.dumps({
                'metadata': {'total_size': sum(t.numel()*t.element_size() for t in tensors.values()),
                             'preserved_key': 'keep this'},
                'weight_map': {name: shard_name for name in tensors}}))
            subprocess.run([sys.executable, str(Path(__file__).with_name('convert-fp8-attention.py')),
                            '--source', str(source), '--output', str(output)],
                           check=True, capture_output=True, text=True)
            with (output/shard_name).open('rb') as stream:
                size = struct.unpack('<Q', stream.read(8))[0]
                actual = json.loads(stream.read(size))
            actual.pop('__metadata__', None)
            index_path = output/'model.safetensors.index.json'
            index = json.loads(index_path.read_text())
            self.assertEqual(set(index['weight_map']), set(actual))
            self.assertEqual(set(index['weight_map'].values()), {shard_name})
            self.assertEqual(index['metadata'], {
                'total_size': sum(info['data_offsets'][1]-info['data_offsets'][0]
                                  for info in actual.values()),
                'preserved_key': 'keep this'})
            manifest = json.loads((output/'fp8-attention-manifest.json').read_text())
            self.assertTrue(manifest['index']['header_coverage_exact'])
            self.assertEqual(manifest['index']['tensor_count'], len(actual))
            self.assertEqual(manifest['index']['sha256'],
                             hashlib.sha256(index_path.read_bytes()).hexdigest())
            before = hashlib.sha256((output/shard_name).read_bytes()).hexdigest()
            # Reproduce the stale copied index without modifying any tensor.
            (output/'model.safetensors.index.json').write_bytes(
                (source/'model.safetensors.index.json').read_bytes())
            subprocess.run([sys.executable, str(Path(__file__).with_name('convert-fp8-attention.py')),
                            '--source', str(source), '--output', str(output), '--rebuild-index-only'],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(index_path.read_text()), index)
            self.assertEqual(hashlib.sha256((output/shard_name).read_bytes()).hexdigest(), before)
            repaired_manifest = json.loads((output/'fp8-attention-manifest.json').read_text())
            self.assertEqual(repaired_manifest['index']['sha256'],
                             hashlib.sha256(index_path.read_bytes()).hexdigest())

if __name__ == '__main__':
    unittest.main()
