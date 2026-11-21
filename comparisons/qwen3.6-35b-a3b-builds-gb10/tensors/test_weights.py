#!/usr/bin/env python3
"""CPU characterization of the actual safetensors/ModelOpt reconstruction path."""
import tempfile
from pathlib import Path
import unittest
import torch
from safetensors.torch import save_file
from weights import Reader, LEVELS, family, fused_global_scale, search

class WeightFormats(unittest.TestCase):
    def test_known_nibbles_and_fp8(self):
        with tempfile.TemporaryDirectory() as directory:
            packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
            save_file({"a.weight": packed, "a.weight_scale": torch.tensor([[2.]], dtype=torch.float8_e4m3fn),
                       "a.weight_scale_2": torch.tensor(3.),
                       "b.weight": torch.tensor([[1., -2.]], dtype=torch.float8_e4m3fn),
                       "b.weight_scale": torch.tensor(.5)}, str(Path(directory)/"model.safetensors"))
            reader = Reader(directory, "cpu")
            expected = torch.tensor([LEVELS + [-x for x in LEVELS]]) * 6
            torch.testing.assert_close(reader.dequantize("a"), expected)
            torch.testing.assert_close(reader.dequantize("b"), torch.tensor([[.5, -1.]]))

    def test_searched_modelopt_pack_matches_error(self):
        torch.manual_seed(11)
        original = torch.randn(4, 64)
        original[0, :16] = 0
        packed, scales, global_scale, error = search(original, chunk=3)
        with tempfile.TemporaryDirectory() as directory:
            save_file({"a.weight": packed, "a.weight_scale": scales, "a.weight_scale_2": global_scale},
                      str(Path(directory)/"model.safetensors"))
            restored = Reader(directory, "cpu").dequantize("a")
        actual = ((restored-original).double().square().sum()/original.double().square().sum()).sqrt().item()
        self.assertAlmostEqual(actual, error, places=6)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue((scales.float() > 0).all())

    def test_shared_fused_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            values = {f"x.self_attn.{p}_proj.weight": torch.full((2, 16), v)
                      for p,v in [("q", 1.), ("k", 2.), ("v", 3.)]}
            values.update({f"x.linear_attn.{p}.weight": torch.full((2, 16), v)
                           for p,v in [("in_proj_qkv", 4.), ("in_proj_z", 8.)]})
            save_file(values, str(Path(directory)/"model.safetensors"))
            reader, cache = Reader(directory, "cpu"), {}
            for p in ["q", "k", "v"]:
                self.assertEqual(fused_global_scale(f"x.self_attn.{p}_proj.weight", reader, cache).item(),
                                 torch.tensor(3/(6*448)).item())
            for p in ["in_proj_qkv", "in_proj_z"]:
                self.assertEqual(fused_global_scale(f"x.linear_attn.{p}.weight", reader, cache).item(),
                                 torch.tensor(8/(6*448)).item())

    def test_family_protection(self):
        self.assertEqual(family("mtp.layers.0.self_attn.q_proj.weight"), "mtp")
        self.assertEqual(family("model.language_model.layers.0.linear_attn.in_proj_a.weight"), "preserved")
        self.assertEqual(family("model.language_model.layers.0.self_attn.q_proj.weight"), "attention")
        self.assertEqual(family("model.language_model.layers.0.linear_attn.in_proj_z.weight"), "deltanet")

if __name__ == "__main__":
    unittest.main()
