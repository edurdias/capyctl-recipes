#!/usr/bin/env python3
"""CPU-only complete source/Quark byte and FP8-scale audit."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import torch
from safetensors import safe_open


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Reader:
    def __init__(self, root):
        self.root = root
        self.index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.files = {s: safe_open(str(root / s), framework='pt', device='cpu') for s in set(self.index.values())}
    def get(self, name):
        return self.files[self.index[name]].get_tensor(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--vllm-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    source, candidate = Reader(a.source), Reader(a.candidate)
    checked, rows = set(), []
    for name in sorted(source.index):
        original = source.get(name)
        expert = re.fullmatch(r'(.*\.experts)\.(gate_up_proj|down_proj)_(blocks|scales|bias)', name)
        if expert:
            prefix, proj, kind = expert.groups()
            suffix = {'blocks': 'weight', 'scales': 'weight_scale', 'bias': 'bias'}[kind]
            names = [f'{prefix}.{e}.{proj}.{suffix}' for e in range(original.shape[0])]
            restored = torch.stack([candidate.get(n) for n in names]).reshape(original.shape)
            source_sha, restored_sha = digest(original), digest(restored)
            assert original.dtype == restored.dtype and source_sha == restored_sha, name
            checked.update(names)
            rows.append({'name': name, 'kind': 'expert', 'bytes': original.nbytes, 'sha256': source_sha, 'candidate_sha256': restored_sha})
        elif re.fullmatch(r'.*\.self_attn\.[qkvo]_proj\.weight', name):
            packed = candidate.get(name)
            scale_name = name.removesuffix('.weight') + '.weight_scale'
            scale = candidate.get(scale_name)
            expected = (original.float().abs().amax(1) / 448).clamp_min(torch.finfo(torch.float32).tiny)
            assert packed.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32, name
            assert scale.shape == (original.shape[0],) and torch.equal(scale, expected), name
            quantized = (original.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            assert torch.equal(packed.view(torch.uint8), quantized.view(torch.uint8)), name
            checked.update([name, scale_name])
            rows.append({'name': name, 'kind': 'attention', 'shape': list(packed.shape), 'bytes': packed.nbytes + scale.nbytes, 'sha256': digest(packed), 'scale_sha256': digest(scale)})
        else:
            other = candidate.get(name)
            source_sha, candidate_sha = digest(original), digest(other)
            assert original.dtype == other.dtype and original.shape == other.shape and source_sha == candidate_sha, name
            checked.add(name)
            rows.append({'name': name, 'kind': 'unchanged', 'bytes': original.nbytes, 'sha256': source_sha, 'candidate_sha256': candidate_sha})
    assert checked == set(candidate.index), sorted(set(candidate.index) ^ checked)
    config = json.loads((a.candidate / 'config.json').read_text())['quantization_config']
    model_path = a.vllm_root / 'model_executor/models/gpt_oss.py'
    model_source = model_path.read_text()
    # Installed GPT-OSS names its GateLinear router explicitly.
    assert 'self.router = GateLinear' in model_source
    assert 'prefix=f"{prefix}.router"' in model_source
    router_names = [f'model.layers.{i}.mlp.router' for i in range(24)]
    assert all(any(fnmatch.fnmatchcase(n, g) for g in config['exclude']) for n in router_names)
    assert any(fnmatch.fnmatchcase('lm_head', g) for g in config['exclude'])
    quark_path = a.vllm_root / 'model_executor/layers/quantization/quark/quark.py'
    report = {'passed': True, 'device': 'cpu', 'source_tensors': len(source.index), 'candidate_tensors': len(checked), 'expert_tensors': sum(r['kind'] == 'expert' for r in rows), 'attention_tensors': sum(r['kind'] == 'attention' for r in rows), 'tensor_bytes': sum(r['bytes'] for r in rows), 'config': config, 'router_module_names': router_names, 'installed_source_hashes': {str(p.relative_to(a.vllm_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [model_path, quark_path]}, 'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()} for p in sorted(a.candidate.iterdir()) if p.is_file()}, 'tensors': rows}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'tensors', 'config', 'router_module_names')}))

if __name__ == '__main__':
    main()
