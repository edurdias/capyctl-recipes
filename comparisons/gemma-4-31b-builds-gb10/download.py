#!/usr/bin/env python3
"""Download only pinned Gemma comparison inputs into the requested directory."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

CHECKPOINTS = {
    'mse': ('melcheikh/gemma-4-31B-it-qat-NVFP4-mse-Blackwell', 'c9e2f904a6ab66d7cd74d442b6bef95faf9d5393'),
    'dflash2': ('RedHatAI/gemma-4-31B-it-speculator.dflash2', 'b0cd01b4918aae6b6388272950c4f9851734674b'),
    'google': ('google/gemma-4-31B-it-qat-w4a16-ct', '52f3f65bc7a02d555763bc923bd1d9094898219d'),
    'redhat': ('RedHatAI/gemma-4-31B-it-NVFP4', '5df75ccad9b74546f5be758a2a68acc2e16bbc9b'),
    'nvidia': ('nvidia/Gemma-4-31B-IT-NVFP4', '4135a98a9b728a548947683219633b25682223ac'),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    manifest = {}
    for name, (repo, revision) in CHECKPOINTS.items():
        pin = api.model_info(repo, revision=revision).sha
        target = args.models / name
        snapshot_download(repo, revision=pin, local_dir=target, max_workers=4,
                          allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.model', '*.txt'])
        manifest[name] = {'checkpoint': repo, 'revision': pin,
                          'safetensors_bytes': sum(p.stat().st_size for p in target.glob('*.safetensors'))}
        args.models.mkdir(parents=True, exist_ok=True)
        (args.models / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'downloaded': name, **manifest[name]}), flush=True)
