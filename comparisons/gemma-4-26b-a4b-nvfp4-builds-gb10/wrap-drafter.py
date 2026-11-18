#!/usr/bin/env python3
"""Create a metadata-only SGLang EAGLE3 wrapper; never rewrite pinned weights."""
import hashlib
import json
from pathlib import Path
from download import model_path

source = model_path('eagle3')
original = json.loads((source/'config.json').read_text())
config = {**original['transformer_layer_config'],
          **{k: v for k, v in original.items() if k not in ('transformer_layer_config', 'auto_map')}}
config.update(model_type='llama', architectures=['LlamaForCausalLMEagle3'],
              target_hidden_size=config['hidden_size'],
              eagle_config={k: original[k] for k in ('eagle_aux_hidden_state_layer_ids', 'norm_before_fc', 'norm_before_residual')})
wrapper = Path.home()/'drafters/gemma4-redhat-eagle3-sglang-226b2692'
wrapper.mkdir(parents=True, exist_ok=False)
for path in source.iterdir():
    if path.is_file() and path.name != 'config.json' and path.suffix in ('.safetensors', '.json', '.model', '.jinja', '.tiktoken'):
        (wrapper/path.name).hardlink_to(path)
(wrapper/'config.json').write_text(json.dumps(config, indent=2)+'\n')
provenance = {'source': str(source), 'source_config_sha256': hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
              'wrapper': str(wrapper), 'wrapper_config_sha256': hashlib.sha256((wrapper/'config.json').read_bytes()).hexdigest(),
              'changes': {'flatten': 'transformer_layer_config fields copied to top level; original outer fields retained',
                          'remove': ['auto_map', 'transformer_layer_config'],
                          'model_type': 'llama', 'architectures': ['LlamaForCausalLMEagle3'],
                          'target_hidden_size': config['hidden_size'], 'eagle_config': config['eagle_config']},
              'weights': 'same-filesystem hardlinks to immutable pinned snapshot; no tensor changes or payload copies',
              'support': 'installed sglang/srt/models/llama_eagle3.py supports reduced vocabulary d2t/t2d, draft_vocab_size, auxiliary layers and these norm settings'}
(wrapper/'wrapper-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print(json.dumps(provenance, indent=2))
