# Converting FrogNano-4B-2609 to MLX 4-bit

How the checkpoint this recipe serves,
[`capyctl/FrogNano-4B-2609-MLX-4bit`](https://huggingface.co/capyctl/FrogNano-4B-2609-MLX-4bit),
was made from
[`microsoft/FrogNano-4B-2609`](https://huggingface.co/microsoft/FrogNano-4B-2609)
at `b90468c11a913c1916b4542b4b7a530ec42024a4`. You do not need to run this to
use the recipe; it is here so the checkpoint can be rebuilt and checked.

`convert.py` run on that revision with the versions below reproduces the
published `model.safetensors`, `config.json`, tokenizer files and
`generation_config.json` byte for byte (checked on 2026-10-03).

## Why it is needed

TensorFold 0.6.3's CUDA engine for Qwen dense models reads MLX affine weights
only, and it refuses a tied output layer:

```text
the tied embedding head is not supported by this packed Qwen decoder
```

FrogNano is BF16 with `tie_word_embeddings: true`. Only 4-bit with groups of
64 is fast on this GPU: TensorFold re-tiles that format for its fast matmul and
leaves every other format on a generic path. An 8-bit conversion loaded and
answered correctly but decoded at 7.6 tokens/s.

## Steps

1. A venv with mlx-lm 0.32.0 and mlx 0.32.3. The CPU backend is enough; the
   conversion takes about 50 s and 4 GB of RAM.

   ```bash
   uv venv -p 3.12 ~/mlx-lm-venv
   uv pip install -p ~/mlx-lm-venv "mlx[cpu]==0.32.3" mlx-lm==0.32.0
   ```

2. The source checkpoint:

   ```bash
   hf download microsoft/FrogNano-4B-2609 \
     --revision b90468c11a913c1916b4542b4b7a530ec42024a4 \
     --local-dir ~/models/FrogNano-4B-2609
   ```

3. Convert:

   ```bash
   ~/mlx-lm-venv/bin/python convert.py ~/models/FrogNano-4B-2609 ~/models/FrogNano-4B-2609-MLX-4bit
   ```

   ```text
   staged untied lm_head (248320, 2560) mlx.core.bfloat16
   [INFO] Loading
   [INFO] Using dtype: bfloat16
   [INFO] Quantizing
   [INFO] Quantized model with 4.502 bits per weight.
   wrote /home/me/models/FrogNano-4B-2609-MLX-4bit
   ```

## What `convert.py` does

1. **Unties the output layer.** In a temporary directory it links the source
   files, writes `lm_head.weight` as an exact copy of
   `model.language_model.embed_tokens.weight` (248320 x 2560, BF16), and sets
   `tie_word_embeddings: false` at the top level of `config.json` and in
   `text_config`. The quantized `lm_head` adds about 0.7 GB over a tied head.
2. **Quantizes** with `mlx_lm convert -q --q-bits 4 --q-group-size 64`.
   mlx-lm's `qwen3_5` loader drops the vision tower (297 tensors and
   `vision_config`) and the MTP layer (15 tensors), so the result is text only.
3. **Restores the original tokenizer files.** With transformers 5.x, mlx-lm
   rewrites `tokenizer.json` and `tokenizer_config.json` and does not copy
   `vocab.json` or `merges.txt`. The script copies the four originals back.
   `chat_template.jinja` comes through unchanged.
4. **Adds `generation_config.json`** with `eos_token_id: [248046, 248044]`
   (`<|im_end|>`, `<|endoftext|>`). The source names only 248044, and
   TensorFold reads end-of-sequence tokens only from `config.json` and
   `generation_config.json`; without this file replies do not stop at the end
   of a turn.

The `README.md` the converter writes is the source model card; the published
repository replaces it with its own card.
