# Shardio r13's checkpoint, at the revision Shardio pins; recorded in pins.json beside the others.
import json
import os

from huggingface_hub import snapshot_download

REPO, SHA = "RadixArk/Qwen3.8-27B-NVFP4", "319f741cce68d7914884900c138a1fbb70a42f30"
base = os.path.expanduser("~/q38-requant/ckpts")
d = os.path.join(base, REPO.replace("/", "--") + "@" + SHA)
snapshot_download(REPO, revision=SHA, local_dir=d)
p = os.path.join(base, "pins.json")
pins = json.load(open(p))
pins[REPO] = {"revision": SHA, "path": d}
json.dump(pins, open(p, "w"), indent=2)
print("done", REPO, SHA, flush=True)
