import hashlib,json,os
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
root=Path.home()/'models/gpt-oss-120b'; root.mkdir(parents=True,exist_ok=True)
info=HfApi().model_info('openai/gpt-oss-120b',revision=revision,files_metadata=True)
records=[]
for s in info.siblings:
    if '/' in s.rfilename or not (s.rfilename.endswith(('.json','.safetensors','.jinja')) or s.rfilename in ('LICENSE','USAGE_POLICY')): continue
    p=Path(hf_hub_download('openai/gpt-oss-120b',s.rfilename,revision=revision,local_dir=root))
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    actual=h.hexdigest(); expected=s.lfs.sha256 if s.lfs else None
    if expected and actual!=expected: raise RuntimeError('LFS hash mismatch: '+s.rfilename)
    records.append(dict(file=s.rfilename,bytes=p.stat().st_size,sha256=actual,lfs_sha256=expected))
    print('VERIFIED',s.rfilename,flush=True)
(root/'download-verification.json').write_text(json.dumps(dict(model='openai/gpt-oss-120b',revision=revision,files=records),indent=2)+'\n')
print('DOWNLOAD COMPLETE',flush=True)
