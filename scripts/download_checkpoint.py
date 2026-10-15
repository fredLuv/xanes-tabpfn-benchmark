"""Download the pinned checkpoint using existing provider access; never embed tokens."""
from pathlib import Path
import hashlib
import json
import shutil
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
pin=json.loads((ROOT/'configs/benchmark.json').read_text())['checkpoint']
source=Path(hf_hub_download(repo_id=pin['repo_id'],filename=pin['filename'],revision=pin['revision']))
if hashlib.sha256(source.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Checkpoint checksum mismatch')
target=ROOT/'weights'/pin['filename'];target.parent.mkdir(exist_ok=True)
if target.exists():
    if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest()!=pin['sha256']:
        raise ValueError('Existing target differs; preserve it and choose a clean checkpoint directory.')
else:shutil.copyfile(source,target)
print('Verified checkpoint:',target.name)
