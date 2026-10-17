#!/usr/bin/env python3
"""Resolve byte-identical input references used by the scientific programs."""
from pathlib import Path
import hashlib,json,zipfile,os,tempfile

ROOT=Path(__file__).resolve().parents[2]

def digest(data):return hashlib.sha256(data).hexdigest()

def safe_path(name):
 p=ROOT/name
 if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError(name)
 return p

def read_reference(ref):
 if 'path' in ref:data=safe_path(ref['path']).read_bytes()
 else:
  with zipfile.ZipFile(safe_path(ref['archive'])) as z:data=z.read(ref['member'])
 if digest(data)!=ref['sha256']:raise ValueError('Input digest mismatch: '+str(ref))
 return data

def prepare():
 refs=json.loads((ROOT/'reproducibility/verification/INPUT_REFERENCES.json').read_text())
 created=0
 for name,ref in refs.items():
  target=safe_path(name)
  if target.exists():
   if digest(target.read_bytes())!=ref['sha256']:raise ValueError('Conflicting input: '+name)
   continue
  data=read_reference(ref);target.parent.mkdir(parents=True,exist_ok=True)
  fd,tmp=tempfile.mkstemp(prefix='.qict-input-',dir=target.parent)
  try:
   with os.fdopen(fd,'wb') as f:f.write(data)
   os.replace(tmp,target);created+=1
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
 return {'input_references_checked':len(refs),'runtime_paths_created':created}

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
