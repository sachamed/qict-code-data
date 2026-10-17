#!/usr/bin/env python3
from pathlib import Path
import argparse,pickle,json,hashlib,math,sys

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent); a=ap.parse_args()
 m=json.loads((a.root/'STEP4_STRICT_CHECKPOINT_MANIFEST.json').read_text())
 p=a.root/m['checkpoint_file']
 ok=sha(p)==m['checkpoint_sha256']
 o=pickle.load(open(p,'rb')); d=o['states']
 support=len(d); norm2=sum(abs(v)**2 for v in d.values())
 ok &= support==m['support'] and abs(norm2-m['norm2'])<1e-15
 out={'status':'PASS_STEP4_STRICT_CHECKPOINT_VERIFY' if ok else 'FAIL_STEP4_STRICT_CHECKPOINT_VERIFY','sha256':sha(p),'support':support,'norm2':norm2,'certified_l2_bound':o['certified_l2_bound'],'scope':m['scope']}
 print(json.dumps(out,indent=2,sort_keys=True)); sys.exit(0 if ok else 2)
if __name__=='__main__': main()
