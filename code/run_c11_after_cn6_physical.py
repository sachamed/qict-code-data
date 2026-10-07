#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, hashlib, numpy as np

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def run(cmd):
 p=subprocess.run(cmd,text=True,capture_output=True)
 if p.returncode!=0: raise RuntimeError(f'command failed {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}')
 return p.stdout

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--L',type=int,required=True); ap.add_argument('--results-dir',type=pathlib.Path,required=True); ap.add_argument('--error-certificate',type=pathlib.Path); a=ap.parse_args()
 rd=a.results_dir; m=rd/f'CN6_MOMENTS_L{a.L}.npz'
 if not m.exists(): raise SystemExit(f'missing {m}')
 z=np.load(m,allow_pickle=False); key='moments' if 'moments' in z.files else ('direct_moments' if 'direct_moments' in z.files else None)
 if key is None: raise SystemExit('moments key missing')
 C=np.asarray(z[key],complex)
 if C.ndim!=3 or C.shape[1:]!=(6,6): raise SystemExit('moments shape must be (M,6,6)')
 c0=float(np.linalg.norm(C[0]-np.eye(6)))
 if c0>1e-9: raise SystemExit(f'C0 is not I6: {c0}')
 here=pathlib.Path(__file__).resolve().parent; c11=here/'c11'
 diag=rd/f'C11_BLOCK_PENCIL_DIAGNOSTIC_L{a.L}.json'; supp=rd/f'C11_SUPPORT_COMPLEXITY_DIAGNOSTIC_L{a.L}.json'; pos=rd/f'C11_RIGOROUS_POSITIVITY_L{a.L}.json'
 run([sys.executable,str(c11/'c11_block_pencil.py'),'--moments',str(m),'--out',str(diag)])
 run([sys.executable,str(c11/'c11_support_complexity_certificate.py'),'--npz',str(m),'--out',str(supp),'--max-depth',str(min(16,len(C)-1))])
 cmd=[sys.executable,str(c11/'c11_error_certified_positivity.py'),'--moments',str(m),'--out',str(pos),'--depth',str(min(16,len(C)-1))]
 if a.error_certificate is not None: cmd += ['--error-certificate',str(a.error_certificate)]
 run(cmd)
 pobj=json.loads(pos.read_text())
 rigorous=bool(pobj.get('positive_certified',False))
 out={
   'status':'PASS_C11_RIGOROUS_POSITIVITY__SPECTRAL_CERT_STILL_REQUIRED' if rigorous else 'C11_RIGOROUS_ACCEPTANCE_BLOCKED',
   'L':a.L,'moments_file':m.name,'moments_sha256':sha256(m),'moment_count':len(C),'C0_fro_error':c0,
   'numerical_diagnostic':diag.name,'support_complexity_diagnostic':supp.name,'rigorous_positivity':pos.name,
   'rigorous_positivity_certified':rigorous,
   'physical_G2_promoted':False,
   'remaining_for_G2':'Validated spectral-support/edge (or exact flat-extension) certificate plus multivolume stability. Numerical block-pencil poles are diagnostics only.'
 }
 (rd/f'C11_GATE_L{a.L}.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
