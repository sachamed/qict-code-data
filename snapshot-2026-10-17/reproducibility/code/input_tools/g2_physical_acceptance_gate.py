#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pathlib,hashlib
VOLUMES=(24,32,48,64,96,128)

def load(p): return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--results-root',type=pathlib.Path,required=True); ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
 rows=[]; ok=True
 for L in VOLUMES:
  rd=a.results_root/f'L{L}'; req={
   'cn6':rd/f'CN6_MATRIX_FREE_CERTIFICATE_L{L}.json',
   'c11':rd/f'C11_RIGOROUS_POSITIVITY_L{L}.json',
   'spectral':rd/f'SPECTRAL_SUPPORT_CERTIFICATE_L{L}.json',
   'g2':rd/f'G2_VOLUME_RESULT_L{L}.json'}
  miss=[k for k,p in req.items() if not p.exists()]
  if miss: rows.append({'L':L,'status':'BLOCKED_MISSING','missing':miss}); ok=False; continue
  c=load(req['cn6']); p=load(req['c11']); s=load(req['spectral']); g=load(req['g2'])
  checks={
   'cn6':c.get('status')=='PASS_MATRIX_FREE_CN6_STREAM',
   'c11':p.get('status')=='PASS_ERROR_CERTIFIED_C11_POSITIVITY',
   'spectral':s.get('status') in ('PASS_RIGOROUS_SPECTRAL_SUPPORT','PASS_RIGOROUS_SPECTRAL_EDGE','PASS_EXACT_FLAT_EXTENSION'),
   'g2':g.get('status')=='PASS_PHYSICAL_G2_VOLUME'}
  good=all(checks.values()); ok &= good; rows.append({'L':L,'status':'PASS_VOLUME_G2_GATE' if good else 'FAIL_VOLUME_G2_GATE','checks':checks})
 mv=a.results_root/'MULTIVOLUME_G2_LIMIT_CERTIFICATE.json'
 mvok=False
 if mv.exists(): mvok=load(mv).get('status')=='PASS_PHYSICAL_MULTIVOLUME_G2_LIMIT'
 ok &= mvok
 out={'status':'PASS_PHYSICAL_G2_ACCEPTANCE' if ok else 'G2_PHYSICAL_ACCEPTANCE_BLOCKED','volumes':rows,'multivolume_limit_certificate_present':mv.exists(),'multivolume_limit_pass':mvok,'physical_G2_promoted':bool(ok)}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
 raise SystemExit(0 if ok else 3)
if __name__=='__main__': main()
