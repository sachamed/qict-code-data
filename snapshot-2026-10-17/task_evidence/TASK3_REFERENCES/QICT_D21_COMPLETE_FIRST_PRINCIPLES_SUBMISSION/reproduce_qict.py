#!/usr/bin/env python3
"""Reproduce the active theorem and extension certificates for QICT-D21."""
from __future__ import annotations
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCRIPTS=[
 ROOT/'code/final_closure/verify_kato_vacuum.py',
 ROOT/'code/final_closure/verify_primary_feshbach.py',
 ROOT/'code/final_closure/verify_family_intertwiner_no_go.py',
 ROOT/'code/final_closure/verify_connected_car_trace_log.py',
 ROOT/'code/final_closure/verify_nonperturbative_planck_closure.py',
 ROOT/'code/final_closure/verify_uv_kernel_closure.py',
 ROOT/'code/microscopic_closure/verify_rank6_tensor_stageA.py',
 ROOT/'code/microscopic_closure/verify_full_sm_gauge_group.py',
 ROOT/'code/microscopic_closure/verify_physical_mixing_closure.py',
 ROOT/'code/microscopic_closure/verify_rg_alpha_closure.py',
 ROOT/'code/microscopic_closure/verify_singlet_scalar_lz.py',
 ROOT/'code/microscopic_closure/verify_global_sm_extension.py',
]
def main():
 failures=[]
 for s in SCRIPTS:
  print('[RUN]',s.relative_to(ROOT),flush=True)
  p=subprocess.run([sys.executable,str(s)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
  if p.returncode:
   failures.append(str(s.relative_to(ROOT))); print(p.stdout[-5000:]); print('[FAIL]')
  else:
   print('\n'.join(p.stdout.splitlines()[-5:])); print('[PASS]')
 if failures:
  print('REPRODUCTION: FAIL'); [print(' -',x) for x in failures]; return 1
 print(f'REPRODUCTION: {len(SCRIPTS)}/{len(SCRIPTS)} PASS')
 print('Scope: exact identities, reduced hierarchy, rank-six tensor representation, global gauge group, flavor/RG/dark-sector claim-boundary audits.')
 return 0
if __name__=='__main__': raise SystemExit(main())
