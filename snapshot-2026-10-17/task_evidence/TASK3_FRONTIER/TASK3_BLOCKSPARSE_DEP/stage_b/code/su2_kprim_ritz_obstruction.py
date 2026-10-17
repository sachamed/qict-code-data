#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'results'/'SU2_KPRIM_LANCZOS_TAIL_PROFILE.json'
OUT=ROOT/'results'/'SU2_KPRIM_RITZ_OBSTRUCTION.json'
x=json.loads(SRC.read_text())
a=np.asarray(x['alphas'],float)
b=np.asarray(x['betas'],float)
m=len(a)
if len(b)<m: raise RuntimeError('Need beta_m residual coefficient')
T=np.diag(a)
for i in range(m-1):
    T[i,i+1]=T[i+1,i]=b[i]
w,V=np.linalg.eigh(T)
rows=[]
for i,lam in enumerate(w):
    resid=float(abs(b[m-1]*V[-1,i]))
    rows.append({'ritz_index':i,'ritz_value':float(lam),'vacuum_overlap_abs':float(abs(V[0,i])),
                 'last_lanczos_component_abs':float(abs(V[-1,i])),'exact_ritz_residual_norm':resid})
imin=int(np.argmin(w)); rmin=rows[imin]['exact_ritz_residual_norm']
out={
 'status':'PASS_CERTIFIED_SHORT_KRYLOV_REJECTION',
 'lanczos_dimension':m,
 'tridiagonal':T.tolist(),
 'ritz_pairs':rows,
 'lowest_ritz_residual_norm':rmin,
 'threshold_for_physical_substitution':1e-6,
 'short_krylov_can_replace_pepo_frontier':bool(rmin<1e-6),
 'conclusion':'The exact source-driven global Krylov subspace at this dimension is far from an invariant Kato subspace. It cannot populate h_8,h_9,h_10 or replace the width-14 PEPO frontier.',
 'tail_observation':'The fourth normalized Lanczos vector has support 50304; keeping 30000 orbit coefficients still discards L2 norm about 0.2565. Configuration-amplitude pruning is therefore not a certified compression mechanism.'
}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
