#!/usr/bin/env python3
from __future__ import annotations
import json, math, time, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import su2_charged_openline_seed as ch
from fast_su2_cg import cg_fast
ch.cg=cg_fast
ROOT=HERE.parent; OUT=ROOT/'results'/'SU2_CHARGED_KPRIM_LANCZOS.json'

def inner(a,b):
    if len(a)>len(b): a,b=b,a
    return sum(np.conjugate(v)*b.get(k,0j) for k,v in a.items())
def norm(a): return math.sqrt(max(0.,float(inner(a,a).real)))
def prune(a,tol=3e-13): return {k:v for k,v in a.items() if abs(v)>tol}
def axpy(y,a,x):
    for k,v in x.items():
        z=y.get(k,0j)+a*v
        if abs(z)<2e-13: y.pop(k,None)
        else:y[k]=z

def HB(v,j2max):
    out={}; calls=0
    for st,amp in v.items():
        if abs(amp)<1e-15:continue
        for pid in range(24):
            nb=ch.neighbors(st,pid,1,j2max); calls+=1
            for q,z in nb.items(): out[q]=out.get(q,0j)+amp*z
    return prune(out),calls

def KE_state(st):
    return float(sum(n*(n+2) for n in st.links))
def Kmat(v,j2max):
    hb,calls=HB(v,j2max)
    out={k:KE_state(k)*a for k,a in v.items()}
    axpy(out,-16.0,hb)
    return prune(out),calls

def lanczos(maxit=3):
    s=ch.seed((2,1,0)); qprev={}; q={s:1+0j}; beta_prev=0.
    alphas=[]; betas=[]; supports=[]; calls=[]; vecs=[]
    for it in range(maxit):
        vecs.append(q); supports.append(len(q))
        z,c=Kmat(q,j2max=min(2+it,5)); calls.append(c)
        alpha=inner(q,z); alphas.append(float(alpha.real))
        axpy(z,-alpha,q)
        if qprev: axpy(z,-beta_prev,qprev)
        # full reorthogonalization to prior vectors to control dictionary roundoff
        for old in vecs[:-1]:
            ov=inner(old,z)
            if abs(ov)>1e-13: axpy(z,-ov,old)
        beta=norm(z)
        if it<maxit-1:
            betas.append(beta)
            qprev,q=q,{k:v/beta for k,v in z.items()}
            beta_prev=beta
    # Ritz eigenvalues of the projected tridiagonal at every depth
    ritz=[]
    for m in range(1,maxit+1):
        T=np.diag(alphas[:m])
        for i in range(m-1): T[i,i+1]=T[i+1,i]=betas[i]
        ev=np.linalg.eigvalsh(T)
        ritz.append({'dim':m,'lowest_no_constant':float(ev[0]),'second_no_constant':float(ev[1]) if m>1 else None,'ritz_gap':float(ev[1]-ev[0]) if m>1 else None})
    orth=0.
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            target=1. if i==j else 0.
            orth=max(orth,abs(inner(vecs[i],vecs[j])-target))
    return {'alphas':alphas,'betas':betas,'supports':supports,'plaquette_calls':calls,'ritz':ritz,'orthogonality_residual':float(orth),'seed_electric_cost':KE_state(s)}

def main():
    t=time.time(); res=lanczos(2)
    checks={
      'seed_electric_cost_9':abs(res['seed_electric_cost']-9)<1e-12,
      'finite_coefficients':all(np.isfinite(res['alphas']+res['betas'])),
      'positive_beta':all(x>0 for x in res['betas']),
      'orthonormal_krylov':res['orthogonality_residual']<2e-11,
      'ritz_gap_positive_at_d2':res['ritz'][-1]['ritz_gap'] is not None and res['ritz'][-1]['ritz_gap']>1e-8,
    }
    out={'status':'PASS_SU2_CHARGED_KPRIM_LANCZOS_D2' if all(checks.values()) else 'FAIL',
         'operator':'Kprim_SU2 minus additive 32*Np constant = sum_l n_l(n_l+2) - 16*H_B',
         'normalization':'n=2j, 4C2=n(n+2), K_B=32*Np-16*sum_p chi_{1/2}(W_p)',
         'result':res,'checks':checks,'seconds':time.time()-t,
         'scope':'Source-Krylov/Ritz construction in the exact charged Gauss spin-network sector. A positive low-dimensional Ritz gap is a finite-Krylov diagnostic, not yet a proof of the full PEPO Kato spectral gap.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
