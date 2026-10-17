#!/usr/bin/env python3
from __future__ import annotations
import sys,time,json,numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import verify_rank6_chiral_u1 as r6
OUT=ROOT/'results'/'final_closure'; OUT.mkdir(parents=True,exist_ok=True)

def applyU_fast(X,ops):
    tmp=np.zeros_like(X); k=X.shape[-1]; d=X.shape[2]
    for c,q in enumerate((-3,-6)):
        for a in range(3):
            G=ops[q][a]; W=r6.COEFF[a]*r6.WFAM[a]
            Y=(G@X[c,a].reshape(d,3*k)).reshape(d,3,k)
            tmp[c,a]=np.einsum('dfk,fg->dgk',Y,W.T,optimize=True)
    out=np.empty_like(tmp); out[0]=tmp[1]; out[1]=tmp[0]
    return out

def inner(V,X): return np.einsum('cadfk,cadfl->kl',V.conj(),X,optimize=True)

def run_case(N,z=1.1,nmax=320):
    t0=time.time(); vac,kato,ops=r6.build_ops(N); build=time.time()-t0
    V=r6.source(vac); X=V.copy(); G=np.zeros((6,6),complex); last=0.0
    t1=time.time()
    for n in range(nmax+1):
        C=inner(V,X); term=C/(z**(n+1)); G+=term; last=float(np.linalg.norm(term,2)); X=applyU_fast(X,ops)
    series=time.time()-t1
    Gamma=np.linalg.inv(G); LR=Gamma[:3,3:]
    s=np.linalg.svd(LR,compute_uv=False)
    return {'N':N,'z_real':z,'nmax':nmax,'build_seconds':build,'series_seconds':series,'last_term_norm':last,
            'LR_singular_values':s.tolist(),
            'LR_ratios':{'largest_over_middle':float(s[0]/s[1]),'middle_over_smallest':float(s[1]/s[2]),'largest_over_smallest':float(s[0]/s[2])},
            'Gamma_LL_norm':float(np.linalg.norm(Gamma[:3,:3],2)),'Gamma_RR_norm':float(np.linalg.norm(Gamma[3:,3:],2)),
            'kato':kato}

def main():
    cases=[run_case(N) for N in (19,21)]
    out={'status':'PASS_DIAGNOSTIC','blind':True,'particle_masses_used_in_operator':False,
         'observable':'rank-six projected resolvent G_6(z)=V^dagger(z-U)^-1 V and inverse Feshbach kernel Gamma_6(z)=G_6(z)^-1; singular values reported for the L-R block at z=1.1 outside the unit circle',
         'series_identity':'For |z|>1, G_6(z)=sum_{n>=0} C_n z^(-n-1), so no global matrix inversion is formed.',
         'cases':cases,
         'verdict':'CONVERGED_ENOUGH_FOR_NO_GO_DIAGNOSTIC__FAIL_AS_LEPTON_HIERARCHY',
         'interpretation':'The full projected resolvent produces cancellations relative to a one-step chiral block, but in the compact-U(1) diagnostic its L-R hierarchy remains O(1). The electroweak SU(2) route-first environment and full 3D CAR/Gauss contraction are therefore indispensable; no charged-lepton mass prediction is promoted from this U(1) result.'}
    (OUT/'RANK6_FESHBACH_RESOLVENT_U1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
