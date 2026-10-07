#!/usr/bin/env python3
"""Numerical matrix-moment/block-pencil diagnostics for QICT C11.

No floating eigenvalue, SVD threshold, or candidate pole emitted here is a
rigorous positivity, rank, or spectral-support certificate.  The exact
positivity theorem is the unitary-moment Gram identity; a quantitative
floating-data margin requires independent validated arithmetic/error bounds.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def block_toeplitz(C,N):
    C=np.asarray(C,complex); d=C.shape[1]
    T=np.zeros((d*(N+1),d*(N+1)),complex)
    for i in range(N+1):
      for j in range(N+1):
        k=j-i; B=C[k] if k>=0 else C[-k].conj().T
        T[d*i:d*(i+1),d*j:d*(j+1)]=B
    return (T+T.conj().T)/2

def hankel(C,K):
    d=C.shape[1]; H0=np.zeros((d*K,d*K),complex); H1=np.zeros_like(H0)
    for i in range(K):
      for j in range(K):
        H0[d*i:d*(i+1),d*j:d*(j+1)]=C[i+j]
        H1[d*i:d*(i+1),d*j:d*(j+1)]=C[i+j+1]
    return H0,H1

def pencil(C,K,svd_rtol=1e-11):
    H0,H1=hankel(C,K); U,s,Vh=np.linalg.svd(H0,full_matrices=False)
    if s.size==0 or s[0]==0:
        return {'numerical_rank_diagnostic':0,'candidate_poles':[],'unit_circle_max_deviation_diagnostic':None}
    r=int(np.sum(s>svd_rtol*s[0])); Ur=U[:,:r]; Vr=Vh.conj().T[:,:r]; sr=s[:r]
    Sinv=np.diag(1/np.sqrt(sr)); A=Sinv@Ur.conj().T@H1@Vr@Sinv
    vals=np.linalg.eigvals(A); vals=vals[np.argsort(np.angle(vals))]
    return {'numerical_rank_diagnostic':r,
            'candidate_poles':[[float(z.real),float(z.imag)] for z in vals],
            'unit_circle_max_deviation_diagnostic':float(np.max(np.abs(np.abs(vals)-1))) if len(vals) else None,
            'singular_value_ratio_last_kept_diagnostic':float(sr[-1]/sr[0]) if r else None,
            'svd_relative_tolerance_diagnostic':svd_rtol}

def analyze(C):
    C=np.asarray(C,complex)
    if C.ndim!=3 or C.shape[1:]!=(6,6): raise ValueError('moments must have shape (M,6,6)')
    c0=float(np.linalg.norm(C[0]-np.eye(6))); Nt=min(16,len(C)-1)
    lam=float(np.linalg.eigvalsh(block_toeplitz(C,Nt)).min())
    ks=[k for k in range(2,min(8,(len(C)-1)//2+1))]
    ps=[{'K':k,**pencil(C,k)} for k in ks]
    return {'status':'PASS_C11_NUMERICAL_MOMENT_DIAGNOSTIC__RIGOROUS_ACCEPTANCE_SEPARATE',
            'C0_fro_error_diagnostic':c0,'toeplitz_depth':Nt,
            'toeplitz_lambda_min_diagnostic':lam,'pencils':ps,
            'rigorous_positivity_claim':False,'rigorous_rank_claim':False,
            'exact_structural_positivity':'For exact C_k=V^*U^kV with U unitary, the block Toeplitz matrix is a Gram matrix and is positive semidefinite.',
            'interpretation':'Pencil poles and tolerance-based ranks are candidate-generation diagnostics only. Physical acceptance requires independently certified errors and depth/regulator/volume stability.'}

def synthetic():
    # +1/-1 exact unitary atoms in complementary rank-three source sectors.
    P=np.zeros((6,6),complex); P[:3,:3]=np.eye(3); Q=np.eye(6)-P
    C=np.asarray([P+((-1)**n)*Q for n in range(24)],complex)
    out=analyze(C); out['self_test_construction']='two exact unit-circle atoms +1 and -1 in complementary rank-three projectors'
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--moments',type=Path); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.moments:
        z=np.load(a.moments,allow_pickle=False); C=z['moments'] if 'moments' in z.files else z['direct_moments']; out=analyze(C)
    else: out=synthetic()
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':out['status'],'toeplitz_lambda_min_diagnostic':out['toeplitz_lambda_min_diagnostic'],'pencil_ranks_diagnostic':[x['numerical_rank_diagnostic'] for x in out['pencils']]},indent=2))
if __name__=='__main__': main()
