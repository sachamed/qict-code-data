#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

def toeplitz(C,N):
    d=C.shape[1]; T=np.zeros((d*(N+1),d*(N+1)),complex)
    for i in range(N+1):
      for j in range(N+1):
        k=j-i; T[d*i:d*(i+1),d*j:d*(j+1)]=C[k] if k>=0 else C[-k].conj().T
    return (T+T.conj().T)/2

def deterministic_measure(m=18,d=6):
    # Distinct unit-circle atoms with deterministic full-rank positive residues.
    a=np.arange(m); phases=0.137+2*np.pi*a/m; poles=np.exp(1j*phases)
    raw=[]
    for q in range(m):
        B=np.empty((d,d),complex)
        for i in range(d):
            for j in range(d):
                amp=(1.0 if i==j else 0.17/(1+abs(i-j))) + 0.025*(q+1)/(m+1)
                ang=0.071*(q+1)*(i+1)+0.113*(q+2)*(j+1)
                B[i,j]=amp*np.exp(1j*ang)
        R=B@B.conj().T + (0.15+0.01*q)*np.eye(d)
        raw.append(R)
    S=sum(raw)
    vals,vec=np.linalg.eigh(S); Sinv=(vec*(1/np.sqrt(vals)))@vec.conj().T
    residues=np.stack([Sinv@R@Sinv for R in raw])
    return poles,residues

def moments(poles,residues,depth):
    C=np.empty((depth+1,residues.shape[1],residues.shape[2]),complex)
    for n in range(depth+1): C[n]=np.einsum('a,aij->ij',poles**n,residues)
    return C

def reconstruct_from_trace_and_matrix_moments(C,m):
    c=np.trace(C,axis1=1,axis2=2)
    H0=np.array([[c[i+j] for j in range(m)] for i in range(m)],complex)
    H1=np.array([[c[i+j+1] for j in range(m)] for i in range(m)],complex)
    poles=eig(H1,H0,right=False)
    # Project tiny radial numerical drift back only for ordering diagnostics; residues use the raw reconstructed poles.
    V=np.array([[poles[a]**n for a in range(m)] for n in range(m)],complex)
    d=C.shape[1]; residues=np.empty((m,d,d),complex)
    for i in range(d):
      for j in range(d): residues[:,i,j]=np.linalg.solve(V,C[:m,i,j])
    return poles,residues,H0

def flat_extension():
    I=sp.I; r=6; atoms=[sp.Integer(1),sp.Integer(-1),I]
    W1=sp.diag(*[sp.Rational(1,2),sp.Rational(1,3),sp.Rational(1,4),sp.Rational(1,5),sp.Rational(1,6),sp.Rational(1,7)])
    W2=sp.diag(*[sp.Rational(1,3),sp.Rational(1,4),sp.Rational(1,5),sp.Rational(1,6),sp.Rational(1,7),sp.Rational(1,8)])
    W3=sp.eye(r)-W1-W2; weights=[W1,W2,W3]
    def Cn(n): return sp.simplify(sum(((q**n)*W for q,W in zip(atoms,weights)),sp.zeros(r)))
    def T(K): return sp.Matrix.vstack(*[sp.Matrix.hstack(*[Cn(j-i) for j in range(K+1)]) for i in range(K+1)])
    r2,r3=T(2).rank(),T(3).rank(); rec=True
    for n in range(-6,7): rec &= sp.simplify(Cn(n+3)-I*Cn(n+2)-Cn(n+1)+I*Cn(n))==sp.zeros(r)
    return {'toeplitz_rank_K2':int(r2),'toeplitz_rank_K3':int(r3),'flat_extension':bool(r2==r3),'recurrence_exact':bool(rec),'C0_identity':bool(Cn(0)==sp.eye(r))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',required=True); a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    m,d,depth=18,6,40
    poles,residues=deterministic_measure(m,d); C=moments(poles,residues,depth)
    T=toeplitz(C,16); toe_min=float(np.linalg.eigvalsh(T).min())
    rp,rr,H0=reconstruct_from_trace_and_matrix_moments(C,m)
    cost=np.abs(rp[:,None]-poles[None,:]); rows,cols=linear_sum_assignment(cost)
    pole_err=float(cost[rows,cols].max())
    # Compare residues after matching estimated-to-reference atom labels.
    res_err=max(float(np.linalg.norm(rr[r]-residues[c],'fro')) for r,c in zip(rows,cols))
    min_res_eig=min(float(np.linalg.eigvalsh((rr[r]+rr[r].conj().T)/2).min()) for r in rows)
    # Reconstruct moments from the inferred poles and residues.
    Crec=moments(rp,rr,depth)
    mom_err=float(max(np.linalg.norm(Crec[n]-C[n],'fro') for n in range(depth+1)))
    c0_err=float(np.linalg.norm(C[0]-np.eye(d),'fro'))
    radial=float(np.max(np.abs(np.abs(rp)-1)))
    hcond=float(np.linalg.cond(H0))
    exact=flat_extension()
    np.savez_compressed(out/'SPECTRAL_BENCHMARK.npz',poles=poles,residues=residues,moments=C,reconstructed_poles=rp,reconstructed_residues=rr)
    result={
      'deterministic_matrix_measure':{
        'atom_count':m,'matrix_dimension':d,'moment_depth':depth,'C0_identity_fro_error':c0_err,
        'toeplitz_depth':16,'toeplitz_lambda_min_numeric':toe_min,'trace_hankel_condition_number':hcond,
        'reconstructed_pole_max_abs_error':pole_err,'reconstructed_pole_max_radial_error':radial,
        'reconstructed_residue_max_fro_error':res_err,'reconstructed_residue_min_hermitian_eigenvalue':min_res_eig,
        'reconstructed_moment_max_fro_error':mom_err},
      'exact_flat_extension_benchmark':exact}
    (out/'SPECTRAL_METHOD_VALIDATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
