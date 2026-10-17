#!/usr/bin/env python3
"""Deterministic finite-dimensional test of the relative-transfer theorem.

This tests mathematical identities and perturbation enclosures, not the interacting
charged QICT contraction or physical masses. Only NumPy is needed.
"""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/RELATIVE_TRANSFER_STABILITY_CERTIFICATE.json'

def unitary(rng,n):
    q,r=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    return q*(np.diag(r).conj()/np.maximum(np.abs(np.diag(r)),1e-14))[None,:]

def isometry(a):
    ev,z=np.linalg.eigh(a.conj().T@a)
    if ev.min()<1e-10:raise ValueError('Source gram is singular')
    return a@((z/np.sqrt(ev)[None,:])@z.conj().T)

def norm(a):return float(np.linalg.norm(a,ord=2))

def main():
    rng=np.random.default_rng(20261016)
    d0=df=3;channels=6
    u0=unitary(rng,d0);uf=unitary(rng,df)
    rho=np.diag([.2,.3,.5]); sqrt_rho=np.diag(np.sqrt(np.diag(rho)))
    ops=rng.normal(size=(channels,df,d0))+1j*rng.normal(size=(channels,df,d0))
    emb=np.column_stack([(o@sqrt_rho).reshape(-1,order='F') for o in ops])
    v=isometry(emb)
    w=np.kron(u0.conj(),uf)
    h0=rng.normal(size=(d0,d0));h0=(h0+h0.T)/2
    hf=rng.normal(size=(df,df));hf=(hf+hf.T)/2
    eps=5e-4
    ev0,z0=np.linalg.eigh(h0);evf,zf=np.linalg.eigh(hf)
    ut0=(z0*np.exp(1j*eps*ev0)[None,:])@z0.conj().T@u0
    utf=(zf*np.exp(1j*eps*evf)[None,:])@zf.conj().T@uf
    vt=isometry(v+eps*(rng.normal(size=v.shape)+1j*rng.normal(size=v.shape)))
    wt=np.kron(ut0.conj(),utf)
    e0=norm(u0-ut0);ef=norm(uf-utf);ev=norm(v-vt)
    nmax=6
    c={n:v.conj().T@np.linalg.matrix_power(w,n)@v for n in range(-nmax,nmax+1)}
    cp={n:vt.conj().T@np.linalg.matrix_power(wt,n)@vt for n in range(-nmax,nmax+1)}
    toe=np.block([[c[j-i] for j in range(nmax+1)] for i in range(nmax+1)])
    min_eigen=float(np.linalg.eigvalsh((toe+toe.conj().T)/2).min())
    merror=max(norm(c[-n]-c[n].conj().T) for n in range(nmax+1))
    identity=norm(c[0]-np.eye(channels))
    bound_w=e0+ef
    errmax=max(norm(c[n]-cp[n]) for n in range(-nmax,nmax+1))
    boundmax=max(2*ev+abs(n)*bound_w for n in range(-nmax,nmax+1))
    bound_ratios=[]
    for n in range(-nmax,nmax+1):
        bound_ratios.append(norm(c[n]-cp[n])/(2*ev+abs(n)*bound_w))
    z=1.7+.4j
    g=v.conj().T@np.linalg.solve(z*np.eye(d0*df)-w,v)
    gt=vt.conj().T@np.linalg.solve(z*np.eye(d0*df)-wt,vt)
    remainder=norm(g-sum(z**(-n-1)*(v.conj().T@np.linalg.matrix_power(w,n)@v) for n in range(61)))
    gb=2*ev/(abs(z)-1)+bound_w/(abs(z)-1)**2
    gres=norm(g-gt)
    # nontrivial comparison against one-branch product on same 3x3 space
    product=u0.conj().T@uf
    one_sample=np.trace(product@product)/d0
    two_sample=np.trace(w@w)/(d0*df)
    report={
      'status':'PASS', 'scope':'Independent 3x3 neutral/charged unitary test, six normalized sources',
      'normalized_C0_operator_error':identity,
      'C_negative_adjoint_max_error':merror,
      'Toeplitz_min_eigenvalue':min_eigen,
      'input_error_u0':e0,'input_error_uf':ef,'input_error_V':ev,
      'moment_error_max':errmax,'moment_bound_max':boundmax,
      'largest_moment_error_to_bound_ratio':max(bound_ratios),
      'resolvent_neumann_remainder_n60':remainder,
      'resolvent_error':gres,'resolvent_error_bound':gb,
      'two_branch_vs_one_branch_second_trace_abs_diff':float(abs(two_sample-one_sample)),
      'physical_C10_closed':False,'physical_masses_determined':False}
    assert identity<1e-11 and merror<1e-11 and min_eigen>-1e-10
    assert max(bound_ratios)<1+1e-10
    assert remainder<1e-11 and gres<=gb*(1+1e-10)
    assert abs(two_sample-one_sample)>1e-3
    OUT.parent.mkdir(exist_ok=True,parents=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__':main()
