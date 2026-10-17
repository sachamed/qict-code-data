#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

rng=np.random.default_rng(20261005)

def unitary(n):
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    q,r=np.linalg.qr(a)
    ph=np.diag(r); ph=np.where(np.abs(ph)>0, ph/np.abs(ph), 1)
    return q@np.diag(np.conj(ph))

def pos(n):
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    return a@a.conj().T + np.eye(n)

def sqrtm_pos(a):
    w,v=np.linalg.eigh(a)
    return (v*np.sqrt(w))@v.conj().T

def jarlskog(V):
    return float(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    # Dirac mass-block covariance under independent left/right unitary family changes.
    M=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    ZL=pos(3); ZR=pos(3)
    R=sqrtm_pos(ZL)@M@sqrtm_pos(ZR)
    UL,UR=unitary(3),unitary(3)
    # Transform the already residue-normalized block as a basis change.
    Rp=UL.conj().T@R@UR
    s=np.linalg.svd(R,compute_uv=False); sp=np.linalg.svd(Rp,compute_uv=False)
    dirac_cov=float(np.max(np.abs(np.sort(s)-np.sort(sp))))

    # CKM/PMNS algebra and rephasing-invariant Jarlskog test.
    Uu,Ud=unitary(3),unitary(3); V=Uu.conj().T@Ud
    unitarity=float(np.linalg.norm(V.conj().T@V-np.eye(3),2))
    pL=np.diag(np.exp(1j*rng.uniform(-np.pi,np.pi,3)))
    pR=np.diag(np.exp(1j*rng.uniform(-np.pi,np.pi,3)))
    Vp=pL@V@pR
    moduli=float(np.max(np.abs(np.abs(Vp)**2-np.abs(V)**2)))
    jdiff=abs(jarlskog(Vp)-jarlskog(V))

    # Majorana/Nambu congruence preserves Takagi singular values.
    B=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); MR=(B+B.T)/2
    D=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    MN=np.block([[np.zeros((3,3),complex),D],[D.T,MR]])
    W=unitary(6); MNp=W.T@MN@W
    sn=np.linalg.svd(MN,compute_uv=False); snp=np.linalg.svd(MNp,compute_uv=False)
    takagi_cov=float(np.max(np.abs(np.sort(sn)-np.sort(snp))))
    symmetry=float(np.linalg.norm(MN-MN.T,2))

    # Color-singlet current commutant test with a deterministic SU(3) unitary.
    lam3=np.diag([1.,-1.,0.]); lam8=np.diag([1.,1.,-2.])/np.sqrt(3)
    U3=expm(1j*(0.31*lam3+0.17*lam8)/2)
    Gamma=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    current=np.kron(np.eye(3),Gamma)
    gauge=np.kron(U3,np.eye(3))
    color_comm=float(np.linalg.norm(gauge@current-current@gauge,2))

    out={
      'classification':'FERMION_OBSERVABLE_COVARIANCE_VALIDATION',
      'dirac_mass_singular_value_basis_covariance_residual':dirac_cov,
      'mixing_unitarity_residual':unitarity,
      'mixing_modulus_rephasing_residual':moduli,
      'jarlskog_rephasing_residual':jdiff,
      'neutral_nambu_symmetry_residual':symmetry,
      'takagi_singular_value_congruence_residual':takagi_cov,
      'color_singlet_weak_current_commutator_residual':color_comm,
      'scope':'deterministic algebraic validation of the observable maps; physical quark and neutrino values are assigned by the common interacting contraction'
    }
    tol=5e-13
    if max(dirac_cov,unitarity,moduli,jdiff,symmetry,takagi_cov,color_comm)>tol:
        raise SystemExit(out)
    Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
