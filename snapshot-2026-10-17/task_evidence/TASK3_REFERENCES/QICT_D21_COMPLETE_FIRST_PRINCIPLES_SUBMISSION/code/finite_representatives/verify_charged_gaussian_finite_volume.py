#!/usr/bin/env python3
"""Finite-volume contraction benchmark for the fully specified Gaussian C7 electron sector.

A periodic 3x3x3 BCC coordinate cell is built explicitly in position space from the
eight fixed Weyl hopping matrices.  The complete 12 L^3 matrix-Julia update is then
constructed from the published electron transfer matrix.  Momentum-zero source moments
are evaluated by repeated application of this finite-volume operator and the two Floquet
roots in each family singular channel are reconstructed from the moments.

This verifies a genuine finite-grid contraction of the Gaussian C7 model.  It is not the
full interacting charged correlator, which additionally requires the local receiver/Clebsch
tensor and controlled gauge-representation convergence.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True); DATA=ROOT/'data'
LBOX=3
A={
 ( 1, 1, 1): np.array([[0,-1+1j],[0,1-1j]],complex)/4,
 ( 1, 1,-1): np.array([[1+1j,0],[-1-1j,0]],complex)/4,
 ( 1,-1, 1): np.array([[0,-1-1j],[0,1+1j]],complex)/4,
 ( 1,-1,-1): np.array([[1-1j,0],[-1+1j,0]],complex)/4,
 (-1, 1, 1): np.array([[0,1+1j],[0,1+1j]],complex)/4,
 (-1, 1,-1): np.array([[1-1j,0],[1-1j,0]],complex)/4,
 (-1,-1, 1): np.array([[0,1-1j],[0,1-1j]],complex)/4,
 (-1,-1,-1): np.array([[1+1j,0],[1+1j,0]],complex)/4,
}

def site_index(x,y,z,L): return (x*L+y)*L+z

def build_aw(L):
    Ns=L**3; AW=np.zeros((2*Ns,2*Ns),complex)
    for x in range(L):
      for y in range(L):
       for z in range(L):
        src=site_index(x,y,z,L)
        for h,M in A.items():
          dst=site_index((x+h[0])%L,(y+h[1])%L,(z+h[2])%L,L)
          AW[2*dst:2*dst+2,2*src:2*src+2]+=M
    return AW

def sqrt_psd(M):
    H=(M+M.conj().T)/2
    w,V=np.linalg.eigh(H)
    if w.min() < -1e-12: raise RuntimeError(f'negative defect eigenvalue {w.min()}')
    return (V*np.sqrt(np.clip(w,0,None)))@V.conj().T

def load_Te():
    T=np.zeros((3,3),complex)
    for row in csv.DictReader(open(DATA/'family_transfer_matrices.csv')):
        if row['sector']=='e':
            T[int(row['i']),int(row['j'])]=float(row['real'])+1j*float(row['imag'])
    return T

def full_gaussian_update(L,Te):
    AW=build_aw(L); Nsp=AW.shape[0]; Isp=np.eye(Nsp,dtype=complex)
    DL=sqrt_psd(np.eye(3,dtype=complex)-Te@Te.conj().T); DR=sqrt_psd(np.eye(3,dtype=complex)-Te.conj().T@Te)
    U=np.block([
      [np.kron(AW,DL), 1j*np.kron(Isp,Te)],
      [1j*np.kron(Isp,Te.conj().T), np.kron(AW.conj().T,DR)],
    ])
    return AW,U

def q0_source(L,family_vec,spin=0,julia='right'):
    Ns=L**3; Nsp=2*Ns; block=Nsp*3
    v=np.zeros(2*block,complex)
    amp=1/math.sqrt(Ns)
    off=block if julia=='right' else 0
    for site in range(Ns):
        ext=2*site+spin
        v[off+3*ext:off+3*ext+3]=amp*family_vec
    return v/np.linalg.norm(v)

def two_root_reconstruction(mom):
    c=np.asarray(mom,complex)
    M=np.array([[c[1],c[0]],[c[2],c[1]]],complex)
    rhs=-np.array([c[2],c[3]],complex)
    a,b=np.linalg.solve(M,rhs)
    return np.roots([1,a,b])

def main():
    Te=load_Te(); Lm,s,Vh=np.linalg.svd(Te); o=np.argsort(s); s=s[o]; R=Vh.conj().T[:,o]
    theta=np.arcsin(s)
    AW,U=full_gaussian_update(LBOX,Te)
    aw_res=float(np.linalg.norm(AW.conj().T@AW-np.eye(AW.shape[0]),'fro'))
    u_res=float(np.linalg.norm(U.conj().T@U-np.eye(U.shape[0]),'fro'))
    rows=[]; worst=0.0; moment_norm=0.0
    for a in range(3):
        v=q0_source(LBOX,R[:,a],spin=0,julia='right')
        psi=v.copy(); mom=[]
        for n in range(8):
            mom.append(np.vdot(v,psi))
            psi=U@psi
        roots=two_root_reconstruction(mom)
        phases=np.sort(np.abs(np.angle(roots)))
        rec=float(phases.mean())
        err=abs(rec-theta[a]); worst=max(worst,err)
        # exact q=0 moment expected for a right Julia basis vector: cos(n theta)
        mn=max(abs(mom[n]-math.cos(n*theta[a])) for n in range(len(mom)))
        moment_norm=max(moment_norm,float(mn))
        for j,r in enumerate(roots,1):
            rows.append({'L':LBOX,'branch':a+1,'root':j,'root_real':float(r.real),'root_imag':float(r.imag),
                         'root_radius':float(abs(r)),'abs_phase':float(abs(np.angle(r))),
                         'target_theta':float(theta[a]),'phase_error':float(abs(abs(np.angle(r))-theta[a]))})
    with open(DATA/'charged_gaussian_finite_volume_roots.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cert={
      'status':'PASS' if aw_res<1e-12 and u_res<1e-11 and worst<1e-11 and moment_norm<1e-11 else 'FAIL',
      'calculation':'complete finite-volume Gaussian C7 electron-sector contraction',
      'periodic_BCC_coordinate_size':LBOX,
      'spatial_sites':LBOX**3,
      'full_update_dimension':int(U.shape[0]),
      'weyl_stream_unitarity_fro':aw_res,
      'full_matrix_julia_unitarity_fro':u_res,
      'target_rest_phases':[float(x) for x in theta],
      'max_reconstructed_phase_error':float(worst),
      'max_moment_error_against_cos_n_theta':float(moment_norm),
      'interpretation':'A stable Gaussian Floquet pole is recovered by an explicit finite-grid contraction. This benchmark does not contain the interacting receiver/gauge collision and is not an electron pole-mass prediction.'
    }
    (CERT/'charged_gaussian_finite_volume_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps(cert,indent=2))
    raise SystemExit(0 if cert['status']=='PASS' else 1)
if __name__=='__main__': main()
