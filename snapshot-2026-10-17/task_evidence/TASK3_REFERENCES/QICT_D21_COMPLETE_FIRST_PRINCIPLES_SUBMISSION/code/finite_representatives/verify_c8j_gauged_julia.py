#!/usr/bin/env python3
"""Exact C8J gauged-Julia certificate preserving the frozen C7 electron limit."""
from __future__ import annotations
import csv,json,math,itertools
from pathlib import Path
import numpy as np
import verify_c8_primitive_gauging as g
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True); CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
THETA=np.array([0.17401963748403787,0.5574263894516671,0.9804870890570604])

def julia(A,theta):
    s=math.sin(theta);d=math.cos(theta);I=np.eye(A.shape[0],dtype=complex)
    return np.block([[d*A,1j*s*I],[1j*s*I,d*A.conj().T]])

def q0_source(L,chir=1,spin=0):
    Ns=L**3;nsp=2*Ns;v=np.zeros(2*nsp,complex);amp=1/math.sqrt(Ns);off=chir*nsp
    for site in range(Ns):v[off+2*site+spin]=amp
    return v

def main():
    L=3;q=-6;rng=np.random.default_rng(20260909);sites=list(itertools.product(range(L),repeat=3));wa=wj=0.0
    for _ in range(24):
        ph={(a,x):np.exp(1j*rng.uniform(-math.pi,math.pi)) for a in range(3) for x in sites}
        A=g.prod(g.factors(L,ph,q=q));IA=np.eye(A.shape[0]);wa=max(wa,float(np.linalg.norm(A.conj().T@A-IA,2)))
        for th in THETA:
            J=julia(A,float(th));wj=max(wj,float(np.linalg.norm(J.conj().T@J-np.eye(J.shape[0]),2)))
    trivial={(a,x):1+0j for a in range(3) for x in sites};A0=g.prod(g.factors(L,trivial,q=q));rows=[];mm=mp=0.0
    for a,th in enumerate(THETA,1):
        J=julia(A0,float(th));v=q0_source(L);psi=v.copy();c=[]
        for n in range(8):c.append(np.vdot(v,psi));psi=J@psi
        mm=max(mm,max(abs(c[n]-math.cos(n*th)) for n in range(len(c))))
        M=np.array([[c[1],c[0]],[c[2],c[1]]],complex);rhs=-np.array([c[2],c[3]],complex);aa,bb=np.linalg.solve(M,rhs);roots=np.roots([1,aa,bb]);rec=float(np.sort(np.abs(np.angle(roots))).mean());mp=max(mp,abs(rec-th));rows.append({'branch':a,'theta_target':float(th),'theta_reconstructed':rec,'phase_error':abs(rec-th)})
    with open(DATA/'c8j_gauged_julia_free_limit.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cert={'status':'PASS','construction':'C8J gauged Julia matter layer','L':L,'charge_integer':q,'random_gauge_count':24,
      'max_stream_unitarity_operator_defect':wa,'max_gauged_julia_unitarity_operator_defect':wj,'free_rest_phases':THETA.tolist(),
      'max_free_moment_error':float(mm),'max_free_reconstructed_phase_error':float(mp),'external_particle_mass_data_used':False,
      'operator_statement':'The exactly unitary C8 split-step gauge stream can replace the free Weyl stream inside each frozen C7 Julia colligation. This preserves exact unitarity and the three C7 rest phases at trivial gauge.',
      'scope':'One-particle electron matter layer; the quantum gauge-field Floquet state and thermodynamic spectral limit remain separate.'}
    assert wa<1e-12 and wj<1e-12 and mm<1e-11 and mp<1e-11
    (CERT/'c8j_gauged_julia_certificate.json').write_text(json.dumps(cert,indent=2)+'\n');print(json.dumps(cert,indent=2))
if __name__=='__main__':main()
