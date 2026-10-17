#!/usr/bin/env python3
"""Exact finite-regulator obstruction: C8 dynamics alone does not select a state."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import compute_c8_static_charge_strip as strip
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True); CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
N=13;R=1;Q=-6

def dense_U(q):
    KE,_,_,_,_=strip.electric_diag(R,N,q); B=strip.theta_B_diag(R,N); dim=N
    eye=np.eye(dim,dtype=complex)
    return np.column_stack([strip.apply_U(eye[:,j],KE,B,R,N) for j in range(dim)])

def moments(v,U,nmax=24):
    out=[];psi=v.copy()
    for _ in range(nmax):out.append(np.vdot(v,psi));psi=U@psi
    return np.array(out)

def main():
    U0=dense_U(0);Uq=dense_U(Q)
    unit0=float(np.linalg.norm(U0.conj().T@U0-np.eye(N),2));unitq=float(np.linalg.norm(Uq.conj().T@Uq-np.eye(N),2))
    vals,vecs=np.linalg.eig(U0);ph=np.angle(vals);i=int(np.argmin(ph));j=int(np.argmax(ph))
    v1=vecs[:,i]/np.linalg.norm(vecs[:,i]);v2=vecs[:,j]/np.linalg.norm(vecs[:,j])
    e1=float(np.linalg.norm(U0@v1-vals[i]*v1));e2=float(np.linalg.norm(U0@v2-vals[j]*v2))
    c1=moments(v1,Uq);c2=moments(v2,Uq);diff=np.abs(c1-c2)
    cert={'status':'PASS_OBSTRUCTION','regulator':'gauge-invariant Z_13 single plaquette','q_open_string':Q,
      'U0_unitarity_operator_defect':unit0,'Uq_unitarity_operator_defect':unitq,
      'stationary_state_1_eigenphase':float(ph[i]),'stationary_state_2_eigenphase':float(ph[j]),
      'stationarity_residual_1':e1,'stationarity_residual_2':e2,
      'max_charged_moment_difference_n0_23':float(diff.max()),'first_n_with_difference_gt_1e-8':int(next(k for k,x in enumerate(diff) if x>1e-8)),
      'C1_state1':[float(c1[1].real),float(c1[1].imag)],'C1_state2':[float(c2[1].real),float(c2[1].imag)],
      'theorem_conclusion':'A fixed C8 unitary admits distinct stationary neutral states that induce different charged moment sequences. The dynamics therefore does not by itself define a unique electron spectral measure; a state-selection theorem or preparation prescription is mathematically necessary.',
      'external_particle_data_used':False}
    assert unit0<1e-12 and unitq<1e-12 and e1<1e-10 and e2<1e-10 and diff.max()>1e-3
    (CERT/'c8_state_selection_obstruction_certificate.json').write_text(json.dumps(cert,indent=2)+'\n');print(json.dumps(cert,indent=2))
if __name__=='__main__':main()
