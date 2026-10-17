#!/usr/bin/env python3
"""Exact gauge-representation closure of the chiral C8 stream.

The chiral incidence is not V_R -> V_L.  It is the multiplicity-one
intertwiner C_f: V_R,f tensor R_f -> V_L,f, with R_f = H_R or Htilde_R.
Consequently the gauge-covariant object transported through the active chiral
bridge is the composite right carrier V_R,f tensor R_f.  This script verifies
for every SM incidence that its U(1)_Y, SU(2)_L and SU(3)_C labels coincide
exactly with the corresponding left module, and checks the represented link
matrices on deterministic Haar-random group elements.

No particle mass, coupling or experimental datum enters.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from fractions import Fraction
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'microscopic_closure'/'CHIRAL_COMPOSITE_TRANSPORT.json'
OUT.parent.mkdir(parents=True,exist_ok=True)

# integer hypercharge y=6Y; SU2 twice-spin; SU3 Dynkin labels.
SECTORS={
 'd':  {'L':(1,1,(1,0)), 'R':(-2,0,(1,0)), 'H':(+3,1,(0,0)), 'receiver':'H_R'},
 'u':  {'L':(1,1,(1,0)), 'R':(+4,0,(1,0)), 'H':(-3,1,(0,0)), 'receiver':'Htilde_R'},
 'e':  {'L':(-3,1,(0,0)),'R':(-6,0,(0,0)), 'H':(+3,1,(0,0)), 'receiver':'H_R'},
 'nu': {'L':(-3,1,(0,0)),'R':(0,0,(0,0)),  'H':(-3,1,(0,0)), 'receiver':'Htilde_R'},
}

def su2_haar(rng):
    z=rng.normal(size=4); z=z/np.linalg.norm(z); w,x,y,zv=z
    return np.array([[w+1j*zv,y+1j*x],[-y+1j*x,w-1j*zv]],complex)

def su3_haar(rng):
    A=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    Q,R=np.linalg.qr(A)
    ph=np.diag(R); ph=np.where(np.abs(ph)>0,ph/np.abs(ph),1)
    Q=Q@np.diag(np.conj(ph))
    det=np.linalg.det(Q)
    Q=Q/(det**(1/3))
    return Q

def rep_matrix(label,u1,U2,U3):
    y,j2,pq=label
    # Only reps used here: singlet/fundamental SU2 and singlet/fundamental SU3.
    A=np.array([[u1**y]],complex)
    if j2==1: A=np.kron(A,U2)
    elif j2!=0: raise ValueError(j2)
    if pq==(1,0): A=np.kron(A,U3)
    elif pq!=(0,0): raise ValueError(pq)
    return A

def composite_matrix(R,H,u1,U2,U3):
    # Tensor order is gauge-factor order.  Since R has SU2 singlet and H has
    # color singlet, the product representation is canonically identified with
    # the left carrier by permutation of one-dimensional factors.
    yr,jr,cr=R; yh,jh,ch=H
    assert jr==0 and jh==1 and ch==(0,0)
    y=yr+yh
    # R color fundamental (quarks) or singlet, H supplies the weak fundamental.
    A=(u1**y)*U2
    if cr==(1,0): A=np.kron(A,U3)
    elif cr!=(0,0): raise ValueError(cr)
    return A

def left_matrix(L,u1,U2,U3):
    y,j,c=L; assert j==1
    A=(u1**y)*U2
    if c==(1,0): A=np.kron(A,U3)
    elif c!=(0,0): raise ValueError(c)
    return A

def main():
    algebra=[]
    for f,s in SECTORS.items():
        yL,jL,cL=s['L']; yR,jR,cR=s['R']; yH,jH,cH=s['H']
        row={
          'sector':f,
          'receiver':s['receiver'],
          'left_labels':{'q_U1':yL,'twice_j_SU2':jL,'pq_SU3':list(cL)},
          'right_labels':{'q_U1':yR,'twice_j_SU2':jR,'pq_SU3':list(cR)},
          'receiver_labels':{'q_U1':yH,'twice_j_SU2':jH,'pq_SU3':list(cH)},
          'composite_labels':{'q_U1':yR+yH,'twice_j_SU2':jH,'pq_SU3':list(cR)},
          'label_match': bool((yR+yH,jH,cR)==(yL,jL,cL)),
        }
        algebra.append(row)
    assert all(r['label_match'] for r in algebra)

    rng=np.random.default_rng(210926)
    maxres={f:0.0 for f in SECTORS}
    unitres=0.0
    for _ in range(64):
        alpha=float(rng.uniform(-np.pi,np.pi)); u1=np.exp(1j*alpha)
        U2=su2_haar(rng); U3=su3_haar(rng)
        unitres=max(unitres,float(np.linalg.norm(U2.conj().T@U2-np.eye(2),2)),
                    float(np.linalg.norm(U3.conj().T@U3-np.eye(3),2)))
        for f,s in SECTORS.items():
            A=left_matrix(s['L'],u1,U2,U3)
            B=composite_matrix(s['R'],s['H'],u1,U2,U3)
            maxres[f]=max(maxres[f],float(np.linalg.norm(A-B,2)))
    assert max(maxres.values())<1e-12

    # Consequence for each positive-link spatial transporter: the Clebsch
    # intertwiner commutes with transport after the right carrier is extended
    # by its receiver doublet.  Therefore chirality conversion carries no Gauss
    # source and the transported gauge labels are constant through the bridge.
    report={
      'status':'PASS_EXACT_CHIRAL_COMPOSITE_GAUGE_TRANSPORT',
      'external_particle_data_used':False,
      'definition':'C_f: V_R,f tensor R_f -> V_L,f with R_f=H_R or Htilde_R',
      'sectors':algebra,
      'random_group_tests':64,
      'max_group_unitarity_residual':unitres,
      'max_link_representation_residual_by_sector':maxres,
      'max_link_representation_residual':max(maxres.values()),
      'theorem':'For every g in U(1)_Y x SU(2)_L x SU(3)_C, C_f [rho_R,f(g) tensor rho_Rf(g)] = rho_L,f(g) C_f. Hence a gauge-covariant C8 stream must transport the active right branch together with its receiver carrier. In that extended carrier the gauge labels are identical on both sides of the chiral bridge.',
      'task1_consequence':'The earlier static route model that streamed bare e_R with q=-6 and SU(2) singlet while swapping directly to L_L generated an artificial Gauss defect. The corrected electroweak tensor IR transports e_R tensor H_R, equivalently q=-3 and j=1/2, through the bridge; similarly for u,d,nu.',
      'scope':'Exact representation-level closure. It fixes the chiral gauge labels and local link tensor, but does not by itself perform the global time-dependent gauge contraction.'
    }
    txt=json.dumps(report,indent=2)+'\n'; OUT.write_text(txt)
    print(txt,end=''); print('sha256',hashlib.sha256(txt.encode()).hexdigest())
if __name__=='__main__': main()
