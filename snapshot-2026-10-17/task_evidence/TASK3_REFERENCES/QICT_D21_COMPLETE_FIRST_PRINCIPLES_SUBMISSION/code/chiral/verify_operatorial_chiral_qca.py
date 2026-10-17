#!/usr/bin/env python3
"""Finite Z_N quantum-link check of the operatorial local matter-gauge hopping theorem.
The theorem in Supplement W is group-theoretic; this script is an independent finite regulator certificate.
"""
import json, math
from pathlib import Path
import numpy as np
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[2]
N=7; omega=np.exp(2j*np.pi/N)
# one fermionic mode at each of two sites, basis |nx,ny>
basis=[(0,0),(0,1),(1,0),(1,1)]; idx={b:i for i,b in enumerate(basis)}
def annih(site):
    A=np.zeros((4,4),complex)
    for j,(nx,ny) in enumerate(basis):
        n=[nx,ny]
        if n[site]:
            sign=(-1)**sum(n[:site]); n2=n.copy();n2[site]=0
            A[idx[tuple(n2)],j]=sign
    return A
cx=annih(0); cy=annih(1); nx=cx.conj().T@cx; ny=cy.conj().T@cy
U=np.zeros((N,N),complex)
for e in range(N): U[(e+1)%N,e]=1
Z=np.diag([omega**e for e in range(N)])
I4=np.eye(4); IN=np.eye(N)
# term cy^dag U cx + h.c.; tensor order matter,link
H=np.kron(cy.conj().T@cx,U)+np.kron(cx.conj().T@cy,U.conj().T)
herm=np.linalg.norm(H-H.conj().T,2)
V=expm(-1j*0.371*H); un=np.linalg.norm(V.conj().T@V-np.eye(4*N),2)
# exact Z_N local gauge transformations; G(a,b)=omega^(a nx+b ny) \otimes Z^(a-b)
maxcomm=0.0
for a in range(N):
  for b in range(N):
    M=expm(2j*np.pi/N*(a*nx+b*ny))
    G=np.kron(M,np.linalg.matrix_power(Z,(a-b)%N))
    maxcomm=max(maxcomm,float(np.linalg.norm(G@V-V@G,2)))
# invariant-subspace preservation follows from commutation; test group-averaged projector
P=np.zeros((4*N,4*N),complex)
for a in range(N):
  for b in range(N):
    M=expm(2j*np.pi/N*(a*nx+b*ny)); G=np.kron(M,np.linalg.matrix_power(Z,(a-b)%N)); P+=G
P/=N*N
proj_idem=np.linalg.norm(P@P-P,2); preserve=np.linalg.norm((np.eye(4*N)-P)@V@P,2)
cert={
 'status':'PASS' if max(herm,un,maxcomm,proj_idem,preserve)<1e-11 else 'FAIL',
 'regulator':'Z_7 one-link quantum-link certificate',
 'hopping_generator_hermiticity_defect':float(herm),
 'gate_unitarity_defect':float(un),
 'max_local_gauge_commutator_defect':float(maxcomm),
 'gauge_projector_idempotence_defect':float(proj_idem),
 'physical_subspace_leakage':float(preserve),
 'scope':'Independent finite-group regulator test of the local operator theorem; not the full Standard-Model G2 contraction.'
}
out=ROOT/'certificates/OPERATORIAL_CHIRAL_QCA.json';out.write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2)); raise SystemExit(0 if cert['status']=='PASS' else 1)
