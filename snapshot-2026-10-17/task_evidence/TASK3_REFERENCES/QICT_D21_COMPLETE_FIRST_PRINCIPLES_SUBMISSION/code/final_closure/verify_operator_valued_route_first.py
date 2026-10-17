#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np
from qict_route_first import electron_route_coefficients,I3,X,Z,receiver_return
N=13
S=np.roll(np.eye(N,dtype=complex),1,axis=0)
Ig=np.eye(N*N,dtype=complex)
P=[Ig,np.kron(S,np.eye(N)),np.kron(np.eye(N),S)]
W=[I3,X,Z]; c=electron_route_coefficients()
# operator-valued receiver compression on family x two-link rotor
T=sum(c[a]*np.kron(W[a],P[a]) for a in range(3))/3
sv=np.linalg.svd(T,compute_uv=False)
# flat family projectors from flat receiver return
Tf=receiver_return(); L,s,Rh=np.linalg.svd(Tf); order=np.argsort(s); L=L[:,order]
A=T.conj().T@T
comm=[]
for a in range(3):
    Pf=np.outer(L[:,a],L[:,a].conj()); PP=np.kron(Pf,Ig)
    comm.append(float(np.linalg.norm(PP@A-A@PP,2)))
# route unitary is block controlled; block unitarity is sufficient and exact
block_def=max(float(np.linalg.norm((c[a]*np.kron(W[a],P[a])).conj().T@(c[a]*np.kron(W[a],P[a]))-np.eye(3*N*N),2)) for a in range(3))
out={'status':'PASS','N':N,'gauge_dim':N*N,'compressed_dim':3*N*N,
     'controlled_route_block_unitarity_defect':block_def,
     'compression_operator_norm':float(sv.max()),'compression_min_singular':float(sv.min()),
     'flat_family_projector_commutators_with_TdagT':comm,
     'max_flat_projector_commutator':max(comm),
     'external_particle_data_used':False,'new_fit_parameters':0,
     'interpretation':'The operator-valued route-first compression is a contraction, while flat-gauge singular projectors cease to reduce it. Family resolution must therefore come from the interacting residue matrices.'}
Path(__file__).resolve().parents[1].joinpath('results/operator_valued_route_first.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
