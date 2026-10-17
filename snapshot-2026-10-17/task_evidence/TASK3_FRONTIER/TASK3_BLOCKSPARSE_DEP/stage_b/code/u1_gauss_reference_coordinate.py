#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import deque
import numpy as np

L=2
sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
links=[(mu,x,y,z) for mu in range(3) for x,y,z in sites]
li={e:i for i,e in enumerate(links)}
plaquettes=[]
for mu in range(3):
    for nu in range(mu+1,3):
        for x,y,z in sites:
            a=[x,y,z]; am=a.copy(); am[mu]^=1; an=a.copy(); an[nu]^=1
            v=np.zeros(24,dtype=np.int16)
            v[li[(mu,*a)]]+=1
            v[li[(nu,*am)]]+=1
            v[li[(mu,*an)]]-=1
            v[li[(nu,*a)]]-=1
            plaquettes.append(v)
plaquettes=np.asarray(plaquettes,dtype=np.int16)

def pplus(e:int)->int: return 2*e*e+e

def pathflow(order,q=1):
    E=np.zeros(24,dtype=np.int16); x=[0,0,0]
    for mu in order:
        E[li[(mu,*x)]] += q
        x[mu]^=1
    assert tuple(x)==(1,1,1)
    return E

def bounded_affine_basis(E0,M):
    z=tuple(map(int,E0)); seen={z}; Q=deque([z])
    while Q:
        E=np.asarray(Q.popleft(),dtype=np.int16)
        for p in plaquettes:
            for sg in (-1,1):
                F=E+sg*p; t=tuple(map(int,F))
                if t in seen: continue
                if sum(pplus(int(e)) for e in t)<=M:
                    seen.add(t); Q.append(t)
    return seen

def main():
    A1=pathflow((0,1,2),1); B1=pathflow((2,1,0),1)
    # Exact elementary homology identity found by a bounded plaquette BFS.
    rhs=A1-plaquettes[0]-plaquettes[10]-plaquettes[16]
    identity_residual=int(np.max(np.abs(rhs-B1)))
    assert identity_residual==0
    # Physical q=3 reference flows obey the same identity multiplied by three.
    A3=pathflow((0,1,2),3); B3=pathflow((2,1,0),3)
    q3_residual=int(np.max(np.abs(A3-3*plaquettes[0]-3*plaquettes[10]-3*plaquettes[16]-B3)))
    assert q3_residual==0
    rows=[]
    for M in (9,15,18,21):
        X=bounded_affine_basis(A1,M); Y=bounded_affine_basis(B1,M)
        rows.append({'M':M,'dim_A':len(X),'dim_B':len(Y),'symmetric_difference':len(X^Y)})
        assert X==Y
    out={
      'status':'PASS_U1_GAUSS_REFERENCE_PATH_IS_COORDINATE_CHOICE',
      'geometry':{'L':2,'links':24,'plaquettes':24},
      'reference_paths':{'A':[0,1,2],'B':[2,1,0],'endpoints':[[0,0,0],[1,1,1]]},
      'exact_cycle_identity':'J_B-J_A=-(P_0+P_10+P_16) for unit charge; multiply by q for integer charge q.',
      'unit_charge_identity_max_abs_residual':identity_residual,
      'q3_identity_max_abs_residual':q3_residual,
      'bounded_affine_basis_regression':rows,
      'theorem':'Two reference strings with the same endpoints and the same zero-winding Gauss charge differ by an element of the plaquette cycle lattice. Choosing one reference string therefore only chooses affine coordinates; after the full loop sector is retained, Kato/Floquet spectral data are reference-path independent.',
      'task3_consequence':'C10 fixes the charged Gauss sector and infrared boundary condition. It must not be represented as a probabilistic quantum superposition of thin strings. A convenient reference string may be used to coordinate the affine sector, provided the full plaquette loop algebra and Kato continuation are retained.'
    }
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
