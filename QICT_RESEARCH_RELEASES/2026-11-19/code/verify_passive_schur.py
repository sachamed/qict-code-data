"""Numerical stress tests for a proved Hermitian Schur-slope identity."""
from pathlib import Path
import json
import numpy as np
rng=np.random.default_rng(20261118)
checks=0
for dimension in (1,2,3,6):
    for external in (2,4,7):
        for draw in range(20):
            X=rng.normal(size=(external,external))+1j*rng.normal(size=(external,external))
            D=(X+X.conj().T)/2
            B=rng.normal(size=(dimension,external))+1j*rng.normal(size=(dimension,external))
            eigen=np.linalg.eigvalsh(D)
            E=float(eigen[-1]+1+rng.random())
            Q=np.linalg.inv(E*np.eye(external)-D)
            K=np.eye(dimension)+B@(Q@Q)@B.conj().T
            e=np.linalg.eigvalsh(K)
            assert e.min()>=1-1e-10
            assert np.linalg.norm(K-K.conj().T)<1e-10
            eta=1e-5
            def Ginv(x):return x*np.eye(dimension)-B@np.linalg.inv(x*np.eye(external)-D)@B.conj().T
            deriv=(Ginv(E+eta)-Ginv(E-eta))/(2*eta)
            assert np.linalg.norm(deriv-K)<1e-7
            checks+=3
p=Path(__file__).parent/'results'/'PASSIVE_SCHUR_CHECK.json'
p.write_text(json.dumps({'passed':True,'tests':checks,'independent_Hermitian_instances':240,'identity':'partial_E Ginv = I + B (E I - D)^(-2) B_dagger >= I','scope':'finite dimensional selfadjoint energy-independent H; normalized orthogonal source projection; real E outside sigma(D)'},indent=2)+'\n')
print('PASS',checks,'checks')
