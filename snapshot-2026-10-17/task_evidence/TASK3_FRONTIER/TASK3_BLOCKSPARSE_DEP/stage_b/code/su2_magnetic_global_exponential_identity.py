#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,math,cmath
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
import su2_global_spin_network as gs
PHI=math.pi/63.0

def rand_su2(rng):
    q=rng.normal(size=4);q/=np.linalg.norm(q);a,b,c,d=q
    return np.array([[a+1j*d,c+1j*b],[-c+1j*b,a-1j*d]],complex)

def links(seed):
    rng=np.random.default_rng(seed);return [rand_su2(rng) for _ in range(24)]

def holonomy(U,p):
    H=np.eye(2,dtype=complex)
    for e,sgn in zip(p['edges'],p['signs']): H=H@(U[e] if sgn==1 else U[e].conj().T)
    return H

def self_test():
    mx=0.0; imag=0.0; rows=[]
    for seed in range(32):
        U=links(410000+seed); chars=[]
        lhs=1+0j
        for p in gs.PLAQ:
            chi=np.trace(holonomy(U,p)); chars.append(float(chi.real)); imag=max(imag,abs(float(chi.imag)))
            lhs*=np.exp(-1j*PHI*(32.0-16.0*chi))
        rhs=np.exp(-1j*PHI*32.0*len(gs.PLAQ))*np.exp(1j*16.0*PHI*sum(chars))
        err=abs(lhs-rhs);mx=max(mx,err)
        rows.append({'seed':seed,'sum_chi_half':sum(chars),'residual_abs':float(err)})
    assert mx<2e-13 and imag<2e-14
    return {'status':'PASS_EXACT_GLOBAL_MAGNETIC_EXPONENTIAL_REDUCTION','samples':32,'plaquettes':24,
            'identity':'prod_p exp[-i phi(32-16 chi_1/2(U_p))] = exp[-i 32 phi N_p] exp[i 16 phi H_B], H_B=sum_p chi_1/2(U_p)',
            'max_identity_residual_abs':mx,'max_character_imag_abs':imag,
            'consequence':'The 24 commuting magnetic multiplication gates need not be character-expanded separately during a Krylov realization; a single action of exp(i16 phi H_B) is mathematically identical. This does not remove the need for frontier compression when H_B Krylov support grows.',
            'rows':rows}
if __name__=='__main__':print(json.dumps(self_test(),indent=2))
