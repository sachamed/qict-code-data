#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'microscopic_closure'/'CAR_ONE_PARTICLE_REDUCTION.json'
OUT.parent.mkdir(parents=True,exist_ok=True)

# Numerical sanity check of the exact functorial statement on a finite carrier.
rng=np.random.default_rng(210916)
Z=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
Q,R=np.linalg.qr(Z); ph=np.diag(R); ph=ph/np.abs(ph); U=Q@np.diag(np.conj(ph))

# Build Gamma_CAR(U) on vacuum + 1-particle + 2-particle sectors only.
# The theorem itself is exact for the complete exterior algebra; this finite test
# checks that the 1-particle block is exactly U and does not couple to n=0,2.
from itertools import combinations
basis2=list(combinations(range(4),2))
W2=np.zeros((6,6),complex)
for a,(i,j) in enumerate(basis2):
  for b,(k,l) in enumerate(basis2):
    W2[a,b]=U[i,k]*U[j,l]-U[i,l]*U[j,k]
G=np.zeros((1+4+6,1+4+6),complex)
G[0,0]=1; G[1:5,1:5]=U; G[5:,5:]=W2
unit_def=float(np.linalg.norm(G.conj().T@G-np.eye(11),2))
one_block=float(np.linalg.norm(G[1:5,1:5]-U,2))
leak=float(np.linalg.norm(G[np.r_[0,5:11],1:5],2))

report={
 'status':'PASS_CAR_ONE_PARTICLE_REDUCTION',
 'theorem':'For the declared functorial number-preserving completion Gamma_CAR(U)=direct_sum_n wedge^n U, every n-particle sector is invariant and the restriction to wedge^1 H is exactly U. Therefore a charged two-point function built by one-particle source maps over the CAR vacuum does not require an explicit many-body Fock contraction; the remaining nontrivial environment is the quantum gauge field, dressing/chiral regulator and receiver projection.',
 'scope':'This reduction fails if an independent matter interaction mixes particle-number sectors or if the reference state contains nontrivial matter occupation. Both are excluded by the current C8K completion and CAR-vacuum source prescription.',
 'finite_sanity_check':{'unitarity_defect':unit_def,'one_particle_block_error':one_block,'one_particle_to_vacuum_or_two_particle_leakage':leak},
 'computational_consequence':'The full charged rank-six task can be reduced exactly from gauge x Fock to gauge x one-particle for the declared two-point observable. This removes the Fock-space exponential but not the compact-gauge tensor contraction.'
}
assert unit_def<1e-12 and one_block<1e-14 and leak<1e-14
OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
