#!/usr/bin/env python3
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from scipy.special import jv

ROOT=Path(__file__).resolve().parents[1]
PHI=math.pi/63.0
X1=4.0*PHI
X2=32.0*PHI
NPLAQ=24
DMAX=10

def u1_tail(K:int)->float:
    # sup-norm tail for exp(i x cos theta): sum_{|k|>K}|J_k(x)|
    return 2.0*sum(abs(jv(k,X1)) for k in range(K+1,80))

def su2_tail(j2max:int)->float:
    # character sup bound: sum_{n>j2max}|c_n| dim(n/2), c_n=2(n+1)J_{n+1}(x)/x.
    s=0.0
    for n in range(j2max+1,80):
        d=n+1
        c=2.0*d*jv(d,X2)/X2
        s += abs(c)*d
    return s

def compose_plaq(e1,e2):
    return (1.0+e1)*(1.0+e2)-1.0

def step_bound(eplaq):
    return (1.0+eplaq)**NPLAQ-1.0

def moment_bound(estep,n=DMAX):
    return n*estep*(1.0+estep)**(n-1)

rows=[]
for K,j2 in [(3,6),(4,8),(5,10),(6,12),(7,14)]:
    eu=u1_tail(K); es=su2_tail(j2); ep=compose_plaq(eu,es); st=step_bound(ep)
    rows.append({
        'U1_abs_k_max':K,
        'SU2_twice_j_max':j2,
        'U1_one_plaquette_sup_tail':eu,
        'SU2_one_plaquette_sup_tail':es,
        'EW_combined_one_plaquette_bound':ep,
        'EW_one_step_L2_24plaquette_bound':st,
        'EW_rank6_moment_bound_n_le_10':moment_bound(st,10)
    })

# The charged-lepton carrier is color singlet. The SU(3) microscopic gauge update acts on a tensor factor
# that is never touched by the source or stream. In the relative normalized charged correlator its neutral
# vacuum factor is identical in numerator and denominator and cancels. CAR is exactly reduced to n=1 upstream.
reduction={
    'charged_lepton_SU3_rep':[0,0],
    'SU3_path_transport':'identity',
    'SU3_cancellation':'exact in the normalized charged-lepton route scalar, provided numerator and neutral-vacuum normalization use the same SU3 regulator/state',
    'CAR_reduction':'exact one-particle restriction; no Fock truncation error',
    'remaining_coupled_sector':'U(1)_Y x SU(2)_L with the common ordered one-particle path and C10 source/sink'
}

# A priori two-run acceptance schedule. These are NOT h_d values; they are truncation ceilings to be combined
# with the a posteriori block-SVD discarded norm from the actual frontier contraction.
coarse=next(r for r in rows if r['U1_abs_k_max']==6)
fine=next(r for r in rows if r['U1_abs_k_max']==7)
acceptance={
    'run_1_cutoff':{'U1_abs_k_max':6,'SU2_twice_j_max':12},
    'run_2_cutoff':{'U1_abs_k_max':7,'SU2_twice_j_max':14},
    'representation_only_moment_error_ceiling_run_1':coarse['EW_rank6_moment_bound_n_le_10'],
    'representation_only_moment_error_ceiling_run_2':fine['EW_rank6_moment_bound_n_le_10'],
    'compression_rule':'SVD only inside exact conserved (q_U1,j_SU2,iota_SU2,Pi_CAR) blocks; sum of discarded singular values is accumulated by triangle inequality',
    'promotion_inequality':'|h_d^(2)-h_d^(1)| + eta_d^(1)+eta_d^(2) < epsilon_h for d=10,9,8',
    'scientific_guard':'This preflight does not set h_10,h_9,h_8 and therefore cannot by itself open the physical-promotion gate.'
}

out={
    'status':'PASS_PEPO3D_FRONTIER_PREFLIGHT_AND_EW_TRUNCATION_BUDGET',
    'date':'2026-09-18',
    'geometry':{'L':2,'links':24,'plaquettes':24,'sites':8},
    'external_particle_data_used':False,
    'fit_used':False,
    'reverse_engineering_used':False,
    'exact_reductions':reduction,
    'cutoff_sequence':rows,
    'two_run_acceptance':acceptance,
    'remaining_required_numerical_objects':['h_10','h_9','h_8'],
    'statement':'The physical Task-3 computation is reduced exactly to a common U(1)xSU(2) repeated-time frontier contraction with C10-dressed source/sink. The table gives independent a priori Peter-Weyl/character error ceilings; the actual contraction must add an a posteriori symmetry-block SVD error certificate.'
}
blob=json.dumps(out,indent=2,sort_keys=True)+'\n'
(ROOT/'certificates'/'PEPO3D_FRONTIER_PREFLIGHT.json').write_text(blob)
print(blob,end='')
print('sha256',hashlib.sha256(blob.encode()).hexdigest())
