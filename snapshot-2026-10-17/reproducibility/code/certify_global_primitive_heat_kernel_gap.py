#!/usr/bin/env python3
"""Explicit whole-support A7 gap from exact compact-group heat-kernel bounds.

This is an analytic certificate for the neutral primitive Hamiltonian.
It is not a numerical many-link vacuum vector or a Floquet Riesz gap.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/global_primitive_gap'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    payload=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(payload,allow_pickle=False) as z:
        links=len(z['links']);faces=len(z['plaquettes'])
        signs=z['plaquette_edge_sign'].astype(int)
        assert np.all(signs.sum(axis=1)==0)
    # e>8/3 gives e^-lambda < b^lambda at heat time one.
    b=F(3,8)
    u1=2*b*b/(1-b**6)
    su2=4*b**3/(1-F(9,4)*b**5)
    first_su3=F(2*3**6,64)*b**4
    second_su3=F(3*4**6,64)*b**9
    ratio_su3=F(4,3)*F(5,4)**6*b**6
    su3=first_su3+second_su3/(1-ratio_su3)
    assert all(0<x<F(1,2) for x in [u1,su2,su3])
    # n>=2: ceiling(3(n+1)^2/4+3(n+1)) -
    # ceiling(3n^2/4+3n) >=6; the growing polynomial ratio decreases.
    for n in range(2,256):
        exponent=lambda t:(3*t*t+12*t+3)//4
        assert exponent(n+1)-exponent(n)>=6
        assert F(n+2,n+1)*F(n+3,n+2)**6<=F(4,3)*F(5,4)**6
    groups={}
    for name,kappa,A in [('U1',2,8),('SU2',3,64),('SU3',4,54)]:
        vmax=A*faces
        exponent=2*vmax+2*links
        groups[name]={'free_Haar_Poincare_gap':kappa,
            'magnetic_cost_upper':vmax,'strict_ground_gap_lower':{
                'numerator':kappa,'denominator_base':3,'denominator_exponent':exponent},
            'ground_amplitude_min_to_max_lower':{
                'numerator':1,'denominator_base':3,'denominator_exponent':vmax+links}}
    worst=groups['SU2']['strict_ground_gap_lower']
    assert worst['denominator_exponent']>groups['U1']['strict_ground_gap_lower']['denominator_exponent']
    assert worst['denominator_exponent']>groups['SU3']['strict_ground_gap_lower']['denominator_exponent']
    full_exponent=worst['denominator_exponent']-1
    result={'status':'EXACT_QUANTITATIVE_GLOBAL_PRIMITIVE_A7_KATO_GAP',
        'links':links,'plaquettes':faces,'heat_kernel_time':1,
        'per_link_heat_kernel_tail_upper_rational':{
            'U1':str(u1),'SU2':str(su2),'SU3':str(su3)},
        'per_link_heat_kernel_interval_uniform':['1/2','3/2'],
        'independent_group_gap_bounds':groups,
        'full_neutral_primitive_gap_lower':{
            'numerator':1,'denominator_base':3,'denominator_exponent':full_exponent},
        'uniform_homotopy':'s in [0,1] for K_E+s*K_B on the fixed isolated neutral gauge complex.',
        'linear_U1_term':'Vanishes on closed neutral Gauss flows by the stored coordinate incidence identity.',
        'primitive_ground':'Unique strictly positive gauge-invariant ground of the auxiliary symmetric Laplacian plus bounded gauge potential; agrees with the neutral frozen A7 Hamiltonian.',
        'normalization':'Rank-one ground projector is a positive trace-one neutral primitive density on the exact Haar carrier.',
        'proof_method':'Uniform character heat-kernel bounds, positive semigroup comparison, ground-state transform and weighted Haar Poincare inequality.',
        'ground_vector_numerically_contracted':False,
        'spatially_uniform_gap_certified':False,
        'Floquet_Riesz_gap_certified':False,
        'full_C10_contract_completed':False,
        'scope':'Whole authentic finite gauge support and all three groups; primitive A7 state only. The exponentially small explicit bound is not an efficient numerical state-contraction certificate or a thermodynamic limit.',
        'payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}
    (OUT/'GLOBAL_PRIMITIVE_HEAT_KERNEL_GAP_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'tail_bounds':{
        'U1':float(u1),'SU2':float(su2),'SU3':float(su3)},
        'global_gap_denominator_power_of_three':full_exponent,
        'full_C10':False},indent=2))


if __name__=='__main__':main()
