"""Exact integer arithmetic for QICT (T2238), T2241 and T2242.

NON-QICT-FULL seed demonstration: oriented single U(1) plaquette only.
"""
import json
from fractions import Fraction
from pathlib import Path

def identity_h(n,eta):
    """Proof witness: h(n+126 eta)-h(n) = 126(4 eta n+252 eta^2+eta)."""
    h=lambda x:2*x*x+x
    return h(n+126*eta)-h(n)==126*(4*eta*n+252*eta*eta+eta)

def phase_one(r):
    # U_E(r)=exp(-i * pi/63 * 8r^2), phase 1 iff 8r^2 divisible by126.
    return (8*r*r)%126==0

def rank(R):
    # r integer, -R <= r <=R.
    return sum(1 for r in range(-R,R+1) if phase_one(r))

def trace_distance_nested(R):
    # two normalized orthoprojectors on nested phase-one flux sets.
    r1=rank(R);r2=rank(2*R)
    return Fraction(r2-r1,r2)

def exact_obstruction():
    assert all(identity_h(n,eta) for n in [-255,-15,-1,0,1,27,518]
               for eta in [-7,-3,-1,0,1,4,8])
    assert all(phase_one(r)==(r%21==0) for r in range(-1000,1001))
    records=[]
    for m in [1,2,4,8,16]:
        R=63*m
        r1=rank(R);r2=rank(2*R)
        assert (r1,r2)==(6*m+1,12*m+1)
        d=trace_distance_nested(R)
        records.append({'R':R,'rank_R':r1,'rank_2R':r2,
                        'trace_distance_exact':f'{d.numerator}/{d.denominator}',
                        'trace_distance':float(d)})
    return {'classification':'EXACT_ELECTRIC_FLOQUET_SEED_NOT_FULL_QICT',
        'periodicity_h_expansion_exact':'h(n+126 eta)-h(n)=126(4 eta n+252 eta^2+eta)',
        'periodicity_h_formula_regression_samples_passed':True,
        'phase_one_flux_condition_exact':'r is divisible by 21',
        'rank_formula_R_63m':'6m+1',
        'trace_distance_formula':'(6m)/(12m+1) -> 1/2',
        'not_trace_norm_cauchy':True,'data':records,
        'physical_Floquet_Kato_density_constructed':False}

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    data=exact_obstruction()
    (root/'results/EXACT_ELECTRIC_ENDPOINT_OBSTRUCTION.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
