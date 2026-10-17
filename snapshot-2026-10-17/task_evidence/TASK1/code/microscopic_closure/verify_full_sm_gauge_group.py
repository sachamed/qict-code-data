#!/usr/bin/env python3
from __future__ import annotations
import cmath, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'global_sm_extension'
OUT.mkdir(parents=True, exist_ok=True)

# One generation, q=6Y. Right-handed fields are listed as physical representations
# for the Z6 kernel check. For anomaly checks we convert RH fields to LH conjugates.
reps=[
    dict(name='Q_L', q=1, color=1, weak=1, mult=6),
    dict(name='u_R', q=4, color=1, weak=0, mult=3),
    dict(name='d_R', q=-2, color=1, weak=0, mult=3),
    dict(name='L_L', q=-3, color=0, weak=1, mult=2),
    dict(name='e_R', q=-6, color=0, weak=0, mult=1),
    dict(name='nu_R',q=0, color=0, weak=0, mult=1),
]

def phase(rep,a,b,n):
    # z=(omega_3^a,(-1)^b,exp(i n*pi/3)); color=1 means fundamental.
    return cmath.exp(2j*math.pi*a*rep['color']/3) * ((-1)**(b*rep['weak'])) * cmath.exp(1j*rep['q']*n*math.pi/3)

kernel=[]
max_defect=0.0
for a in range(3):
    for b in range(2):
        for n in range(6):
            defects=[abs(phase(r,a,b,n)-1) for r in reps]
            if max(defects)<1e-12:
                kernel.append((a,b,n))
            max_defect=max(max_defect, min(max(defects), 2.0))
expected=[(n%3,n%2,n) for n in range(6)]
assert sorted(kernel)==sorted(expected), (kernel,expected)

# LH Weyl fields for local anomalies. color T(fund)=1/2; weak T(doublet)=1/2.
# multiplicities not associated with the gauged factor are included explicitly.
# Q_L: q=1, color triplet, weak doublet
# u_R^c: q=-4, anti-triplet; d_R^c: q=+2, anti-triplet
# L_L: q=-3, weak doublet; e_R^c: q=+6; nu_R^c:0
A33Y=0.5*(2*1 + (-4) + 2)
A22Y=0.5*(3*1 + (-3))
Agr=6*1 + 3*(-4) + 3*2 + 2*(-3) + 6
A111=6*1**3 + 3*(-4)**3 + 3*2**3 + 2*(-3)**3 + 6**3
# SU(3)^3: two fundamental components in Q_L versus two antifundamental singlets.
A333=2-1-1
witten_doublets=3+1

checks={
    'kernel_has_six_elements':len(kernel)==6,
    'kernel_equals_diagonal_Z6':sorted(kernel)==sorted(expected),
    'SU3^2_U1_anomaly_zero':abs(A33Y)<1e-15,
    'SU2^2_U1_anomaly_zero':abs(A22Y)<1e-15,
    'grav^2_U1_anomaly_zero':Agr==0,
    'U1^3_anomaly_zero':A111==0,
    'SU3^3_anomaly_zero':A333==0,
    'Witten_SU2_anomaly_absent':witten_doublets%2==0,
}
assert all(checks.values())

obj={
 'status':'PASS_8_OF_8',
 'claim_status':'PROVEN_RELATIVE_TO_FINITE_ALGEBRA_REPRESENTATION',
 'finite_algebra':'C + H + M3(C)',
 'integer_hypercharge_normalization':'q=6Y',
 'global_gauge_group':'(SU(3)_C x SU(2)_L x U(1)_Y)/Z6',
 'Z6_generator':'(omega_3 I_3, -I_2, exp(i*pi/3))',
 'central_kernel_elements_a_b_n':kernel,
 'anomalies':{'SU3^2U1':A33Y,'SU2^2U1':A22Y,'grav^2U1':Agr,'U1^3':A111,'SU3^3':A333,'number_SU2_doublets_mod2':witten_doublets%2},
 'checks':checks,
 'lattice_uniqueness_boundary':'The split-step lattice alone does not select SU(3). Gauge-covariant streaming can be defined for any compact group with a chosen unitary representation. SU(3)_C is fixed here by the M3(C) summand and the faithful chiral representation, not by lattice topology alone.'
}
(OUT/'FULL_SM_GAUGE_GROUP_CERTIFICATE.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
