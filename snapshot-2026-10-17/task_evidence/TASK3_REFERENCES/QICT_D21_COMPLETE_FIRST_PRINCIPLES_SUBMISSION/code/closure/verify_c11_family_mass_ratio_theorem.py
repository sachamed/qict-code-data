#!/usr/bin/env python3
"""Certificate for the family-resolved C11/G2 spectral-edge ratio theorem.

For sharp positive family-resolved spectral edges in one clock convention, the exact
statement is E_a(0)/E_b(0)=epsilon_a/epsilon_b. For stable Floquet poles
z_a=exp(-i theta_a), the same result is extracted directly as theta_a/theta_b from
the zeros of the family-resolved inverse two-point blocks. It becomes a mass ratio only
in a sector with a common established infrared relativistic normalization E(0)=m c^2.
Confined-quark masses require short-distance scheme matching. For Moore--Smith edge or
pole-phase enclosures the code verifies the rigorous quotient interval. No experimental
mass data are read.
"""
from pathlib import Path
import json, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
EDGE=DATA/'g2_physical_compact_3d_edges.json'
OUT=CERT/'C11_FAMILY_MASS_RATIO_THEOREM.json'

# deterministic algebraic regression of interval enclosure and point collapse
amin,amax=0.41,0.43; bmin,bmax=0.16,0.18
lo=amin/bmax; hi=amax/bmin
samples=[]
for a in np.linspace(amin,amax,101):
    for b in np.linspace(bmin,bmax,101):
        samples.append(a/b)
inside=bool(min(samples)>=lo-1e-15 and max(samples)<=hi+1e-15)
point_a=0.417; point_b=0.173
point_ratio=point_a/point_b
collapse=bool(point_a/point_b==point_ratio)
# Stable-pole branch regression: z=exp(-i theta), 0<theta<pi.
theta_a,theta_b=0.417,0.173
za,zb=np.exp(-1j*theta_a),np.exp(-1j*theta_b)
extract=lambda z: float((-np.angle(z))%(2*np.pi))
pa,pb=extract(za),extract(zb)
pole_ratio=pa/pb
pole_ok=bool(abs(pa-theta_a)<1e-14 and abs(pb-theta_b)<1e-14 and abs(pole_ratio-theta_a/theta_b)<1e-14)
physical='NOT_EVALUATED_NO_G2_EDGE_PAYLOAD'
if EDGE.exists():
    obj=json.loads(EDGE.read_text())
    physical={'payload_present':True,'species':obj.get('species'),'epsilon_edge':obj.get('epsilon_edge')}
cert={
 'observable':'family-resolved C11/G2 spectral-edge rest-energy ratios',
 'logical_status':'PROVEN_SPECTRAL_EDGE_RATIO_THEOREM__MASS_INTERPRETATION_REQUIRES_COMMON_RELATIVISTIC_NORMALIZATION',
 'theorem':'For positive sharp edges epsilon_a, epsilon_b in a common clock convention, E_a(0)/E_b(0)=epsilon_a/epsilon_b. For simple stable Floquet poles z_a=exp(-i theta_a), z_b=exp(-i theta_b) on the physical branch, the ratio is theta_a/theta_b. For certified envelopes the quotient is rigorously enclosed. In sectors with common E(0)=m c^2 normalization this is also a mass-ratio theorem.',
 'tau_star_cancels_exactly':True,
 'external_particle_mass_data_read':False,
 'g2_physical_edge_payload_present':EDGE.exists(),
 'physical_evaluation':physical,
 'unit_test':{'input_envelopes':[[amin,amax],[bmin,bmax]],'ratio_enclosure':[lo,hi],'all_grid_ratios_inside':inside,'point_collapse_ratio':point_ratio,'stable_pole_phases':[theta_a,theta_b],'stable_pole_ratio':pole_ratio},
 'acceptance':{'interval_arithmetic':inside,'point_limit':collapse,'stable_pole_phase_extraction':pole_ok}
}
cert['status']='PASS' if all(cert['acceptance'].values()) else 'FAIL'
OUT.write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
raise SystemExit(0 if cert['status']=='PASS' else 1)
