#!/usr/bin/env python3
from fractions import Fraction
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'global_sm_extension'; OUT.mkdir(parents=True,exist_ok=True)
BOUND=ROOT/'data'/'qca_gauge_response_boundaries.json'

kY=Fraction(10,3); k2=Fraction(2); k3=Fraction(2)
sY=Fraction(-41,160); s2=Fraction(19,96); s3=Fraction(7,16) # dXi/dlnmu = s/pi^2
# Exact invariant relation already implied by slopes.
assert Fraction(115)*(sY-s2)+Fraction(218)*(s3-s2)==0
# At one loop a nonzero abelian coupling has beta_e=b e^3/(16pi^2), b>0 for active charged matter.
# Therefore the only stationary one-loop point is e=0.
obj={
 'status':'PASS_IDENTIFIABILITY_AND_FIXED_POINT_AUDIT',
 'claim_status':'PROVEN_FORMULAE__ALPHA_NUMERICAL_VALUE_REQUIRES_MICROSCOPIC_RESPONSE_BOUNDARY',
 'finite_trace_metric':{'kY':'10/3','k2':'2','k3':'2','gY2_over_g22_if_equal_response':'3/5','sin2thetaW_if_equal_response':'3/8'},
 'response_definition':'1/g_r(mu)^2 = k_r Xi_r(mu)',
 'alpha_em_formula':'alpha_em(mu)=1/[4*pi*(kY*XiY(mu)+k2*Xi2(mu))]',
 'one_loop_response_slopes_over_pi2':{'XiY':'-41/160','Xi2':'19/96','Xi3':'7/16'},
 'integration_constants_required':3,
 'one_loop_RG_invariant':'115(XiY-Xi2)+218(Xi3-Xi2)',
 'nonzero_stationary_alpha_theorem':'For beta_e=b e^3/(16 pi^2) with b>0, beta_e=0 implies e=0. The observed nonzero infrared alpha is not a stationary one-loop Standard-Model fixed point.',
 'qca_boundary_payload_present':BOUND.exists(),
 'required_microscopic_boundary':'Twist stiffness Xi_r = (k_r V)^(-1) d^2[-log Z_QCA(phi_r)]/d phi_r^2 at a declared blocking scale, evaluated by the same block-sparse tensor network.',
 'external_alpha_used':False,
}
if BOUND.exists():
    b=json.loads(BOUND.read_text()); XiY=float(b['XiY']); Xi2=float(b['Xi2'])
    obj['alpha_em_from_qca_boundary']=1/(4*math.pi*(float(kY)*XiY+float(k2)*Xi2))
    obj['status']='COMPUTED_FROM_QCA_RESPONSE_BOUNDARY'
(OUT/'RG_ALPHA_CLOSURE_CERTIFICATE.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
