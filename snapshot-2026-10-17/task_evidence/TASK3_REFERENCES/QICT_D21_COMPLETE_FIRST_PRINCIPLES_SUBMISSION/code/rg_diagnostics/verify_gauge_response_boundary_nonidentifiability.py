#!/usr/bin/env python3
"""Exact one-loop integration-constant count for QICT gauge responses."""
from fractions import Fraction
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
# dXi/dlnmu = slopes/pi^2
sY=Fraction(-41,160); s2=Fraction(19,96); s3=Fraction(7,16)
dY2=sY-s2; d32=s3-s2
# Find integer null vector (a,b) for a*dY2+b*d32=0
# dY2=-109/240, d32=23/96; choose 115 and 218.
assert Fraction(115)*dY2+Fraction(218)*d32==0
out={
 'status':'PASS_GAUGE_RESPONSE_BOUNDARY_NONIDENTIFIABILITY',
 'claim_status':'PROVEN',
 'external_coupling_data_used':False,
 'one_loop_slopes_over_pi2':{'XiY':'-41/160','Xi2':'19/96','Xi3':'7/16'},
 'general_solution':'Xi_r(mu)=C_r+s_r*pi^{-2} ln(mu/mu0)',
 'integration_constants':3,
 'finite_trace_coefficients':{'kY':'10/3','k2':'2','k3':'2'},
 'theorem':'The finite trace coefficients and beta-function slopes do not determine the three additive response constants C_r. Therefore alpha_s(M_Z), alpha_em(M_Z), and sin^2(theta_W)(M_Z) are not parameter-free outputs until a microscopic QICT calculation supplies boundary values for the response functions.',
 'one_loop_invariant':'J_resp=115(XiY-Xi2)+218(Xi3-Xi2)',
 'invariant_derivative':'0',
 'equal_response_statement':'XiY=Xi2=Xi3 is not RG invariant because the three slopes are unequal.'
}
(CERT/'GAUGE_RESPONSE_BOUNDARY_NONIDENTIFIABILITY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
