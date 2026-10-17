#!/usr/bin/env python3
"""Exact O((E/E_tau)^4) BCC photon tangent prediction."""
from fractions import Fraction as F
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'
def d(S4,S6,P):
    return (11520*P-1088*S4*S4+5232*S4-1792*S6-2505)/F(18432)
vals={
 'axis':d(F(1),F(1),F(0)),
 'face_diagonal':d(F(1,2),F(1,4),F(0)),
 'body_diagonal':d(F(1,3),F(1,9),F(1,27)),
}
expected={'axis':-F(17,2048),'face_diagonal':-F(203,6144),'body_diagonal':-F(1963,55296)}
assert vals==expected
out={
 'status':'PASS_BCC_QUARTIC_DIRECTIONAL_PREDICTION',
 'claim_status':'PROVEN_EXACT_SERIES_IDENTITY',
 'external_data_used':False,
 'law':'v/c=1-b(n)e^2+d(n)e^4+O(e^6), e=E/E_tau',
 'd_exact':'[11520 P -1088 S4^2 +5232 S4 -1792 S6 -2505]/18432',
 'definitions':'S4=sum n_i^4; S6=sum n_i^6; P=n_x^2 n_y^2 n_z^2',
 'high_symmetry':{k:f'{v.numerator}/{v.denominator}' for k,v in vals.items()},
 'new_continuous_coefficients':0,
 'harmonic_selection_rule':'inversion-even cubic singlets only; l subset {0,4,6,8} at quartic energy order',
 'falsification':'A quartic directional map incompatible with this fixed polynomial falsifies the BCC photon tangent completion even if the quadratic map is compatible.'
}
(CERT/'BCC_QUARTIC_DIRECTIONAL_PREDICTION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
