#!/usr/bin/env python3
"""Exact scale-free directional prediction of the BCC photon tangent law."""
from fractions import Fraction
from pathlib import Path
import csv, json, math
ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; DATA=ROOT/'data'; CERT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
# b=(11-8 S4)/32
vals={
 'axis':(Fraction(1,1),Fraction(3,32)),
 'face_diagonal':(Fraction(1,2),Fraction(7,32)),
 'body_diagonal':(Fraction(1,3),Fraction(25,96)),
}
ratio=[vals['axis'][1]*96,vals['face_diagonal'][1]*96,vals['body_diagonal'][1]*96]
assert ratio==[9,21,25]
mean=Fraction(31,160) # <S4>=3/5 on S2
var=Fraction(1,525)   # Var(b), from E[S4^2]=41/105
spread=vals['body_diagonal'][1]-vals['axis'][1]
assert spread==Fraction(1,6)
rows=[]
for name,(s4,b) in vals.items():
    rows.append({'direction':name,'S4':f'{s4.numerator}/{s4.denominator}','b_exact':f'{b.numerator}/{b.denominator}','b_float':float(b)})
with (DATA/'BCC_SCALE_FREE_PHOTON_DIRECTIONAL_PREDICTION.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
out={
 'status':'PASS_SCALE_FREE_BCC_PHOTON_PREDICTION',
 'claim_status':'PROVEN',
 'external_photon_data_used':False,
 'law':'v_gamma/c = 1 - b(n)(E/E_tau)^2 + O((E/E_tau)^4), b(n)=(11-8 sum_i n_i^4)/32',
 'sign':'subluminal at quadratic order in every direction because 3/32 <= b <= 25/96',
 'high_symmetry_coefficients':{k:v[1].__str__() for k,v in vals.items()},
 'parameter_free_ratio_fixed_energy':'axis:face_diagonal:body_diagonal = 9:21:25',
 'equivalent_relations':['b_face/b_axis=7/3','b_body/b_axis=25/9','(b_body-b_axis)/(b_face-b_axis)=4/3'],
 'spherical_mean_b':'31/160',
 'spherical_variance_b':'1/525',
 'rms_cubic_anisotropy':math.sqrt(1/525),
 'max_minus_min_b':'1/6',
 'experimental_domain':'E/E_tau << 1; compare photons of the same observed energy after transforming source directions into the fixed microscopic BCC frame.',
 'interpretation':'The overall amplitude contains E_tau, but the angular ratios are independent of tau_* and of all continuously fitted coefficients. A statistically significant incompatible angular ratio falsifies the BCC photon tangent completion.',
 'experimental_confirmation_status':'NOT_YET_CONFIRMED; existing source-corrected Fermi-LAT d=6 intervals are consistent with zero and therefore do not establish this anisotropy.'
}
(CERT/'BCC_SCALE_FREE_PHOTON_DIRECTIONAL_PREDICTION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
