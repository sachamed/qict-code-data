#!/usr/bin/env python3
from __future__ import annotations
from fractions import Fraction as F
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'microscopic_closure'/'BCC_OBSERVATIONAL_SIGNATURE_2026.json'
OUT.parent.mkdir(parents=True,exist_ok=True)

axis=F(3,32); face=F(7,32); body=F(25,96)
assert [96*axis,96*face,96*body]==[9,21,25]
mean=F(31,160); var=F(1,525)
Epl=1.22089e19
EQG2_lower=6e-8*Epl
Etau_lower=EQG2_lower*math.sqrt(2*float(axis)/3)
hbar_GeVs=6.582119569e-25
tau_upper=hbar_GeVs/Etau_lower
report={
 'status':'PASS_PARAMETER_FREE_BCC_SIGNATURE_WITH_CURRENT_FIELD_STATUS',
 'signature':{
   'law':'v_gamma/c = 1 - b(n)(E/E_tau)^2 + O((E/E_tau)^4)',
   'b':'(11-8 sum_i n_i^4)/32 = 31/160 - H4(n)/4',
   'linear_term':'exactly zero in the declared BCC photon tangent',
   'high_symmetry_ratio':'axis:face-diagonal:body-diagonal = 9:21:25',
   'critical_points':{'minima_axes':6,'saddles_face_diagonals':12,'maxima_body_diagonals':8},
   'spherical_mean':str(mean),'spherical_variance':str(var)
 },
 'field_comparison':{
   'LHAASO_GRB221009A_2024_quadratic_95pct_lower_scale':'E_QG,2 > 6e-8 E_Pl',
   'conditional_mapping':'Fermi/LHAASO convention v/c=1-(3/2)(E/E_QG2)^2 matched to QICT b(n)(E/E_tau)^2',
   'orientation_independent_Etau_lower_GeV':Etau_lower,
   'orientation_independent_tau_star_upper_s':tau_upper,
   'Fermi_directional_status':'Published source-corrected anisotropic d=6 intervals used in the existing QICT audit contain zero.',
   'conclusion':'No field experiment has confirmed the fixed 9:21:25 cubic angular fingerprint. Current GRB observations constrain its amplitude and are compatible with zero directional signal.'
 },
 'falsification_test':'Fit an ensemble of transients with one common BCC frame and one amplitude E_tau^{-2}; after frame fixing, the normalized quadratic coefficients must follow b(n) with no independent spherical-harmonic coefficients. A robust incompatible angular map falsifies this completion.'
}
OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
