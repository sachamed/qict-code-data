#!/usr/bin/env python3
"""Exact response-flow integration and quantitative electromagnetic boundary deficit."""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'global_sm_extension'/'RG_ALPHA_BOUNDARY_QUANTITATIVE.json'; OUT.parent.mkdir(parents=True,exist_ok=True)
kY=F(10,3); k2=F(2,1); k3=F(2,1)
sY=F(-41,160); s2=F(19,96); s3=F(7,16) # dXi/dlnmu = s/pi^2
sD=kY*sY+k2*s2
assert sD==F(-11,24)
assert F(115)*(sY-s2)+F(218)*(s3-s2)==0
# Post-comparison only: current CODATA 2022 value (latest available at NIST on 2026-09-17).
alpha_ext=7.2973525643e-3
ainv_ext=137.035999177
D_ext=1/(4*math.pi*alpha_ext)
obj={
 'status':'EXECUTED_EXACT_RG_AND_ALPHA_BOUNDARY_COUNT',
 'external_alpha_used_in_derivation':False,
 'finite_trace_metric':{'kY':'10/3','k2':'2','k3':'2'},
 'response_flow':{
   'XiY':'C_Y-(41/160)/pi^2 * ln(mu/mu0)',
   'Xi2':'C_2+(19/96)/pi^2 * ln(mu/mu0)',
   'Xi3':'C_3+(7/16)/pi^2 * ln(mu/mu0)',
   'integration_constants_full_gauge_sector':3,
   'RG_invariant':'115(XiY-Xi2)+218(Xi3-Xi2)',
 },
 'electromagnetic_combination':{
   'D_em_definition':'D_em=kY*XiY+k2*Xi2=1/(4*pi*alpha_em)',
   'exact_slope':'dD_em/dlnmu=-11/(24*pi^2)',
   'exact_inverse_alpha_slope_in_this_massless_one_loop_window':'d(alpha_em^-1)/dlnmu=-11/(6*pi)',
   'solution':'D_em(mu)=C_em-[11/(24*pi^2)] ln(mu/mu0)',
   'independent_boundary_constants_needed_for_alpha_alone':1,
 },
 'stationary_condition':{
   'result':'No nonzero stationary electromagnetic coupling exists at one loop in the declared Standard-Model regime. beta_e=0 selects e=0, hence alpha_em=0.',
   'consequence':'A stability condition beta_e=0 cannot derive alpha approximately 1/137. A nonzero value requires a microscopic response boundary (or a separately demonstrated nonperturbative fixed point).'
 },
 'postcomparison_CODATA_2022_latest_available':{
   'alpha':alpha_ext,'alpha_inverse':ainv_ext,
   'required_D_em':D_ext,
   'hypothetical_Xi_if_XiY_equals_Xi2_at_same_scale':D_ext/float(kY+k2),
   'source':'NIST/CODATA 2022 recommended constants; latest CODATA set available as of 2026-09-17.'
 },
 'qca_boundary_payload_present':(ROOT/'data'/'qca_gauge_response_boundaries.json').exists(),
 'closure_statement':'The beta functions determine slopes but not the additive response boundary. The missing scalar C_em is exactly the datum that a block-sparse QCA twist-stiffness calculation must supply; substituting CODATA for it would be a fit.'
}
OUT.write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
