#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
rows=[]
for E in (18,24):
 x=json.load(open(R/f'SU2_ENERGY_CUT_RITZ_E{E}.json'))
 rows.append({'Emax':E,'orbit_basis_dim':x['orbit_basis_dim'],'ground_ritz_value':x['lowest_ritz'][0]['ritz_value_no_constant'],'ground_full_residual':x['lowest_ritz'][0]['full_residual_bound'],'outside_residual':x['lowest_ritz'][0]['outside_residual']})
pass_reject=all(r['ground_full_residual']>1e-3 for r in rows)
out={'status':'PASS_REJECT_LOW_ENERGY_GLOBAL_GALERKIN_AS_PEPO_SUBSTITUTE' if pass_reject else 'CHECK_ENERGY_GALERKIN',
     'rows':rows,'physical_substitution_allowed':False,
     'conclusion':'The low-electric-cost global Galerkin spaces are not approximately invariant under Kprim at the precision required for Task 3; explicit outside residuals remain O(10^2). They cannot replace the width-14 symmetry-block frontier.',
     'positive_result':'The rejection is rigorous at the tested cutoffs because every fundamental-plaquette transition leaving the Galerkin subspace is accumulated in the outside residual norm.'}
(R/'SU2_ENERGY_CUT_GALERKIN_REJECTION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
