#!/usr/bin/env python3
"""Independent consistency check for the stored finite q3/Kprim recurrence.
No claim of full physical closure is made.
"""
from pathlib import Path
import numpy as np, json
root=Path(__file__).resolve().parent
def from_unique_source(filename):
    matches=list(root.parent.rglob(filename))
    assert len(matches)==1, f'Expected one canonical payload for {filename}, found {len(matches)}'
    return json.loads(matches[0].read_text())
mom=from_unique_source('KPRIM_Q3_SCALAR_RECURRENCE_COMPLETE.json')
ritz=from_unique_source('SU2_KPRIM_RITZ_OBSTRUCTION.json')
q=float(mom['q3_KE_q3'])-16*float(mom['q3_HB_q3'])
n2=float(mom['KEq3_norm2'])-32*float(mom['q3_KE_HB_q3'])+256*float(mom['HBq3_norm2'])
beta3=float(mom['beta3']);beta4=np.sqrt(max(0,n2-q*q-beta3*beta3))
assert abs(q-float(mom['alpha3']))<1e-10
assert abs(beta4-float(mom['beta4']))<1e-9
T=np.array(ritz['tridiagonal'],float)
eval,vec=np.linalg.eigh(T)
rho=np.abs(beta4*vec[-1,:])
expected=np.array([x['exact_ritz_residual_norm'] for x in ritz['ritz_pairs']],float)
expected_eval=np.array([x['ritz_value'] for x in ritz['ritz_pairs']],float)
assert np.max(np.abs(eval-expected_eval))<1e-8
assert np.max(np.abs(rho-expected))<1e-8
assert rho.min()>float(ritz['threshold_for_physical_substitution'])
print(json.dumps({'status':'PASS_FINITE_Q3_LANCZOS_CONSISTENCY', 'alpha3':q,'beta4':beta4,'min_ritz_residual':float(rho.min()),'disallows_physical_substitution':True},indent=2))
