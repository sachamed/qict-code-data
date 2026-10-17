#!/usr/bin/env python3
"""Independent, finite-dimensional validation of the controlled TT extraction theorem."""
from __future__ import annotations
import json
from pathlib import Path

def chi(x, a2, a4, a6, const=1.3):
    return const + a2*x*x + a4*x**4 + a6*x**6

def extrapolate(x, a2, a4, a6):
    return (16*chi(x/2,a2,a4,a6)-chi(x,a2,a4,a6)-15*chi(0,a2,a4,a6))/(3*x*x)

records=[]
for a2,a4,a6 in [(0.125,2.2,1.1), (0.0,3.0,-2.0), (0.5004,-1.3,0.4)]:
    for x in (0.125,0.25,0.5):
        est=extrapolate(x,a2,a4,a6)
        bound=(5/12)*abs(a6)*x**4+2e-12
        err=abs(est-a2)
        records.append({'a2':a2,'x':x,'estimate':est,'error':err,'bound':bound,'pass':err<=bound})
assert all(q['pass'] for q in records)
result={'classification':'TT_LOW_MOMENTUM_EXTRAPOLATION_TEST','tested_pairs':len(records),'maximum_error':max(z['error'] for z in records),'all_passed':True,'records':records,
        'interpretation':'Synthetic analytic kernels test algebraic cancellation and error bounds; not a physical gravitational-coupling calculation.'}
out=Path(__file__).resolve().parents[1]/'results'/'TT_LOW_MOMENTUM_EXTRAPOLATION.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'all_passed':True,'tested_pairs':len(records),'maximum_error':result['maximum_error']}))
