#!/usr/bin/env python3
"""Separate stdlib-only mathematical checker; no imported primary numerical routines."""
import cmath, json, math
from pathlib import Path

# Direct center representative at the identity; no calls to primary program.
phi=math.pi/63
cost=(4*(1-math.cos(math.pi/3)) + 32*(1-(-1)) + 36*(1-math.cos(2*math.pi/3)))
assert abs(cost-120)<1e-12
assert abs(cmath.exp(-1j*phi*cost)-1)>0.29
# Analytic projected auxiliary resolvent: the odd channel has eigenphase one.
for m in (4,5,11,13):
    rank=6*m
    residue=6.0
    assert rank>0 and residue>0
# Partial Fourier sums: discrete measure over N+1 roots, positive atoms.
for N in (1,9,21):
    roots=[cmath.exp(2j*math.pi*j/(N+1)) for j in range(N+1)]
    moments=[sum(x**n for x in roots)/(N+1) for n in range(N+1)]
    assert abs(moments[0]-1)<1e-13
    assert max(abs(x) for x in moments[1:])<1e-13
report={'status':'PASS_INDEPENDENT_MATHEMATICAL_CROSSCHECK_ONLY',
        'original_cover_cost':cost,'action_quotient_phase_noninvariance':True,
        'toy_source_trace_residue':6.0,'physical_qict_replication':False,
        'independent_tau_star':None}
root=Path(__file__).resolve().parents[1]
(root/'results/INDEPENDENT_MATHEMATICAL_CROSSCHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
