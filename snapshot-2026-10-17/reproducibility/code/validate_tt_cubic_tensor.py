#!/usr/bin/env python3
"""Exact TT cubic-invariant counterexample in the QICT spectral source analysis."""
import json
from pathlib import Path
import numpy as np

n_a = np.array([1., 0., 0.])
e_a = np.diag([0., 1., -1.]) / np.sqrt(2)
n_b = np.array([1., 1., 0.]) / np.sqrt(2)
u = np.array([1., -1., 0.]) / np.sqrt(2)
v = np.array([0., 0., 1.])
e_b = (np.outer(u,u) - np.outer(v,v)) / np.sqrt(2)

def diagnostics(n,e):
    return {
      'unit_n':float(np.linalg.norm(n)),
      'tt_transverse_residual':float(np.linalg.norm(n @ e)),
      'tt_trace_residual':float(abs(np.trace(e))),
      'tt_norm':float(np.linalg.norm(e)),
      'cubic_gradient_coefficient':float(np.sum(n**2 * np.diag(e)**2)),
      'rotational_gradient_coefficient':float(np.sum(n**2)*np.sum(e**2)),
    }
a,b=diagnostics(n_a,e_a),diagnostics(n_b,e_b)
assert max(a['tt_transverse_residual'], b['tt_transverse_residual'],a['tt_trace_residual'],b['tt_trace_residual'])<1e-14
assert abs(a['cubic_gradient_coefficient'])<1e-14
assert abs(b['cubic_gradient_coefficient']-1/8)<1e-14
assert abs(a['rotational_gradient_coefficient']-1)<1e-14
assert abs(b['rotational_gradient_coefficient']-1)<1e-14
result={'test':'cubic_rank_four_TT_gradient_invariant','case_axis':a,'case_face':b,'coefficient_difference':b['cubic_gradient_coefficient']-a['cubic_gradient_coefficient'],'status':'passed','physical_scope':'finite-dimensional invariant identity; not a computed QICT graviton pole'}
out=Path(__file__).resolve().parent.parent/'results'/'TT_CUBIC_DIRECTIONAL_CERTIFICATE.json'
out.parent.mkdir(exist_ok=True,parents=True)
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
