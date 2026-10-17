#!/usr/bin/env python3
"""Algebraic, deterministic check of the source-history contour ordering."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'CONTOUR_ORDERING_CERTIFICATE.json'
I = np.eye(2, dtype=complex)
X = np.array([[0., 1.], [1., 0.]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
rho = np.diag([0.8, 0.2]).astype(complex)
base = np.cos(0.41)*I - 1j*np.sin(0.41)*X

def U(j):
    return (np.cos(j)*I - 1j*np.sin(j)*Z) @ base

j_plus, j_minus = 0.23, -0.17
up, um = U(j_plus), U(j_minus)
trace_physical = np.trace(up @ rho @ um.conj().T)
trace_cyclic = np.trace(rho @ um.conj().T @ up)
trace_reversed = np.trace(rho @ up @ um.conj().T)
z_equal = np.trace(rho @ base.conj().T @ base)
z_exchange = np.trace(um @ rho @ up.conj().T)
# Independent analytic derivative of Tr[rho U0^dag e^{-ij Z} U0] at j=0
expected_derivative = -1j*np.trace(rho @ base.conj().T @ Z @ base)
h = 1e-6
fd = (np.trace(rho @ base.conj().T @ U(h)) - np.trace(rho @ base.conj().T @ U(-h)))/(2*h)
err_order = abs(trace_physical-trace_cyclic)
err_fd = abs(fd-expected_derivative)
err_unitarity = abs(z_equal-1)
err_exchange = abs(z_exchange-np.conj(trace_physical))
separation = abs(trace_reversed-trace_cyclic)
passed = (err_order < 1e-14 and err_fd < 1e-9 and err_unitarity < 1e-14 and err_exchange < 1e-14 and separation > 1e-3)
result = {
    'classification':'FINITE_DIMENSIONAL_CONTOUR_ORDERING_TEST',
    'basis_dimension':2,
    'physical_trace_cyclic_error':float(err_order),
    'derivative_finite_difference_error':float(err_fd),
    'equal_source_unitarity_error':float(err_unitarity),
    'branch_exchange_error':float(err_exchange),
    'separation_from_reversed_order':float(separation),
    'scope':'Finite-dimensional source-ordering and positivity identities; independent of continuum or phenomenological fits.',
    'pass':bool(passed),
}
OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
if not passed: raise SystemExit('Contour-ordering validation failed')
