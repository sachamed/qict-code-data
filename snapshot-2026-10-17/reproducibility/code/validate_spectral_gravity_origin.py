#!/usr/bin/env python3
"""Deterministic validation of the Kato--Riesz formulas used for QICT gravity response.

This script validates algebraic identities only. It does NOT assign physical values to
QICT gamma_R or epsilon_0; those require the source-deformed interacting BCC update.
"""
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "SPECTRAL_GRAVITY_ORIGIN_CERTIFICATE.json"

rng = np.random.default_rng(20261007)
n = 9
# Nondegenerate isolated spectrum kept inside (-pi,pi), so U=exp(-iK) has a fixed log branch.
evals = np.array([-1.35,-0.88,-0.42,-0.10,0.27,0.61,0.93,1.22,1.56], float)
K0 = np.diag(evals)
A = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
K1 = (A + A.conj().T) / (2*n)
B = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
K2 = (B + B.conj().T) / (2*n)
idx = 0
k0 = evals[idx]
psi = np.eye(n, dtype=complex)[:, idx]
Q = np.eye(n) - np.outer(psi, psi.conj())
S = Q @ np.diag([0.0 if j==idx else 1.0/(k0-evals[j]) for j in range(n)]) @ Q
analytic = float(np.real(psi.conj() @ K2 @ psi + 2*psi.conj() @ K1 @ S @ K1 @ psi))

def tracked_eig(s):
    K = K0 + s*K1 + 0.5*s*s*K2
    w, v = np.linalg.eigh(K)
    overlaps = np.abs(v.conj().T @ psi)**2
    return float(w[np.argmax(overlaps)])

h = 2e-4
fd = (tracked_eig(h)-2*tracked_eig(0.0)+tracked_eig(-h))/(h*h)
err = abs(fd-analytic)

# Cubic BCC nearest-neighbour body diagonals: quadratic tensor must be isotropic.
vecs=[]
for sx in (-1,1):
    for sy in (-1,1):
        for sz in (-1,1):
            vecs.append(np.array([sx,sy,sz],float)/math.sqrt(3.0))
vecs=np.array(vecs)
M2=np.einsum('ni,nj->ij',vecs,vecs)/len(vecs)
isotropy_err=float(np.linalg.norm(M2-np.eye(3)/3,ord='fro'))

# Protecting-gap metadata from the existing QICT Kato checkpoint; this demonstrates
# that the package already carries a nontrivial isolated-state computation, but the
# metric-source derivatives required for physical gamma_R are a separate calculation.
common_path = ROOT / "results" / "COMMON_CONTRACTION_CHARACTERIZATION.json"
checkpoint={}
if common_path.exists():
    d=json.loads(common_path.read_text())
    checkpoint=d.get('strict_kato_checkpoint',{})

cert = {
  "purpose":"Validate the exact finite-dimensional Kato second-derivative identity and BCC quadratic isotropy used in the spectral origin theorem.",
  "kato_identity":{
    "analytic_second_derivative":analytic,
    "finite_difference_second_derivative":fd,
    "absolute_error":err,
    "step":h,
    "pass": bool(err < 2e-7)
  },
  "bcc_quadratic_isotropy":{
    "second_moment":M2.tolist(),
    "frobenius_error_from_I_over_3":isotropy_err,
    "pass": bool(isotropy_err < 1e-14)
  },
  "existing_qict_kato_checkpoint":checkpoint,
  "scientific_scope":{
    "proved_by_manuscript":"gamma_R and epsilon_0 are spectral functionals of the source-deformed BCC update under the stated gap/locality/common-net hypotheses.",
    "not_claimed":"No numerical physical value of gamma_R or epsilon_0 is inferred from this self-test or from the checkpoint gap.",
    "required_for_numerical_prediction":"Evaluate the metric-source derivatives of the full interacting BCC update and take the declared thermodynamic/common-regulator limits."
  }
}
OUT.write_text(json.dumps(cert,indent=2))
print(json.dumps(cert,indent=2))
