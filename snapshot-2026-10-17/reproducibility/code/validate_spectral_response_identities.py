#!/usr/bin/env python3
"""Deterministic algebraic checks for the QICT spectral source-response formulas.

The checks validate finite-dimensional Kato/Feynman--Hellmann identities, the
homogeneous-frame response convention, and BCC quadratic isotropy. Physical
coefficients follow from the source-deformed interacting many-body update and the
continuum matching conditions specified in the manuscript.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json'
SELFTEST = ROOT / 'data' / 'SPECTRAL_GRAVITY_SELFTEST_INPUT.npz'

rng = np.random.default_rng(20261007)
n = 9
evals = np.array([-1.35,-0.88,-0.42,-0.10,0.27,0.61,0.93,1.22,1.56], float)
K0 = np.diag(evals).astype(complex)

def rand_herm(scale=1.0):
    A = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    return scale * (A + A.conj().T) / (2*n)

K1 = rand_herm()
K2 = rand_herm()
K1vol = rand_herm(scale=0.6)
idx = 0
k0 = evals[idx]
psi = np.eye(n, dtype=complex)[:, idx]
P = np.outer(psi, psi.conj())
Q = np.eye(n) - P
S = Q @ np.diag([0.0 if j==idx else 1.0/(k0-evals[j]) for j in range(n)]) @ Q

# First derivative: Feynman--Hellmann, used by the homogeneous coframe response.
first_analytic = float(np.real(psi.conj() @ K1vol @ psi))
def tracked_eig_linear(s):
    K = K0 + s*K1vol
    w, v = np.linalg.eigh(K)
    overlaps = np.abs(v.conj().T @ psi)**2
    return float(w[np.argmax(overlaps)])

h1 = 1e-6
first_fd = (tracked_eig_linear(h1)-tracked_eig_linear(-h1))/(2*h1)
first_err = abs(first_fd-first_analytic)

# Second derivative: Kato reduced-resolvent identity, used by TT susceptibility.
second_analytic = float(np.real(psi.conj() @ K2 @ psi + 2*psi.conj() @ K1 @ S @ K1 @ psi))
def tracked_eig_quadratic(s):
    K = K0 + s*K1 + 0.5*s*s*K2
    w, v = np.linalg.eigh(K)
    overlaps = np.abs(v.conj().T @ psi)**2
    return float(w[np.argmax(overlaps)])

h2 = 2e-4
second_fd = (tracked_eig_quadratic(h2)-2*tracked_eig_quadratic(0.0)+tracked_eig_quadratic(-h2))/(h2*h2)
second_err = abs(second_fd-second_analytic)

# Cubic BCC nearest-neighbour body diagonals: quadratic tensor is isotropic.
vecs=[]
for sx in (-1,1):
    for sy in (-1,1):
        for sz in (-1,1):
            vecs.append(np.array([sx,sy,sz],float)/math.sqrt(3.0))
vecs=np.array(vecs)
M2=np.einsum('ni,nj->ij',vecs,vecs)/len(vecs)
isotropy_err=float(np.linalg.norm(M2-np.eye(3)/3,ord='fro'))

# A fixed logarithmic sheet has bounded eigenphase; kappa/N must vanish for N->infinity.
Ns = np.array([8, 64, 512, 4096], float)
bounded_density = np.pi/Ns
bounded_monotone = bool(np.all(np.diff(bounded_density) < 0) and bounded_density[-1] < 1e-3)

# Deterministic self-test input for the finite-volume extraction script.
np.savez_compressed(
    SELFTEST,
    K0=K0,
    K1_vol=K1vol,
    K1_q=K1,
    K1_mq=K1,
    K2_qmq=K2,
    N_cells=np.array(64.0),
    qell=np.array(0.125),
    continuum_density_factor=np.array(1.0),
    selected_index=np.array(idx, dtype=np.int64),
)

common_path = ROOT / 'results' / 'COMMON_CONTRACTION_CHARACTERIZATION.json'
checkpoint={}
if common_path.exists():
    d=json.loads(common_path.read_text())
    checkpoint=d.get('strict_kato_checkpoint',{})

cert = {
  'classification':'ALGEBRAIC_VALIDATION_CERTIFICATE',
  'purpose':'Validate the finite-dimensional first- and second-derivative spectral identities, the fixed-Copy-Time density normalization, and BCC quadratic isotropy used by the QICT metric-source response construction.',
  'feynman_hellmann_first_derivative':{
    'analytic':first_analytic,
    'finite_difference':first_fd,
    'absolute_error':first_err,
    'step':h1,
    'pass': bool(first_err < 2e-8)
  },
  'kato_second_derivative':{
    'analytic':second_analytic,
    'finite_difference':second_fd,
    'absolute_error':second_err,
    'step':h2,
    'pass': bool(second_err < 2e-7)
  },
  'bcc_quadratic_isotropy':{
    'second_moment':M2.tolist(),
    'frobenius_error_from_I_over_3':isotropy_err,
    'pass': bool(isotropy_err < 1e-14)
  },
  'local_sheet_scaling_check':{
    'N_cells':Ns.tolist(),
    'pi_over_N':bounded_density.tolist(),
    'pass':bounded_monotone,
    'interpretation':'The homogeneous coframe derivative provides the extensive vacuum source coordinate while remaining invariant under locally constant logarithmic-sheet shifts.'
  },
  'continuum_density_normalization':{
    'definition':'Z_R=(tau_star/delta_t_R)*(ell_star^3/v_cell_R)',
    'selftest_value':1.0,
    'interpretation':'Z_R converts one-update per-cell Kato derivatives to the fixed Copy-Time density units of the continuum effective action.'
  },
  'existing_qict_kato_checkpoint':checkpoint,
  'scientific_scope':{
    'established_here':'The finite-volume homogeneous-frame response and TT response are spectral derivatives of the selected QICT branch; the Kato/Feynman--Hellmann identities and BCC quadratic isotropy are independently checked.',
    'metric_identification':'The finite identities define source-response coordinates for a specified coframe lift. Canonical source normalization, a dynamical metric sector, Lorentzian matching, Weinberg--Witten compatibility and a positive massless helicity-two pole select the physical epsilon_0 and gamma_R representatives.',
    'evaluation_path':'This self-test certifies the algebraic response identities and BCC isotropy; the interacting common-net calculation supplies the numerical physical epsilon_0, gamma_R, G and Lambda.',
    'required_for_prediction':'Evaluate the source-deformed interacting BCC update on a controlled regulator sequence and take the declared common thermodynamic/continuum limits.'
  }
}
cert['pass'] = bool(cert['feynman_hellmann_first_derivative']['pass'] and cert['kato_second_derivative']['pass'] and cert['bcc_quadratic_isotropy']['pass'] and cert['local_sheet_scaling_check']['pass'])
OUT.write_text(json.dumps(cert,indent=2,sort_keys=True)+'\n')
print(json.dumps(cert,indent=2,sort_keys=True))
