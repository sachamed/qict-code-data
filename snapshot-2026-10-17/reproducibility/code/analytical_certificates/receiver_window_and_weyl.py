"""Small-matrix verifications of finite-receiver arc and Weyl-twirl inequalities.

These checks use an explicit synthetic unitary; they do not construct the
interacting QICT G2 input matrices or certify a physical pole mass.
"""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20261010)
phases = np.array([-2.7, -1.8, -0.9, 0.1, 0.5, 1.1, 1.9, 2.7])
U = np.diag(np.exp(1j * phases))
A = rng.standard_normal((8, 6)) + 1j * rng.standard_normal((8, 6))
V, _ = np.linalg.qr(A)
C = {n: V.conj().T @ np.linalg.matrix_power(U, n) @ V for n in range(1, 15)}
center, halfwidth = 0.3, 0.85
c = np.cos(halfwidth)
B = (np.exp(-1j * center) * C[1] + np.exp(1j * center) * C[1].conj().T) / 2
mask = np.abs(np.angle(np.exp(1j * (phases - center)))) <= halfwidth
M = V.conj().T @ np.diag(mask.astype(float)) @ V
lo = (B - c * np.eye(6)) / (1-c)
hi = (B + np.eye(6)) / (1+c)
arc_low_margin = float(np.linalg.eigvalsh(M - lo).min())
arc_high_margin = float(np.linalg.eigvalsh(hi - M).min())
assert arc_low_margin > -1e-12 and arc_high_margin > -1e-12

# Certified moment perturbation envelope; exact actual error is explicitly measured.
E = rng.standard_normal((6, 6)) + 1j*rng.standard_normal((6, 6))
E *= 1e-5/np.linalg.norm(E, 2)
Chat = C[1] + E
eta = np.linalg.norm(E, 2)
Bhat = (np.exp(-1j*center)*Chat + np.exp(1j*center)*Chat.conj().T)/2
lo_noisy = (Bhat - (c+eta)*np.eye(6))/(1-c)
hi_noisy = (Bhat + (1+eta)*np.eye(6))/(1+c)
assert np.linalg.eigvalsh(M - lo_noisy).min() > -1e-12
assert np.linalg.eigvalsh(hi_noisy - M).min() > -1e-12

r, N, theta = 0.55, 5, 0.22
P = (1-r*r)/(1 - 2*r*np.cos(theta - phases) + r*r)
F = V.conj().T @ np.diag(P) @ V
FN = np.eye(6, dtype=complex)
for n in range(1, N+1):
    FN += r**n * (np.exp(-1j*n*theta)*C[n] + np.exp(1j*n*theta)*C[n].conj().T)
poisson_error = float(np.linalg.norm(F-FN,2))
poisson_bound = 2*r**(N+1)/(1-r)
assert poisson_error <= poisson_bound + 1e-12
assert np.linalg.eigvalsh(F).min() > -1e-12

# Exact Feshbach identity and complementary-memory resolvent certificate.
Pproj=V @ V.conj().T
Qproj=np.eye(8)-Pproj
K=C[1]
ell_sq=float(np.linalg.norm(np.eye(6)-K.conj().T@K,2))
z=1.38+0.13j
G=V.conj().T@np.linalg.inv(z*np.eye(8)-U)@V
G0=np.linalg.inv(z*np.eye(6)-K)
Q=Qproj
Sigma=V.conj().T@U@Q@np.linalg.inv(z*np.eye(8)-Q@U@Q)@Q@U@V
feshbach_inverse=np.linalg.inv(z*np.eye(6)-K-Sigma)
feshbach_identity_error=float(np.linalg.norm(feshbach_inverse-G,2))
resolvent_discrepancy=float(np.linalg.norm(G-G0,2))
resolvent_discrepancy_bound=ell_sq/(abs(z)-1)**3
assert feshbach_identity_error < 1e-11
assert resolvent_discrepancy <= resolvent_discrepancy_bound + 1e-12

omega=np.exp(2j*np.pi/3)
X=np.roll(np.eye(3, dtype=complex), 1, axis=0)
Z=np.diag([1, omega, omega**2])
assert np.linalg.norm(Z@X-omega*X@Z)<1e-13
T = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
H = T.conj().T @ T
ex = float(np.linalg.norm(H@X - X@H,2))
ez = float(np.linalg.norm(H@Z - Z@H,2))
T_H = sum(np.linalg.matrix_power(X,a) @ np.linalg.matrix_power(Z,b) @ H @ np.linalg.matrix_power(Z,b).conj().T @ np.linalg.matrix_power(X,a).conj().T for a in range(3) for b in range(3))/9
fro_error = float(np.linalg.norm(T_H-np.trace(H)*np.eye(3)/3,2))
twirl_error = float(np.linalg.norm(H-T_H,2))
twirl_bound = (2/3)*(ex+ez)
spread = float(np.ptp(np.linalg.eigvalsh(H)))
assert fro_error < 1e-12
assert twirl_error <= twirl_bound + 1e-12
assert spread <= 2*twirl_bound + 1e-12

out={
    'domain': 'synthetic_unitary_matrix_verification_only',
    'arc_lower_matrix_psd_min_eigenvalue':arc_low_margin,
    'arc_upper_matrix_psd_min_eigenvalue':arc_high_margin,
    'computed_moment_error':float(eta),
    'poisson_truncation_actual_operator_error':poisson_error,
    'poisson_truncation_proven_upper_bound':poisson_bound,
    'weyl_conjugation_identity_operator_error':fro_error,
    'feshbach_identity_error':feshbach_identity_error,
    'complementary_memory_resolvent_discrepancy':resolvent_discrepancy,
    'complementary_memory_resolvent_upper_bound':resolvent_discrepancy_bound,
    'weyl_twirl_actual_operator_error':twirl_error,
    'weyl_twirl_proven_upper_bound':twirl_bound,
    'family_eigenvalue_spread':spread,
    'family_spread_proven_upper_bound':2*twirl_bound,
    'status':'PASS'
}
output=Path(__file__).parents[2]/'results'/'analytical_certificates'/'receiver_window_and_weyl.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
