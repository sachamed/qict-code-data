"""Certified-in-form spectral perturbation inequalities and numerical witnesses.

FINITE MATHEMATICAL MODELS ONLY. Not a QICT interacting provider.
Assumptions: both operators unitary, same Hilbert space, a common contour
Gamma strictly in both resolvent sets with uniform gap g, and source
isometries. For other embeddings first transport into common Hilbert space.
"""
import json
from pathlib import Path
import numpy as np


def hermitian_projector_from_clusters(U, indices):
    w, B = np.linalg.eig(U)
    # Handles simple diagonalizable normal U in examples, not a general
    # eigenvalue-certifying algorithm. Orthogonalize the chosen eigenspace.
    Q, _ = np.linalg.qr(B[:, list(indices)])
    return Q @ Q.conj().T


def quantitative_projector_bound(delta_operator, contour_length, gap):
    if not np.isfinite(delta_operator) or delta_operator < 0:
        raise ValueError('Invalid operator perturbation')
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError('Uniform, certified positive contour gap required')
    if contour_length <= 0:
        raise ValueError('Contour length must be positive')
    return float(contour_length * delta_operator / (2*np.pi*gap**2))


def compressed_residue_bound(delta_operator, source_delta, contour_length, gap):
    """For source isometries V,V', valid for P,P' spectral projections."""
    if source_delta < 0 or source_delta > 2:
        raise ValueError('Invalid isometry difference')
    return quantitative_projector_bound(delta_operator, contour_length, gap) + 2*source_delta


def regression(epsilon=0.02, m=3, radius=0.5):
    """Unitary pair, spectral gap and source compression with six channels."""
    if not 0 < epsilon < 0.2: raise ValueError('epsilon must be small positive')
    if not 0 < radius < 1: raise ValueError('Contour radius invalid')
    I6=np.eye(6, dtype=complex)
    I_m=np.eye(m, dtype=complex)
    U0=np.kron(I_m, np.kron(np.diag([1, 1j]), I6))
    rotation=np.array([[np.cos(epsilon), -np.sin(epsilon)],
                       [np.sin(epsilon), np.cos(epsilon)]], dtype=complex)
    W=np.kron(I_m, np.kron(rotation, I6))
    U1=W@U0@W.conj().T
    P0=np.kron(I_m, np.kron(np.diag([1, 0]), I6))
    P1=W@P0@W.conj().T
    V=np.zeros((12*m,6),complex)
    for j in range(m): V[12*j:12*j+6,:]=I6/np.sqrt(m)
    # Gamma is a circle around z=1: gap to eigenvalue 1 is r;
    # gap to eigenvalue i is |1-i|-r (minimum over contour).
    gap=min(radius, np.sqrt(2)-radius)
    operator_delta=float(np.linalg.norm(U0-U1,2))
    projector_delta=float(np.linalg.norm(P0-P1,2))
    upper=quantitative_projector_bound(operator_delta,2*np.pi*radius,gap)
    R0=V.conj().T@P0@V
    R1=V.conj().T@P1@V
    compressed_difference=float(np.linalg.norm(R0-R1,2))
    compressed_upper=compressed_residue_bound(operator_delta,0.,2*np.pi*radius,gap)
    lower=float(np.linalg.eigvalsh(R1)[0])
    # Stable spectral excitation in this non-QICT model, no mass prediction.
    assert projector_delta <= upper+1e-11
    assert compressed_difference <= compressed_upper+1e-11
    assert lower >= 1-compressed_upper-1e-11
    assert np.linalg.norm(U0.conj().T@U0-np.eye(12*m))<1e-12
    assert np.linalg.norm(U1.conj().T@U1-np.eye(12*m))<1e-12
    assert np.linalg.norm(V.conj().T@V-I6)<1e-12
    return {'classification':'NON_QICT_FINITE_SPECTRAL_PERTURBATION_WITNESS',
            'epsilon':epsilon,'haar_grid_m':m,'common_gap':gap,
            'contour_radius':radius, 'operator_difference_norm':operator_delta,
            'projector_difference_norm':projector_delta,
            'rigorous_projector_difference_upper':upper,
            'six_source_compressed_residue_difference':compressed_difference,
            'rigorous_compressed_residue_difference_upper':compressed_upper,
            'smallest_compressed_residue_eigenvalue':lower,
            'residue_lower_from_perturbation':max(0.,1-compressed_upper),
            'physical_qict_pole_certified':False}

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result={'theorem':'||P(U)-P(U_prime)|| <= length(Gamma)/(2*pi*g^2) ||U-U_prime||, with Gamma uniformly g away from both spectra',
            'source_bound':'||V*PV-V_prime*P_prime*V_prime|| <= ||P-P_prime||+2||V-V_prime||',
            'models':[regression(epsilon=e,m=m) for m,e in [(3,0.02),(5,0.04),(7,0.07)]]}
    (root/'results/QUANTITATIVE_SPECTRAL_STABILITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'test_cases':len(result['models']),
            'max_operator_norm':max(x['operator_difference_norm'] for x in result['models']),
            'min_residue_eigenvalue':min(x['smallest_compressed_residue_eigenvalue'] for x in result['models'])},indent=2))
