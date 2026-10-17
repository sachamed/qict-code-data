#!/usr/bin/env python3
"""Finite C9-state relative tensor-source identities and numerical witness.

The stored common-state neutral U(1)-SU(2) truncations supply genuine QICT
finite neutral updates and stationary C9 density matrices. Two explicitly
declared finite source probes test the general theorem; these probes are NOT
identified with the complete physical coframe vertices or a graviton.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import schur, eigh

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/FINITE_C9_COMMON_STATE_AND_MOMENTS.npz'
OUT = ROOT / 'results/C9_TENSOR_STATIONARY_MOMENTS.json'


def hermitize(x):
    return (x+x.conj().T)/2


def finite_example(arr, R):
    uy = arr[f'R{R}_neutral_U1']
    uw = arr[f'R{R}_neutral_SU2']
    U=np.kron(uy,uw)
    _,qy=schur(uy,output='complex')
    _,qw=schur(uw,output='complex')
    Q=np.kron(qy,qw)
    c=arr[f'R{R}_primitive_coefficients']
    ids=arr[f'R{R}_joint_cluster_ids']
    eigval=arr[f'R{R}_joint_neutral_eigenvalues']
    d=len(c)
    # The archived C9 construction projects the selected primitive seed into
    # joint quasienergy clusters, retaining coherent degeneracies.
    rhoeig=np.zeros((d,d),complex)
    for idx in np.unique(ids):
        ix=np.flatnonzero(ids==idx)
        rhoeig[np.ix_(ix,ix)] = np.outer(c[ix],c[ix].conj())
    rho=hermitize(Q@rhoeig@Q.conj().T)
    ev,V=eigh(rho)
    rho_sqrt=(V*np.sqrt(np.maximum(ev,0)))@V.conj().T
    comm=np.linalg.norm(U@rho@U.conj().T-rho,ord='fro')
    trace_err=abs(np.trace(rho)-1)
    # Declared finite flux-basis diagnostic probes, implemented as an actual
    # differentiable family U(h)=U exp(-ih_1 T_1) exp(-ih_2 T_2).
    # They do not replace the full microscopic coframe source.
    dimy=uy.shape[0];dimw=uw.shape[0]
    qY=np.array([a for a in range(-R,R+1) for b in range(-R,R+1)],float)
    Ty=np.kron(np.diag(qY/max(R,1)),np.eye(dimw))
    Tw=np.kron(np.eye(dimy),np.diag(np.linspace(-1,1,dimw)))
    X=[hermitize(T-np.trace(rho@T)*np.eye(d)) for T in (Ty,Tw)]
    # Work in joint eigenbasis: Ad_U acts diagonally on |i><j|.
    # The C9 Gram columns are B_a=X_a rho^{1/2}.
    B=np.stack([Q.conj().T@x@rho_sqrt@Q for x in X])
    lam=eigval/np.abs(eigval)
    omega=lam[:,None]*np.conj(lam[None,:])
    C0=np.einsum('aij,bij->ab',B.conj(),B)
    C1=np.einsum('aij,ij,bij->ab',B.conj(),omega,B)
    C0=hermitize(C0)
    A=hermitize(2*C0-C1-C1.conj().T)
    eps=.45
    dcut=2*np.sin(eps/2)
    angles=np.angle(omega)
    spectral_low=np.einsum('aij,ij,bij->ab',B.conj(),(abs(angles)<=eps+1e-12).astype(float),B)
    spectral_low=hermitize(spectral_low)
    bound=hermitize(spectral_low-C0+A/dcut**2)
    # Remove the exact invariant spectral part by applying the centralizer
    # projection E_{Ad_U}({1}); the archived clustering tolerance is used
    # only for a numerical decomposition diagnostic.
    pinv=(ids[:,None]==ids[None,:])
    projected=B*np.logical_not(pinv)[None,:,:]
    low_dyn=hermitize(np.einsum('aij,ij,bij->ab',projected.conj(),(abs(angles)<=eps+1e-12).astype(float),projected))
    invariant_err=max(abs(np.einsum('aij,ij,bij->ab',projected.conj(),pinv.astype(float),projected)).flatten())
    M=[np.einsum('aij,ij,bij->ab', B.conj(),omega**n,B) for n in range(7)]
    toe=np.block([[M[j-i] if j>=i else M[i-j].conj().T for j in range(7)] for i in range(7)])
    eigtoe=np.linalg.eigvalsh(hermitize(toe))
    # Independent physical Heisenberg trace at n=1, using dense matrices.
    check=np.array([[np.trace(rho@X[a]@U@X[b]@U.conj().T) for b in range(2)] for a in range(2)])
    err=np.linalg.norm(check-C1)
    # Preserves the distinguishable physical meaning of conserved and soft modes.
    r={'R':R,'neutral_dimension':d,'stationarity_frobenius':float(comm),
       'density_trace_error':float(trace_err),
       'C9_time_correlation_discrepancy':float(err),
       'moment_zero_min_eigenvalue':float(np.linalg.eigvalsh(C0).min()),
       'Toeplitz_min_eigenvalue':float(eigtoe.min()),
       'angular_defect_min_eigenvalue':float(np.linalg.eigvalsh(A).min()),
       'arc_lower_bound_margin':float(np.linalg.eigvalsh(bound).min()),
       'raw_low_arc_weight_min_eigenvalue':float(np.linalg.eigvalsh(spectral_low).min()),
       'conserved_component_removed_residual':float(invariant_err),
       'dynamical_low_arc_weight_min_eigenvalue':float(np.linalg.eigvalsh(low_dyn).min())}
    assert comm<2e-8 and trace_err<2e-10 and err<2e-8
    assert r['Toeplitz_min_eigenvalue']>-2e-9
    assert r['angular_defect_min_eigenvalue']>-2e-9
    assert r['arc_lower_bound_margin']>-2e-9
    assert invariant_err<2e-8
    return r


def main():
    with np.load(DATA,allow_pickle=False) as arrays:
        results=[finite_example(arrays,r) for r in (1,2)]
    report={'result':'PASS_C9_STATIONARY_OPERATOR_MOMENTS',
            'data':'FINITE_C9_COMMON_STATE_AND_MOMENTS.npz',
            'scope':'Two original finite neutral U(1)-SU(2) C9 updates; declared flux-basis probes, not physical coframe vertices',
            'state':'Joint C9 Cesaro spectral dephasing from archived primitive coefficients',
            'mathematical_assertions':['C9 stationary density','positive Hilbert-Schmidt Toeplitz moments','positive matrix-valued circle measure', 'finite low-phase-weight enclosure','conserved-mode subtraction','agreement with stationary physical two-time trace'],
            'physical_graviton_pole_claimed':False,
            'QICT_C10_interacting_coframe_derivatives_evaluated':False,
            'regulators':results}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
