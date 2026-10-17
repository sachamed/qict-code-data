#!/usr/bin/env python3
"""Physical two-time fields on the authentic shared-face common regulator.

The old fixed Hilbert--Schmidt GNS moments are retained as comparison
moments. They are not silently substituted for a nonstationary physical
field correlation. Static leptonic source/dual sectors only.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, math, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from certify_exact_resonance_reduction import normalized_witness
from compute_common_cyclic_car_kato import update

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility/results/physical_two_time'
PHI = math.pi / 63
DEPTH = 6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_matrix(neutral, charged):
    index = {tuple(map(int, row)): i for i, row in enumerate(charged)}
    rows, cols, vals = [], [], []
    for j, (a, b, J) in enumerate(neutral):
        a, b, J = int(a), int(b), int(J)
        for c in (J - 1, J + 1):
            if c >= 0:
                rows.append(index[(a, b, c, J, J)])
                cols.append(j)
                vals.append(math.sqrt((c + 1) / (2 * (J + 1))))
    return sparse.csr_matrix((vals, (rows, cols)),
                             shape=(len(charged), len(neutral)))


def weak_update(v, E, M, inverse=False):
    pe = np.exp(-1j * PHI * (E % 126))
    if inverse:
        return pe.conj() * (np.exp(1j * PHI * 64) *
            expm_multiply(-1j * PHI * 16 * M, v, traceA=0))
    return np.exp(-1j * PHI * 64) * expm_multiply(
        1j * PHI * 16 * M, pe * v, traceA=0)


def field_kernel(neutral_seed, neutral_step, apply_source, charged_step):
    neutral = neutral_seed.copy()
    forward = apply_source(neutral_seed)
    fields, echo_checks = [], []
    for t in range(DEPTH + 1):
        sink = apply_source(neutral)
        v = sink.copy()
        for _ in range(t):
            v = charged_step(v, True)
        fields.append(v)
        echo_checks.append(np.vdot(sink, forward))
        print(json.dumps({'time': t, 'norm_error': float(abs(np.vdot(v, v)-1))}),
              flush=True)
        if t != DEPTH:
            neutral = neutral_step(neutral)
            forward = charged_step(forward, False)
    kernel = np.array([[np.vdot(v, w) for w in fields] for v in fields])
    discrepancy = float(np.max(abs(kernel[:, 0] - echo_checks)))
    assert discrepancy < 2e-11
    return kernel, discrepancy


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    wd = ROOT / 'reproducibility/results/charged_su2_theta'
    wp = wd / 'CHARGED_SU2_THETA_SOURCES_AND_RETURNS.npz'
    wcpath = wd / 'CHARGED_SU2_THETA_CERTIFICATE.json'
    wc = json.loads(wcpath.read_text())
    assert sha(wp) == wc['data_sha256']
    with np.load(wp, allow_pickle=False) as z:
        neutral = z['R60_neutral_labels']
        charged = z['R60_labels']
        seed = z['R60_neutral_source']
        old_source = z['R60_source']
        M0 = sparse.csr_matrix((z['R60_neutral_magnetic_csr_data'],
            z['R60_neutral_magnetic_csr_indices'], z['R60_neutral_magnetic_csr_indptr']),
            shape=(len(neutral), len(neutral)))
        Mf = sparse.csr_matrix((z['R60_charged_magnetic_csr_data'],
            z['R60_charged_magnetic_csr_indices'], z['R60_charged_magnetic_csr_indptr']),
            shape=(len(charged), len(charged)))
    O = source_matrix(neutral, charged)
    gram = O.T @ O - sparse.eye(len(neutral), format='csr')
    gram_error = float(abs(gram.data).max()) if gram.nnz else 0.
    intertwining = Mf @ O - O @ M0
    intertwining_error = float(sparse.linalg.norm(intertwining))
    assert gram_error < 1e-14 and intertwining_error < 2e-11
    assert np.linalg.norm(O @ seed - old_source) < 2e-13
    print(json.dumps({'source_isometry_max_error': gram_error,
        'magnetic_intertwining_Frobenius_error': intertwining_error,
        'source_columns_tested': len(neutral)}), flush=True)
    del gram, intertwining
    E0 = np.array([3*int(a)*(int(a)+2)+3*int(b)*(int(b)+2)+int(c)*(int(c)+2)
                   for a,b,c in neutral], dtype=np.int64)
    Ef = np.array([3*int(a)*(int(a)+2)+3*int(b)*(int(b)+2)+int(c)*(int(c)+2)
                   for a,b,c,*rest in charged], dtype=np.int64)
    weak, weak_check = field_kernel(seed, lambda v: weak_update(v,E0,M0),
        lambda v: O @ v, lambda v,inv: weak_update(v,Ef,Mf,inv))
    weak_stationarity = float(np.max(abs(weak[1:,1:] - weak[:-1,:-1])))

    yd = ROOT / 'reproducibility/results/coupled_boundary'
    ycpath = yd / 'COUPLED_BOUNDARY_GNS_CERTIFICATE.json'
    yc = json.loads(ycpath.read_text())
    yp = yd / 'COUPLED_PLAQUETTE_BOUNDARY_MOMENTS.npz'
    with np.load(yp, allow_pickle=False) as z:
        ylabels = z['L3_ground_labels']
    yvalues = normalized_witness(yc['cutoffs'][-1]['ground'])
    L = 504
    k = np.arange(L, dtype=np.int64)
    k = np.where(k < (L+1)//2, k, k-L)
    r, s = k[:,None], k[None,:]
    E = 8*r*r + 8*s*s - 4*r*s
    angle = 2*np.pi*np.arange(L)/L
    PB = np.exp(-1j*PHI*(8-4*np.cos(angle[:,None])-4*np.cos(angle[None,:])))
    yseed = np.zeros((L,L), complex)
    for (a,b), value in zip(ylabels, yvalues):
        yseed[int(a)%L,int(b)%L] = value
    PE0 = np.exp(-1j*PHI*(E%126))
    ykernels, ychecks = [], []
    for q in (-3,-6):
        PEf = np.exp(-1j*PHI*((E+4*q*(r-s)+2*q*q+q)%126))
        kernel, check = field_kernel(yseed, lambda v: update(v,PE0,PB),
            lambda v: v.copy(), lambda v,inv: update(v,PEf,PB,inv))
        ykernels.append(kernel)
        ychecks.append(check)
    left, right = ykernels[0]*weak, ykernels[1]
    kernels = np.zeros((DEPTH+1,DEPTH+1,6,6), complex)
    for a in range(6):
        kernels[:,:,a,a] = left if a < 3 else right
    parity = (1 + (-1.)**(np.arange(DEPTH+1)[:,None]-np.arange(DEPTH+1)[None,:]))/2
    cell = kernels*parity[:,:,None,None]
    block = kernels.transpose(0,2,1,3).reshape(6*(DEPTH+1),6*(DEPTH+1))
    cellblock = cell.transpose(0,2,1,3).reshape(block.shape)
    eigen = float(np.linalg.eigvalsh(block).min())
    celleigen = float(np.linalg.eigvalsh(cellblock).min())
    assert eigen > -2e-11 and celleigen > -2e-11
    diagonal_error = float(max(np.linalg.norm(kernels[t,t]-np.eye(6))
                               for t in range(DEPTH+1)))
    assert diagonal_error < 2e-11
    stationary_defect = float(np.max(abs(kernels[1:,1:] - kernels[:-1,:-1])))
    oldpath = ROOT / 'reproducibility/results/common_coupled_electroweak/COMMON_COUPLED_ELECTROWEAK_GNS_RESOLVENTS.npz'
    with np.load(oldpath, allow_pickle=False) as z:
        old = z['moments'][:DEPTH+1]
    difference = float(np.max(abs(kernels[:,0]-old)))
    pseudo_toeplitz = np.block([[kernels[j-i,0] if j>=i else kernels[i-j,0].conj().T
        for j in range(DEPTH+1)] for i in range(DEPTH+1)])
    pseudo_min = float(np.linalg.eigvalsh(pseudo_toeplitz).min())
    sgpath = ROOT / 'reproducibility/results/coupled_su2_spin_network/COUPLED_SU2_SPIN_NETWORK_CERTIFICATE.json'
    sg = json.loads(sgpath.read_text())
    state = F(sg['cutoffs'][-1]['ground_projector_trace_norm_error_upper_rational']) + \
        2*F(yc['cutoffs'][-1]['ground']['state_trace_distance_upper_rational'])
    n = 2*DEPTH
    d = 2*(60+1-16)
    assert 5*d > 8*n
    weak_alias = min(F(2), F(2)**(n+1)*F(22*n,5*d)**d)
    ydegree = L - 2*int(np.max(abs(ylabels)))
    assert 5*ydegree > n
    yalias = min(F(2), F(2)**(n+1)*F(11*n,20*ydegree)**ydegree)
    error = state + weak_alias + yalias
    data = OUT / 'PHYSICAL_TWO_TIME_LEPTON_KERNELS.npz'
    np.savez_compressed(data, physical_kernel=kernels, physical_kernel_with_cell=cell,
        weak_fundamental_kernel=weak, U1_left_kernel=ykernels[0], U1_right_kernel=ykernels[1],
        fixed_HS_comparison_moments=old, source_csr_data=O.data,
        source_csr_indices=O.indices, source_csr_indptr=O.indptr,
        source_csr_shape=np.array(O.shape), times=np.arange(DEPTH+1))
    result = {'status':'PHYSICAL_TWO_TIME_STATIC_LEPTON_KERNELS_ON_AUTHENTIC_SHARED_FACES',
        'actual_plaquettes':[2925,2908], 'source_link':10377,
        'time_range':[0,DEPTH], 'max_magnetic_gate_count_in_closed_word':n,
        'regulator':{'U1_length_per_cycle':L,'SU2_outer_twice_spin':60,
            'SU2_neutral_dimension':len(neutral),'SU2_charged_dimension':len(charged),
            'source_columns_tested':len(neutral)},
        'observable':'K_ab(t,s)=Tr rho U0^(-t) Oa^dagger Uf^(t-s) Ob U0^s',
        'method':'Hilbert--Schmidt Gram of Uf^(-t) Oa U0^t sqrt(rho); factorized sparse propagation.',
        'neutral_color_cancellation_exact':True,
        'color_cancellation_scope':'Same neutral SU3 evolution and identity source in each static leptonic sector; no claim for quarks or moving interactions.',
        'right_weak_cancellation_exact':True,
        'source_isometry_max_diagnostic':gram_error,
        'source_magnetic_intertwining_Frobenius_diagnostic':intertwining_error,
        'independent_forward_inverse_echo_diagnostics':[weak_check]+ychecks,
        'equal_time_I6_error_diagnostic':diagonal_error,
        'full_two_time_block_min_eigenvalue_diagnostic':eigen,
        'cell_two_time_block_min_eigenvalue_diagnostic':celleigen,
        'stationarity_defect_diagnostic':stationary_defect,
        'weak_stationarity_defect_diagnostic':weak_stationarity,
        'difference_from_fixed_HS_comparison_moments_diagnostic':difference,
        'artificial_Toeplitz_min_eigenvalue_diagnostic':pseudo_min,
        'state_trace_norm_error_rational':str(state),
        'weak_closed_return_bound_rational':str(weak_alias),
        'U1_closed_return_bound_rational':str(yalias),
        'per_entry_analytic_state_and_regulator_bound_rational':str(error),
        'per_entry_analytic_state_and_regulator_bound':float(error),
        'full_time_block_operator_error_bound_rational':str((DEPTH+1)*error),
        'bound_scope':'Exact rational seed-state and omitted-return bounds; ordinary floating point propagation is diagnostic, not interval certified.',
        'floating_point_interval_certified':False,
        'state_stationary_under_frozen_Floquet':False,
        'one_variable_physical_resolvent_certified':False,
        'all_occupation_weak_Gauss_CAR_certified':False,
        'spatial_thermodynamic_limit_certified':False,
        'full_moving_C10_completed':False,
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
            (wp,wcpath,yp,ycpath,sgpath,oldpath)},
        'data_sha256':sha(data), 'seconds':time.monotonic()-started}
    (OUT / 'PHYSICAL_TWO_TIME_KERNEL_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','equal_time_I6_error_diagnostic',
        'stationarity_defect_diagnostic','difference_from_fixed_HS_comparison_moments_diagnostic',
        'per_entry_analytic_state_and_regulator_bound','seconds')},indent=2),flush=True)


if __name__ == '__main__':
    main()
