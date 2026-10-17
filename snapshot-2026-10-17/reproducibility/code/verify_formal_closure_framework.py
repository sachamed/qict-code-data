#!/usr/bin/env python3
"""Exact and independent tests of the proposed Haar/Kato closure lemmas.

Actual QICT seed records provide the extensive counterexample. Small
matrices illustrate logical implications only and are explicitly labelled.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, math, tempfile
import numpy as np
from physical_source_certified_causal_cone_rank6 import (
    error_budget, KatoBufferRecord, character_cutoffs, _tail_exact)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility/results/formal_framework'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    seedpath = ROOT / 'reproducibility/results/global_primitive_seed/GLOBAL_A7_KATO_SEED_CERTIFICATE.json'
    seed = json.loads(seedpath.read_text())
    dp = seedpath.parent / 'GLOBAL_A7_FIRST_DERIVATIVE_LOOP_DATA.npz'
    assert hashlib.sha256(dp.read_bytes()).hexdigest() == seed['data_sha256']
    with np.load(dp,allow_pickle=False) as z:
        records = z['loop_amplitude_records']
    count = seed['plaquettes_used']
    norm2 = sum((F(int(num),int(den))**2 for p,g,num,den,e in records),F(0))
    variance = sum(((F(int(num),int(den))*int(e))**2
                    for p,g,num,den,e in records),F(0))
    assert len(records) == 5*count
    assert norm2 == F(629*count,288) == F(seed['ground_vector_derivative_norm_squared_rational'])
    assert variance == 336*count
    cubic = []
    for L in (2,4,8,16,32,64):
        N = 3*L*L*(L+1)
        boundary = 6*L*L
        ratio = F(boundary,N)
        assert ratio == F(2,L+1)
        cubic.append({'side_length':L,'plaquettes':N,'boundary_faces':boundary,
            'boundary_volume_ratio_rational':str(ratio),
            'N_times_ratio_rational':str(N*ratio)})
    # A common Hilbert space and a fixed rank-one test functional suffice
    # to disprove convergence inferred solely from a uniform norm bound.
    Q = np.diag([0.,1.])
    oscillating = [float(((-1)**N*Q)[1,1]) for N in range(2,8)]
    assert oscillating == [1.,-1.,1.,-1.,1.,-1.]
    X = np.array([[0,1],[1,0]],complex)
    Y = np.array([[0,-1j],[1j,0]],complex)
    P = np.diag([1.,0.])
    R = Q.copy()
    product = R @ X
    response = -(R @ X @ P + P @ X @ R)
    assert np.linalg.norm(product-product.conj().T) > 1
    assert np.array_equal(response,response.conj().T)
    # The physical field kernel is positive as a TWO-TIME Gram matrix.
    # Making its first column Toeplitz need not preserve positivity.
    fields = []
    for t in range(3):
        fields.append(np.linalg.matrix_power(Y,-t) @ np.linalg.matrix_power(X,t) @ P)
    K = np.array([[np.trace(a.conj().T @ b) for b in fields] for a in fields])
    echo = K[:,0]
    T = np.array([[echo[j-i] if j>=i else echo[i-j].conjugate()
                   for j in range(3)] for i in range(3)])
    assert np.allclose(echo,[1,1j,1])
    assert np.allclose(np.linalg.eigvalsh(T),[-1,2,2])
    assert np.linalg.eigvalsh(K).min() > -1e-12
    rho9 = (P + X @ P @ X)/2
    assert np.array_equal(rho9,np.eye(2)/2)
    assert np.array_equal(X @ rho9 @ X,rho9)
    # A sqrt(rho) moment is not uniformly Lipschitz in trace distance.
    eps = F(1,10**8)
    sigma = np.diag([1-float(eps),float(eps)])
    sq = np.diag(np.sqrt(np.diag(sigma)))
    fixed_difference = float(abs(np.trace(sq @ X @ sq @ X)))
    matrix_frobenius = math.sqrt(6)*fixed_difference
    legacy_bound = 12*float(eps)
    corrected = error_budget(0,float(eps))['buffer_fro_bound']
    assert matrix_frobenius > legacy_bound and matrix_frobenius <= corrected
    tail_tests = []
    for x,N in [(F(1,10),0),(F(2,5),3),(F(16,20),7),(F(2),2)]:
        exact = _tail_exact(x,N)
        partial = sum((x**k/F(math.factorial(k)) for k in range(N+1,N+81)),F(0))
        assert exact >= partial > 0
        tail_tests.append({'x':str(x),'N':N,'positive_tail_majorant_rational':str(exact)})
    # A positive untyped number cannot become a certified physical gap.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)/'untyped.npz'
        np.savez(p,rho=P,rho_gram_factor=P,buffer=1,
            uniform_contour_gap_lower_bound=1.,riesz_residual_upper_bound=0.)
        assert KatoBufferRecord.load(p).protecting_gap_certified is False
    cutoffs = [character_cutoffs(L,(L-3)//2) for L in (24,32)]
    data = OUT/'FRAMEWORK_COUNTEREXAMPLE_DATA.npz'
    np.savez_compressed(data,two_time_Gram=K,artificial_Toeplitz=T,
        physical_echo=echo,finite_C9_density=rho9,
        oscillating_test_functional=np.array(oscillating))
    result = {
        'status':'PASS_EXACT_AUDIT_WITH_CORRECTED_CONDITIONAL_THEOREMS',
        'actual_QICT_extensive_counterexample':{
            'plaquettes':count,'independent_normalized_loop_records':len(records),
            'free_neutral_electric_gap':8,
            'centered_magnetic_expectation':0,
            'centered_magnetic_vacuum_variance_rational':str(variance),
            'Kato_projector_tangent_operator_norm_squared_rational':str(norm2),
            'per_plaquette_tangent_norm_squared_rational':'629/288',
            'extensive_uniform_operator_bound_implied_by_gap':False,
            'intensive_response_norm_squared_formula':'629/(288*N)',
            'fluctuation_response_norm_squared_formula':'629/288',
            'Haar_twirl_changes_this_invariant_response':False,
            'scope':'Exact electric seed of KE+s*KB on the authentic full serialized gauge support; no extrapolation to physical endpoint s=1.'},
        'three_dimensional_regular_cube_geometry':cubic,
        'geometric_scaling_when_N_counts_plaquettes':'O(N^(-1/3)) for uniformly shaped three-dimensional families; Lipschitz alone is insufficient.',
        'exact_Haar_centering_identity':'T(A-cI)=T(A)-cI for every bounded A and scalar c.',
        'Haar_invariance_of_state_requires_Gauss_not_geometry':True,
        'Kato_operator_role':{
            'reduced_resolvent':'R=Q(H-E0)^(-1)Q; norm<=1/Delta',
            'self_adjoint_projector_response':'S_H[A]=-RAP-PAR for self-adjoint A',
            'response_norm_bound':'norm(S_H[A])=norm(RAP)<=norm(A)/Delta',
            'response_to_scalar_identity':'zero',
            'simple_product_R_A_self_adjoint_in_general':False,
            'projector_response_is_a_dynamical_Hamiltonian':False},
        'bounded_sequence_counterexample':{'synthetic':True,'matrix_elements':oscillating,
            'operator_norm':1,'full_sequence_weak_star_convergence':False,
            'purpose':'Disproves inference of convergence from uniform boundedness alone; not a change to the frozen QICT operator.'},
        'corrected_convergence_hypotheses':['Common separable Hilbert space with consistent embeddings',
            'Uniform bound on extended self-adjoint operators',
            'Cauchy matrix elements on a common dense subspace'],
        'corrected_convergence_conclusion':'Whole-sequence ultraweak convergence to a bounded self-adjoint operator; without Cauchy data only subsequential compactness.',
        'state_budget_counterexample':{'synthetic':True,'trace_distance_rational':str(eps),
            'fixed_GNS_six_channel_difference_Frobenius':matrix_frobenius,
            'old_linear_12D_bound':legacy_bound,'correct_general_Holder_bound':corrected,
            'source_operator_isometry_assumed':True},
        'physical_echo_counterexample':{'synthetic':True,
            'artificial_Toeplitz_eigenvalues':[-1,2,2],
            'two_time_Gram_positive':True,'finite_C9_restores_stationarity':True},
        'normal_C9_trace_norm_limit_criterion':'Tr(P_point(U)*rho)=1; finite-rank Riesz projection is not required.',
        'typed_gap_loader_rejects_untyped_positive_number':True,
        'rational_tail_checks':tail_tests,
        'representation_preflight_cutoffs':cutoffs,
        'uniform_spatial_endpoint_gap_established':False,
        'Floquet_Haar_finite_Riesz_obstruction_removed_by_Haar_centering':False,
        'gravitational_vacuum_cancellation_derived':False,
        'nontrivial_copy_time_generator_derived':False,
        'full_physical_closure':False,
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (seedpath,dp)},
        'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
    (OUT/'FORMAL_CLOSURE_FRAMEWORK_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'actual_QICT_tangent_norm_squared':str(norm2),
        'actual_QICT_centered_variance':str(variance),
        'false_linear_bound_detected':True,'corrected_theorem_requires_Cauchy_data':True},indent=2))


if __name__=='__main__':
    main()
