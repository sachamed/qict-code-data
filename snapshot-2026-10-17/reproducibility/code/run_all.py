#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.metadata, json, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
P = ROOT/'data/MODEL_PARAMETERS.json'
R = ROOT/'results'
R.mkdir(parents=True, exist_ok=True)
GENERATED = [
    'POSITIVE_TAPER_VALIDATION.json', 'INEXACT_MOMENT_POSITIVITY_REPAIR.json', 'MEASURE_IR_VALIDATION.json',
    'MODEL_PARAMETER_VALIDATION.json',
    'PRIMITIVE_DERIVATION_VALIDATION.json', 'PRIMITIVE_ROTOR_CONVERGENCE.csv',
    'REDUCED_LEPTON_INVARIANTS.json', 'CERTIFIED_REDUCED_LEPTON_INTERVALS.json', 'SU2_ROTOR_CONVERGENCE.csv',
    'SU2_TENSOR_AUDIT.json', 'U1_SECTOR_VALIDATION.json', 'SU3_LOCAL_TENSORS.npz', 'SU3_REPRESENTATION_TAILS.csv',
    'SU3_VALIDATION.json', 'L24_PRECONTRACTION_VALIDATION.json', 'RIESZ_NUMERICAL_VALIDATION.json',
    'SPECTRAL_BENCHMARK.npz', 'SPECTRAL_METHOD_VALIDATION.json', 'C11_NUMERICAL_VALIDATION.json',
    'EXACT_IDENTITY_VALIDATION.json', 'SECTOR_OBSERVABLE_MAP.json', 'PHOTON_SIGNATURE_MOMENTS.json', 'FERMION_OBSERVABLE_VALIDATION.json', 'SYMBOLIC_PROOF_CHECKS.json', 'COMMON_CONTRACTION_CHARACTERIZATION.json', 'RESULT_STATUS_REGISTRY.csv', 'RUN_MANIFEST.json', 'EXACT_MATRIX_MOMENT_FLAT_EXTENSION_SELFTEST.json', 'CONTINUUM_CONSISTENCY_CERTIFICATE.json', 'CONTINUUM_CONSISTENCY_MATRIX.csv', 'RG_CONTINUUM_CERTIFICATE.json', 'QICT_4D_RESPONSE_SELFTEST.json', 'SPECTRAL_RESPONSE_IDENTITY_CERTIFICATE.json', 'SPECTRAL_GRAVITY_SELFTEST_EXTRACTION.json', 'CONTOUR_ORDERING_CERTIFICATE.json'
]
# Checkpointed runner preserves all previously certified outputs during resumption.
# Old destructive cleanup is intentionally disabled.

calls = [
    [PYTHON, str(ROOT/'code/validate_model_parameters.py'), '--parameters', str(P), '--out', str(R/'MODEL_PARAMETER_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/derive_primitive_rotor.py'), '--parameters', str(P), '--outdir', str(R)],
    [PYTHON, str(ROOT/'code/compute_reduced_lepton.py'), '--parameters', str(P), '--outdir', str(R)],
    [PYTHON, str(ROOT/'code/validate_u1_sector.py'), '--input', str(ROOT/'data/common_physical_chain/U1_B0_RIESZ_KATO_COMPONENTS.npz'), '--out', str(R/'U1_SECTOR_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/audit_su2_tensors.py'), '--sequential', str(ROOT/'data/su2/SU2_L24_SEQUENTIAL_TENSORS.zip'), '--cluster', str(ROOT/'data/su2/SU2_L24_FOUR_PLAQUETTE_STATE.zip'), '--out', str(R/'SU2_TENSOR_AUDIT.json')],
    [PYTHON, str(ROOT/'code/compute_su3_color.py'), '--parameters', str(P), '--outdir', str(R)],
    [PYTHON, str(ROOT/'code/validate_l24_precontraction_payload.py'), '--archive', str(ROOT/'data/l24_precontraction/L24_RANK6_PRIMARY_DATA.zip'), '--out', str(R/'L24_PRECONTRACTION_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/validate_riesz.py'), '--out', str(R/'RIESZ_NUMERICAL_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/validate_spectral_methods.py'), '--outdir', str(R)],
    [PYTHON, str(ROOT/'code/validate_c11_suite.py'), '--benchmark', str(R/'SPECTRAL_BENCHMARK.npz'), '--out', str(R/'C11_NUMERICAL_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/validate_exact_identities.py'), '--out', str(R/'EXACT_IDENTITY_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/build_sector_observable_map.py'), '--parameters', str(P), '--out', str(R/'SECTOR_OBSERVABLE_MAP.json')],
    [PYTHON, str(ROOT/'code/compute_photon_signature_moments.py'), '--out', str(R/'PHOTON_SIGNATURE_MOMENTS.json')],
    [PYTHON, str(ROOT/'code/validate_fermion_observable_maps.py'), '--out', str(R/'FERMION_OBSERVABLE_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/symbolic_proof_checks.py'), '--out', str(R/'SYMBOLIC_PROOF_CHECKS.json')],
    [PYTHON, str(ROOT/'code/evaluate_common_contraction.py')],
]

# Continuum and cross-sector validation on the retained canonical data.
calls += [
    [PYTHON, str(ROOT/'code/validate_continuum_consistency.py'), '--out', str(R/'CONTINUUM_CONSISTENCY_CERTIFICATE.json'), '--matrix', str(R/'CONTINUUM_CONSISTENCY_MATRIX.csv')],
    [PYTHON, str(ROOT/'code/validate_rg_continuum_specialization.py'), '--out', str(R/'RG_CONTINUUM_CERTIFICATE.json')],
    [PYTHON, str(ROOT/'code/validate_4d_response_identities.py')],
    [PYTHON, str(ROOT/'code/validate_contour_ordering.py')],
    [PYTHON, str(ROOT/'code/validate_spectral_response_identities.py')],
    [PYTHON, str(ROOT/'code/compute_spectral_gravity_response.py'), str(ROOT/'data/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz'), '--output', str(R/'SPECTRAL_GRAVITY_SELFTEST_EXTRACTION.json')],
]
# Regenerate finite reference coordinates and their result-status registries.
calls += [
 [PYTHON,str(ROOT/'code/compute_sector_predictions.py'),'--parameters',str(P),'--out',str(R/'SECTOR_PREDICTIONS.json')],
 [PYTHON,str(ROOT/'code/compute_flavor_diagnostics.py'),'--benchmarks',str(ROOT/'data/FINITE_FLAVOR_BENCHMARKS.json'),'--references',str(ROOT/'data/EXPERIMENTAL_REFERENCE_VALUES_2026.json'),'--out',str(R/'FLAVOR_DIAGNOSTICS.json')],
 [PYTHON,str(ROOT/'code/compute_decisive_sector_tests.py'),'--benchmarks',str(ROOT/'data/FINITE_FLAVOR_BENCHMARKS.json'),'--references',str(ROOT/'data/EXPERIMENTAL_REFERENCE_VALUES_2026.json'),'--out',str(R/'FLAVOR_COORDINATE_COMPARISON.json')],
 [PYTHON,str(ROOT/'code/compute_photon_observational_bounds.py'),'--references',str(ROOT/'data/EXPERIMENTAL_REFERENCE_VALUES_2026.json'),'--out',str(R/'PHOTON_OBSERVATIONAL_BOUNDS.json')],
 [PYTHON,str(ROOT/'code/derive_complete_mass_coordinates.py')],
 [PYTHON,str(ROOT/'code/build_full_rg_and_chi2_audit.py')],
 [PYTHON,str(ROOT/'code/validate_response_bounds.py')],
]
calls += [
 [PYTHON,str(ROOT/'code/execute_physical_g2_contract_audit.py'),'--input-root',str(ROOT/'data/physical_g2_inputs'),'--out',str(R/'physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json')],
 [PYTHON,str(ROOT/'code/validate_rg_nonperturbative_specialization.py')],
 [PYTHON,str(ROOT/'code/validate_qict_4d_completion.py')],
 [PYTHON,str(ROOT/'code/validate_spectral_gravity_origin.py')],
 [PYTHON,str(ROOT/'code/physical_source_boundary_provider.py'),'--out',str(R/'physical_g2_execution/BOUNDARY_PROVIDER_ALGORITHM_SELFTEST.json')],
 [PYTHON,str(ROOT/'code/matrix_free_riesz_kato_physical.py'),'--self-test','--out',str(R/'physical_g2_execution/MATRIX_FREE_RIESZ_KATO_BACKEND_SELFTEST.json')],
 [PYTHON,str(ROOT/'code/certify_su2_rotor.py')],
 [PYTHON,str(ROOT/'code/certify_reduced_lepton_intervals.py'),'--rotor',str(R/'CERTIFIED_SU2_ROTOR.json'),'--out',str(R/'CERTIFIED_REDUCED_LEPTON_INTERVALS.json')],
 [PYTHON,str(ROOT/'code/validate_visible_thresholds.py')],
 [PYTHON,str(ROOT/'code/build_prediction_registry.py'),'--results',str(R),'--out',str(R/'FINAL_PREDICTION_REGISTRY.csv')],
 [PYTHON,str(ROOT/'code/build_result_status_registry.py'),'--results',str(R),'--out',str(R/'RESULT_STATUS_REGISTRY.csv')],
]
calls.append([PYTHON, str(ROOT/'code/analytical_certificates/receiver_window_and_weyl.py')])
calls.append([PYTHON, str(ROOT/'code/validate_positive_taper.py')])
calls.append([PYTHON, str(ROOT/'code/certify_inexact_matrix_moments.py')])
calls.append([PYTHON, str(ROOT/'code/certify_identifiability_and_coherent_obstruction.py')])
calls.append([PYTHON, str(ROOT/'code/certify_matter_gravity_bridge.py')])
calls.append([PYTHON, str(ROOT/'code/certify_reduced_operator_dilation.py')])
calls.append([PYTHON, str(ROOT/'code/verify_route_alpha_origin.py')])
calls.append([PYTHON, str(ROOT/'code/g2_folded_cone.py')])
calls.append([PYTHON, str(ROOT/'code/plan_g2_causal_contraction.py')])
calls.append([PYTHON, str(ROOT/'code/validate_measure_ir.py')])
calls.append([PYTHON, str(ROOT/'code/validate_axiomatic_descent.py')])
calls.append([PYTHON, str(ROOT/'code/verify_parameter_identifiability.py')])
calls.append([PYTHON, str(ROOT/'code/verify_forty_target_identities.py')])
calls.append([PYTHON, str(ROOT/'code/verify_metrological_closure.py')])
# Refresh the extended result certificates. These were previously retained as historical
# outputs although their producers were present in the release.  This makes each
# current result auditable under the same manifest and execution entry point.
for extra in [
    'verify_finite_moment_selfenergy.py',
    'verify_family_resolvent_and_flavor.py',
    'verify_flavor_correspondence_bounds.py',
    'validate_metric_source_orbit.py',
    'verify_overlap_tensor_criteria.py',
    'validate_tt_cubic_tensor.py',
    'validate_tt_extrapolation.py',
    'validate_unified_closure.py',
    'analytical_finite_depth_checks.py',
    'validate_claim_readiness.py',
    'verify_receiver_symmetry_and_limits.py',
]:
    calls.append([PYTHON, str(ROOT/'code'/extra)])
calls.append([PYTHON, str(ROOT/'code/certify_coherent_second_moment_ball.py')])
# Run the same ordered computation list through an atomic, resumable executor.
from qict_resumable_runner import main as _run_checkpointed
if __name__ == '__main__':
    _run_checkpointed(calls, ROOT, R)
