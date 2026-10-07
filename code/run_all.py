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
    'MODEL_PARAMETER_VALIDATION.json',
    'PRIMITIVE_DERIVATION_VALIDATION.json', 'PRIMITIVE_ROTOR_CONVERGENCE.csv',
    'REDUCED_LEPTON_INVARIANTS.json', 'SU2_ROTOR_CONVERGENCE.csv',
    'SU2_TENSOR_AUDIT.json', 'U1_SECTOR_VALIDATION.json', 'SU3_LOCAL_TENSORS.npz', 'SU3_REPRESENTATION_TAILS.csv',
    'SU3_VALIDATION.json', 'L24_PRECONTRACTION_VALIDATION.json', 'RIESZ_NUMERICAL_VALIDATION.json',
    'SPECTRAL_BENCHMARK.npz', 'SPECTRAL_METHOD_VALIDATION.json', 'C11_NUMERICAL_VALIDATION.json',
    'EXACT_IDENTITY_VALIDATION.json', 'SECTOR_PREDICTIONS.json', 'PHOTON_SIGNATURE_MOMENTS.json', 'FERMION_OBSERVABLE_VALIDATION.json', 'FLAVOR_PHENOMENOLOGY_DIAGNOSTICS.json', 'FLAVOR_COORDINATE_COMPARISON.json', 'PHOTON_OBSERVATIONAL_BOUNDS.json', 'SYMBOLIC_PROOF_CHECKS.json', 'COMMON_CONTRACTION_CHARACTERIZATION.json', 'COMPLETE_MASS_COORDINATES.json', 'COMPLETE_MASS_COORDINATES.csv', 'FINAL_PREDICTION_REGISTRY.csv', 'RUN_MANIFEST.json', 'EXACT_MATRIX_MOMENT_FLAT_EXTENSION_SELFTEST.json', 'UNIFIED_CLOSURE_CERTIFICATE.json', 'UNIFIED_CLOSURE_MATRIX.csv', 'RG_NONPERTURBATIVE_CERTIFICATE.json'
]
for name in GENERATED:
    q = R/name
    if q.exists(): q.unlink()

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
    [PYTHON, str(ROOT/'code/compute_sector_predictions.py'), '--parameters', str(P), '--out', str(R/'SECTOR_PREDICTIONS.json')],
    [PYTHON, str(ROOT/'code/compute_photon_signature_moments.py'), '--out', str(R/'PHOTON_SIGNATURE_MOMENTS.json')],
    [PYTHON, str(ROOT/'code/validate_fermion_observable_maps.py'), '--out', str(R/'FERMION_OBSERVABLE_VALIDATION.json')],
    [PYTHON, str(ROOT/'code/compute_flavor_diagnostics.py'), '--benchmarks', str(ROOT/'data/FINITE_FLAVOR_BENCHMARKS.json'), '--references', str(ROOT/'data/EXPERIMENTAL_REFERENCE_VALUES_2026.json'), '--out', str(R/'FLAVOR_PHENOMENOLOGY_DIAGNOSTICS.json')],
    [PYTHON, str(ROOT/'code/compute_decisive_sector_tests.py'), '--benchmarks', str(ROOT/'data/FINITE_FLAVOR_BENCHMARKS.json'), '--references', str(ROOT/'data/EXPERIMENTAL_REFERENCE_VALUES_2026.json'), '--out', str(R/'FLAVOR_COORDINATE_COMPARISON.json')],
    [PYTHON, str(ROOT/'code/compute_photon_observational_bounds.py'), '--references', str(ROOT/'data/EXPERIMENTAL_REFERENCE_VALUES_2026.json'), '--out', str(R/'PHOTON_OBSERVATIONAL_BOUNDS.json')],
    [PYTHON, str(ROOT/'code/symbolic_proof_checks.py'), '--out', str(R/'SYMBOLIC_PROOF_CHECKS.json')],
    [PYTHON, str(ROOT/'code/evaluate_common_contraction.py')],
    [PYTHON, str(ROOT/'code/derive_complete_mass_coordinates.py')],
]

# Physical G2 contract execution and backend checks on the supplied L=24/C9 payloads.
pg2 = R/'physical_g2_execution'
pg2.mkdir(parents=True, exist_ok=True)
for q in pg2.glob('*'):
    if q.is_file(): q.unlink()
calls += [
    [PYTHON, str(ROOT/'code/matrix_free_riesz_kato_physical.py'), '--self-test', '--out', str(pg2/'MATRIX_FREE_RIESZ_KATO_BACKEND_SELFTEST.json')],
    [PYTHON, str(ROOT/'code/physical_source_boundary_provider.py'), '--out', str(pg2/'BOUNDARY_PROVIDER_ALGORITHM_SELFTEST.json')],
    [PYTHON, str(ROOT/'code/execute_physical_g2_contract_audit.py'), '--input-root', str(ROOT/'data/physical_g2_inputs'), '--out', str(pg2/'PHYSICAL_G2_CONTRACT_AUDIT.json')],
    [PYTHON, str(ROOT/'code/validate_unified_closure.py'), '--out', str(R/'UNIFIED_CLOSURE_CERTIFICATE.json'), '--matrix', str(R/'UNIFIED_CLOSURE_MATRIX.csv')],
    [PYTHON, str(ROOT/'code/validate_rg_nonperturbative_specialization.py'), '--out', str(R/'RG_NONPERTURBATIVE_CERTIFICATE.json')],
    [PYTHON, str(ROOT/'code/build_prediction_registry.py'), '--results', str(R), '--out', str(R/'FINAL_PREDICTION_REGISTRY.csv')],
    [PYTHON, str(ROOT/'code/build_full_rg_and_chi2_audit.py')],
]
for call in calls:
    subprocess.run(call, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(base: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for pat in patterns:
        for path in sorted(base.rglob(pat)):
            if path.is_file():
                out[str(path.relative_to(ROOT))] = sha256(path)
    return out

versions = {}
for pkg in ['numpy','scipy','sympy','mpmath']:
    versions[pkg] = importlib.metadata.version(pkg)
manifest = {
    'classification': 'REPRODUCIBLE_CALCULATION_MANIFEST',
    'python': platform.python_version(),
    'package_versions': versions,
    'code_sha256': hash_tree(ROOT/'code', ('*.py','requirements.txt')),
    'primary_data_sha256': hash_tree(ROOT/'data', ('*.json','*.zip','*.npz','*.csv','*.txt')),
    'result_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in sorted(R.rglob('*')) if path.is_file() and path.name != 'RUN_MANIFEST.json'},
}
(R/'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('All research calculations completed successfully.')
