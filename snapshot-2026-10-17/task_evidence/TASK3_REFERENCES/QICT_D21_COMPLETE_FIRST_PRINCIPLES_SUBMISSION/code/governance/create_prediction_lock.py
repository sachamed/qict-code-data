#!/usr/bin/env python3
"""Create a deterministic QICT precomparison specification digest.

The digest covers the theory source, prediction algorithms, numerical tolerances,
and software environment specification. Prospective held-out comparison uses
an independent immutable timestamp for the resulting JSON and SHA-256 digest.
"""
from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'preregistration'/'QICT_PREDICTION_LOCK.json'
SHA=ROOT/'preregistration'/'QICT_PREDICTION_LOCK.sha256'
FILES=[
 'manuscript/QICT_MAIN_MANUSCRIPT.tex',
 'DOCUMENT_INDEX.json',
 'requirements.txt',
 'preregistration/FAILURE_CRITERIA.json',
 'code/predictions/mass_spectrum_from_g2_edges.py',
 'code/predictions/mixing_from_interacting_vertices.py',
 'code/closure/verify_g2_convergence_protocol.py',
 'code/closure/verify_regulator_point_prediction_criterion.py',
 'code/closure/verify_g3_mixing_gate.py',
 'code/closure/verify_g4_mass_ratio_gate.py',
 'code/closure/verify_c11_full_family_projector_invariant.py',
 'code/closure/verify_c11_family_current_observable.py',
 'code/closure/verify_c11_receiver_vertex_matching.py',
 'code/closure/verify_c11_family_mass_ratio_theorem.py',
 'code/finite_representatives/verify_complementary_path_kernel.py',
 'supplements/QICT_Supplement_L_C11_Intrinsic_Observable_Completion.tex',
 'supplements/QICT_Supplement_V_Physical_Mass_and_Mixing_Definitions.tex',
 'supplements/QICT_Supplement_X_Assumptions_Global_Gauge_Consistency_and_Convergence.tex',
 'data/family_transfer_matrices.csv',
 'data/c11_family_current_observables.csv',
 'code/governance/verify_parameter_free_data_firewall.py',
 'code/photon_metrology/verify_bcc_scale_free_directional_prediction.py',
 'code/photon_metrology/verify_bcc_quartic_directional_prediction.py',
 'code/photon_metrology/verify_bcc_cubic_harmonic_fingerprint.py',
 'validation_addendum/feshbach_krylov_closure/code/verify_feshbach_krylov_closure.py',
 'validation_addendum/feshbach_krylov_closure/code/verify_local_sparse_feshbach_krylov.py',
 'validation_addendum/feshbach_krylov_closure/QICT_FESHBACH_KRYLOV_CLOSURE_NOTE.tex',
 'preregistration/QICT_BCC_SEALED_PREDICTION.md',
]
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
entries=[]
for rel in FILES:
    p=ROOT/rel
    if not p.exists(): raise FileNotFoundError(rel)
    entries.append({'path':rel,'sha256':sha256(p),'bytes':p.stat().st_size})
canonical=''.join(f"{x['path']}\0{x['sha256']}\n" for x in entries).encode()
payload=hashlib.sha256(canonical).hexdigest()
obj={
 'theory':'Quantum Information Copy Time (QICT)',
 'status':'FIXED_LOCAL_SPECIFICATION_DIGEST',
 'scope':'code + theory source + algorithms + tolerances, separated from experimental comparison tables',
 'payload_sha256':payload,
 'files':entries,
 'external_timestamp_required':True,
 'rule':'Any change to a locked file, tolerance, target list, regulator prescription, or observable definition requires a new lock before comparison.'
}
OUT.write_text(json.dumps(obj,indent=2)+'\n')
SHA.write_text(payload+'  QICT_PREDICTION_LOCK\n')
print(json.dumps(obj,indent=2))
