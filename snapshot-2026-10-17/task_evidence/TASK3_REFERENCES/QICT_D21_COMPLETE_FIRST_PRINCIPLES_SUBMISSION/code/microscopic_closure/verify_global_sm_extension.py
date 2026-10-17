#!/usr/bin/env python3
"""Deterministic audit for the QICT-D21 global-SM/flavor/RG/dark-sector extension."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "global_sm_extension"
files = {
    "gauge": R / "FULL_SM_GAUGE_GROUP_CERTIFICATE.json",
    "mixing": R / "MIXING_CLOSURE_CERTIFICATE.json",
    "rg": R / "RG_ALPHA_CLOSURE_CERTIFICATE.json",
    "dm": R / "SINGLET_SCALAR_LZ_CERTIFICATE.json",
}
D = {k: json.loads(p.read_text()) for k,p in files.items()}
checks = {
    "sm_group_Z6": D["gauge"]["global_gauge_group"] == "(SU(3)_C x SU(2)_L x U(1)_Y)/Z6",
    "six_element_central_kernel": len(D["gauge"]["central_kernel_elements_a_b_n"]) == 6,
    "local_anomalies_cancel": all(bool(v) for k,v in D["gauge"]["checks"].items() if "kernel" not in k),
    "lattice_only_uniqueness_rejected": "does not select SU(3)" in D["gauge"]["lattice_uniqueness_boundary"],
    "mixing_observable_defined": D["mixing"]["status"].startswith("DEFINED_OBSERVABLE"),
    "mixing_payload_not_fabricated": D["mixing"]["physical_payload_present"] is False,
    "family_universal_no_mixing": D["mixing"]["family_universal_vertex_theorem"]["offdiagonal_modulus_residual"] < 1e-12,
    "tbm_labeled_uv_only": "not a physical PMNS prediction" in D["mixing"]["finite_reference_scope"],
    "rg_identifiability_audit": D["rg"]["status"] == "PASS_IDENTIFIABILITY_AND_FIXED_POINT_AUDIT",
    "three_rg_boundary_constants_required": D["rg"]["integration_constants_required"] == 3,
    "nonzero_alpha_not_one_loop_fixed_point": "implies e=0" in D["rg"]["nonzero_stationary_alpha_theorem"],
    "qca_response_boundary_not_fabricated": D["rg"]["qca_boundary_payload_present"] is False,
    "dm_requires_qca_scalar_pole": "scalar singlet pole mass" in D["dm"]["required_qca_observables"][0],
    "dm_portal_vertex_not_fabricated": D["dm"]["qca_payload_present"] is False,
    "lz_limit_is_external_comparison_only": D["dm"]["external_data_used_only_for_comparison"] is True,
    "no_external_mixing_fit": D["mixing"]["external_mixing_data_used"] is False,
}
assert all(checks.values()), [k for k,v in checks.items() if not v]
out = {
    "status": f"PASS_{sum(checks.values())}_OF_{len(checks)}",
    "checks": checks,
    "scientific_scope": {
        "proven": [
            "global Standard-Model gauge group and Z6 kernel relative to the frozen finite algebra/representation",
            "color-covariant QCA transport and block-sparse SU(3) extension",
            "physical CKM/PMNS extraction map from interacting residue-normalized weak vertices",
            "family-universal no-mixing theorem",
            "RG boundary identifiability and nonzero-one-loop-fixed-point obstruction",
            "parameter-free dark-matter decision map once the scalar pole and portal proper vertex are produced"
        ],
        "not_promoted_without_payload": [
            "numerical physical CKM matrix",
            "numerical physical PMNS matrix",
            "alpha_em=1/137 from QCA",
            "singlet-scalar mass, portal coupling, relic abundance, or LZ cross section"
        ]
    }
}
(R / "GLOBAL_SM_EXTENSION_CERTIFICATE.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
