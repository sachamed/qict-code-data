from __future__ import annotations
import json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SB = ROOT / "stage_b"
R = SB / "results"
OUT = R / "FINAL_PHYSICAL_CLOSURE_AUDIT.json"

def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None

expected = {
    # Infrared/source-sector reduction.
    "C10_KATO_GAUSS_SECTOR_REDUCTION.json": "PASS_C10_KATO_GAUSS_SECTOR_REDUCTION",
    "C10_SPLITSTEP_CORE_SECTOR_MAP.json": "PASS_C10_SPLITSTEP_CORE_AFFINE_SECTOR_MAP",
    "U1_GAUSS_REFERENCE_COORDINATE.json": "PASS_U1_GAUSS_REFERENCE_PATH_IS_COORDINATE_CHOICE",
    "SU2_REFERENCE_PATH_LOOP_RELATION.json": "PASS_SU2_REFERENCE_PATHS_DIFFER_BY_CLOSED_LOOP_COORDINATE",
    # Exact reductions of spectator/global charged structure.
    "SU3_RELATIVE_CANCELLATION.json": "PASS_EXACT_SU3_CANCELLATION_IN_RELATIVE_LEPTON_CORRELATOR",
    "RELATIVE_ELECTRIC_DEFECT_REDUCTION.json": "PASS_RELATIVE_ELECTRIC_DEFECT_LOCALIZATION",
    "LOCAL_DEFECT_FRONTIER_WIDTH.json": "PASS_LOCAL_DEFECT_FRONTIER_WIDTH_14",
    # Common-history matter/gauge structure.
    "CONDITIONAL_PRODUCT_GROUP_INFLUENCE.json": "PASS_EXACT_CONDITIONAL_PRODUCT_GROUP_INFLUENCE_DECOMPOSITION",
    "C8_L2_INTERNAL_REDUCTION.json": "PASS_EXACT_L2_C8_POSITION_ELIMINATION",
    "PATHWIDTH_EXACT_CERTIFICATE.json": "PASS_EXACT_LINK_SWEEP_PATHWIDTH_14",
    # Neutral environment engines/regressions.
    "SU2_K0_ORBIT_KRYLOV_D4_PARALLEL.json": "PASS_K0_HB_EXACT_D4",
    "U1_L2_KATO_S24_ORBIT_REPRODUCED.json": "PASS_S24_ORBIT_KATO_REFINEMENT",
    "U1_TRANSLATION_ONLY_NEUTRAL_OPERATOR.json": "PASS_U1_TRANSLATION_ONLY_NEUTRAL_OPERATOR",
    "PHYSICAL_ERROR_BUDGET.json": "PASS_FROZEN_PHYSICAL_ERROR_BUDGET",
    "FRONTIER_ENGINE_ARCHITECTURE_SELECTION.json": "PASS_FRONTIER_ENGINE_ARCHITECTURE_SELECTION",
}
checks = {}
for fn, status in expected.items():
    x = load(fn)
    checks[fn] = bool(x and x.get("status") == status)

# The old requirement for a standalone C10/Kato boundary-to-core charged state is
# superseded only if all three exact structural replacement certificates pass:
# (i) C10 -> unique affine L=2 Gauss sector with trivial torus homology,
# (ii) charged/neutral difference localizes to electric defects in the relative correlator,
# (iii) those unary defects do not enlarge the exact width-14 spatial frontier.
structural_source_reduction_pass = all(checks.get(k, False) for k in (
    "C10_SPLITSTEP_CORE_SECTOR_MAP.json",
    "RELATIVE_ELECTRIC_DEFECT_REDUCTION.json",
    "LOCAL_DEFECT_FRONTIER_WIDTH.json",
))

hfile = R / "PEPO3D_H_ROUTE_SCALARS.json"
h = json.loads(hfile.read_text()) if hfile.exists() else None
required_h = ["h_8", "h_9", "h_10"]
coarse_fine_present = False
promotion_inequality_pass = False
frontier_convergence_pass = False
if h:
    if all(k in h for k in ("coarse", "fine")):
        coarse_fine_present = all(k in h["coarse"] and k in h["fine"] for k in required_h)
    elif all(k in h for k in required_h):
        coarse_fine_present = all(isinstance(h[k], dict) and "coarse" in h[k] and "fine" in h[k] for k in required_h)
    promotion_inequality_pass = bool(h.get("promotion_pass", False))
    frontier_convergence_pass = bool(h.get("frontier_convergence_pass", False))

all_exact_reductions_pass = all(checks.values()) and structural_source_reduction_pass
all_pass = all_exact_reductions_pass and coarse_fine_present and frontier_convergence_pass and promotion_inequality_pass

remaining=[]
if not all_pass:
    if not all_exact_reductions_pass:
        remaining.append("Repair one or more exact structural-reduction certificates listed in checks.")
    if not coarse_fine_present:
        remaining.append("Execute the neutral-Kato width-14 U(1)/SU(2) influence frontiers with the common dimension-4 C8 auxiliary and branch-conditioned local electric defects at both frozen cutoffs for d=8,9,10.")
        remaining.append("Write the six actually computed coarse/fine amplitudes to stage_b/results/PEPO3D_H_ROUTE_SCALARS.json; no surrogate observable is accepted.")
    if coarse_fine_present and not frontier_convergence_pass:
        remaining.append("Pass the block-SVD/frontier convergence criterion and record its accumulated discarded-norm certificates.")
    if coarse_fine_present and not promotion_inequality_pass:
        remaining.append("Pass the frozen two-cutoff promotion inequality including analytic Peter-Weyl tails and numerical frontier errors.")

obj = {
    "status": "PASS_TASK3_PHYSICAL_ALL_PASS" if all_pass else "BLOCKED_TASK3_PHYSICAL_PROMOTION",
    "all_pass": all_pass,
    "exact_structural_reductions_all_pass": all_exact_reductions_pass,
    "checks": checks,
    "standalone_boundary_to_core_charged_state_required": False,
    "standalone_boundary_to_core_replacement": {
        "status": "PASS" if structural_source_reduction_pass else "FAIL",
        "ingredients": [
            "C10 split-step pushforward selects one affine L=2 source-sink Gauss sector with trivial torus homology",
            "relative-correlator transform moves charged/non-neutral gauge dependence to local electric defects while magnetic multiplication remains in the neutral environment",
            "the resulting unary defects leave the exact spatial pathwidth equal to 14"
        ],
        "scope": "This is a structural reduction, not a numerical evaluation of h_d."
    },
    "physical_h_coarse_fine_present": coarse_fine_present,
    "frontier_convergence_pass": frontier_convergence_pass,
    "two_cutoff_promotion_inequality_pass": promotion_inequality_pass,
    "required_physical_outputs": required_h,
    "remaining_if_blocked": remaining,
    "anti_false_promotion": "No static, one-link, neutral-vacuum return factor, Wilson-string-probability state, finite-K auxiliary model, S24-only diagnostic, or post-summed product-group amplitude is accepted as h_d. The neutral Kato state is an environment; the charged observable still contains the common C8 history and its local electric defects."
}
text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
OUT.write_text(text)
print(json.dumps({
    "status": obj["status"], "all_pass": all_pass,
    "structural_source_reduction_pass": structural_source_reduction_pass,
    "sha256": hashlib.sha256(text.encode()).hexdigest(), "checks": checks
}, indent=2))
