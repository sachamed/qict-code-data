#!/usr/bin/env python3
"""Audit the archived C10 geometry against the common physical GNS carrier.

This script reads the authentic frozen-action and L24 data.  Its conclusions
are deliberately about serialized information, never a surrogate charged
transfer or a physical fermion spectrum.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reproducibility/data/common_physical_chain"
ACTION = ROOT / ("reproducibility/data/physical_g2_inputs/"
                 "L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04/"
                 "components/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json")
AUDIT = ROOT / "reproducibility/results/physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json"
OUT = ROOT / "reproducibility/results/common_carrier/COMMON_CARRIER_AUDIT.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze() -> dict:
    action = json.loads(ACTION.read_text())
    audit = json.loads(AUDIT.read_text())
    payload = DATA / "L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz"
    assert sha256(ACTION) == audit["action_sha256"]
    assert sha256(payload) == audit["precontraction"]["payload_sha256"]
    assert action["phi_star"] == "pi/63"
    assert action["factor_order"] == [
        "Gamma(S_z)", "Gamma(S_y)", "Gamma(S_x)", "U_cell",
        "exp(-i phi_* K_E)", "exp(-i phi_* K_B)", "U_int^rf",
    ]

    with np.load(payload, allow_pickle=False) as z:
        n = int(len(z["c10_nodes"]))
        source = np.asarray(z["c10_edge_src"], dtype=np.int64)
        destination = np.asarray(z["c10_edge_dst"], dtype=np.int64)
        probability = np.asarray(z["c10_edge_probability"], dtype=float)
        boundary = np.asarray(z["c10_edge_is_boundary"], dtype=bool)
        assert np.array_equal(boundary, destination == -1)
        assert np.all((source >= 0) & (source < n))
        assert np.all((destination >= -1) & (destination < n))
        assert np.all(probability >= 0)
        degree = np.bincount(source, minlength=n)
        row_sum = np.bincount(source, weights=probability, minlength=n)
        boundary_sum = np.bincount(source[boundary],
                                   weights=probability[boundary], minlength=n)
        escaped_rows = boundary_sum > 0
        interior = ~boundary
        # Aggregate duplicate graph destinations, retaining the -1 exterior
        # separately.  This is a population kernel, not a quantum transfer.
        kernel = coo_matrix((probability[interior],
                             (source[interior], destination[interior])),
                            shape=(n, n)).tocsr()
        kernel.sum_duplicates()
        interior_mass = np.asarray(kernel.sum(axis=1)).ravel()
        row_norm_squared = np.asarray(kernel.multiply(kernel).sum(axis=1)).ravel()
        branches = np.asarray(z["su2_connected_branch_amplitudes"])
        branch_count = np.asarray(z["su2_connected_branch_count"])
        mask_two = branch_count == 2
        interference = branches[mask_two, 0] * np.conj(branches[mask_two, 1])
        branch_norms = np.sum(abs(branches) ** 2, axis=1)
        assert n == 3925 and len(source) == 17576
        assert int(boundary.sum()) == 3752 and int(escaped_rows.sum()) == 866
        assert len(branches) == 261346 and np.max(abs(branch_norms-1)) < 3e-15
        assert np.max(abs(row_sum - 1)) < 3e-15
        assert np.max(row_norm_squared) < 1
        geometry = {
            "vertices": n,
            "directed_edges": int(len(source)),
            "boundary_edges_with_destination_minus_one": int(boundary.sum()),
            "sources_touching_open_boundary": int(escaped_rows.sum()),
            "sources_with_all_probability_to_open_boundary":
                int(np.count_nonzero(np.isclose(boundary_sum, 1))),
            "exterior_probability_total_across_sources": float(boundary_sum.sum()),
            "maximum_full_row_normalization_residual":
                float(np.max(abs(row_sum-1))),
            "aggregated_interior_node_pairs": int(kernel.nnz),
            "interior_row_mass_min": float(interior_mass.min()),
            "interior_row_mass_max": float(interior_mass.max()),
            "maximum_interior_population_row_squared_norm":
                float(row_norm_squared.max()),
            "outdegree_histogram": {str(k): int(v) for k, v in
                                    zip(*np.unique(degree, return_counts=True))},
            "nonzero_row_count": int(np.count_nonzero(degree)),
            "unitary_on_node_basis_from_population_kernel": False,
            "reason": "An open boundary has destination -1; the 3925 by 3925 "
                      "interior population matrix has all row squared norms below one. "
                      "Coherent boundary legs and interference are required.",
        }
        recoupling = {
            "connected_pairs": int(len(branches)),
            "one_branch_pairs": int(np.count_nonzero(branch_count == 1)),
            "two_branch_pairs": int(mask_two.sum()),
            "positive_real_interference_pairs": int(np.count_nonzero(interference.real > 0)),
            "negative_real_interference_pairs": int(np.count_nonzero(interference.real < 0)),
            "maximum_imaginary_interference": float(np.max(abs(interference.imag))),
            "maximum_branch_norm_residual": float(np.max(abs(branch_norms-1))),
            "meaning": "Signed amplitudes for local SU(2) pairs, without their "
                       "volume-matched contraction into the common boundary density.",
        }

    checkpoints = {}
    for name in ["KATO_STRICT_K0_K12_48_CP.npz",
                 "SU2_B0_ISOLATED_RIESZ_KATO.npz",
                 "U1_B0_RIESZ_KATO_COMPONENTS.npz",
                 "BOUNDARY_CONDITIONED_UCONE_KERNEL_L24.npz"]:
        path = DATA / name
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.files)
            item = {"sha256": sha256(path), "keys": keys}
            if name.startswith("KATO_STRICT"):
                item["state_vector_dimension"] = int(len(z["v"]))
                item["state_norm_error"] = float(abs(np.linalg.norm(z["v"])-1))
                item["complete_charged_U_or_six_sources_in_same_file"] = False
            elif name.startswith("SU2_B0"):
                u = z["U"]
                item["sector_dimension"] = int(u.shape[0])
                item["U_dagger_U_minus_I_Frobenius"] = float(
                    np.linalg.norm(u.conj().T @ u - np.eye(u.shape[0])))
                item["U_is_certified_full_unitary"] = False
            elif name.startswith("U1_B0"):
                item["sector_dimensions"] = [int(z[k].shape[0]) for k in keys
                                             if k.startswith("rho_")]
            else:
                item["diamond_plaquettes"] = int(len(z["diamond_pids"]))
                item["contains_complete_charged_matvec"] = False
        checkpoints[name] = item

    flags = audit["physical_contract_checks"]
    assert len(flags) == 9 and not any(flags.values())
    result = {
        "scope": "Authentic finite C10 geometry and source-carrier type audit; no "
                 "physical charged moment has been computed.",
        "action_sha256": sha256(ACTION), "payload_sha256": sha256(payload),
        "action_group_metadata": action["gauge_group"],
        "group_domain_decision": "Factorwise fundamental magnetic gate does not "
                                 "descend to the claimed diagonal Z6 quotient; "
                                 "use product-group cover as in main m004.",
        "geometry": geometry, "local_su2_recoupling": recoupling,
        "separate_checkpoint_carriers": checkpoints,
        "physical_common_carrier_requirements": {
            "neutral": "Same-regulator Gauss-constrained H_0, complete U_0, "
                       "selected endpoint density rho_0 and certified Riesz gap.",
            "charged": "Same-regulator Gauss-constrained H_f, complete U_f "
                       "including open-boundary legs and all frozen layers.",
            "sources": "Six gauge-covariant typed maps O_a:H_0->H_f; proof that "
                       "G_ab=Tr(rho_0 O_a^dagger O_b) has rank six.",
            "relative_unitary": "W=conjugate(U_0) tensor U_f on "
                                "HS(H_0,H_f), with vec(U_f X U_0^dagger) convention.",
            "boundary": "Volume-matched quantum shell channels, Kato density, "
                        "boundary-to-path insertion and core recoupling.",
            "continuum": "Nested certified Kato buffers and common-regulator "
                         "error bounds for charged spectral moments.",
        },
        "physical_contract_checks": flags,
        "completed_contract_count": int(sum(bool(v) for v in flags.values())),
        "physical_six_by_six_moments_constructible_from_this_archive": False,
        "physical_masses_mixing_or_graviton_constructed": False,
    }
    return result


def main() -> None:
    result = analyze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)),
                      "boundary_edges": result["geometry"]["boundary_edges_with_destination_minus_one"],
                      "boundary_rows": result["geometry"]["sources_touching_open_boundary"],
                      "completed_physical_contracts": result["completed_contract_count"]}, indent=2))


if __name__ == "__main__":
    main()
