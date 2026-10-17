#!/usr/bin/env python3
"""Exact flux-period and finite-rank Kato obstruction for the frozen Haar carrier.

The full operator conclusion is analytic and conditional on the factor types
specified in main m004 and supplement s23/s44: electric spectral exponential,
magnetic multiplication, and gauge-configuration-decomposable matter/route
gates. The numerical checks below certify the genuine L24 cellular cycles,
integer phase identities, and the exact electric seed's cutoff behaviour.
No complete charged C10 matvec or physical six-source state is synthesized.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import jv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reproducibility/data/common_physical_chain"
OUT = ROOT / "reproducibility/results/flux_kato"
PHI_DENOMINATOR = 63
ACTION = ROOT / "reproducibility/data/physical_g2_inputs/L24_RANK6_STATE_CONTRACTION_PAYLOAD_2026-10-04/components/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def electric_cost(n: np.ndarray) -> np.ndarray:
    return 2*n*n+n


def verify_integer_periods() -> dict:
    # E=N/c allows both integer-normalized and physical hypercharge conventions.
    # Q=126*c^2 gives (h((N+Q eta)/c)-h(N/c))/126 an integer.
    rows = []
    for c in (1, 6):
        q = 126*c*c
        checks = 0
        for n in range(-500, 501):
            for eta in (-2, -1, 0, 1, 2):
                # Difference times c^2, using only exact Python integers.
                a = 2*(n+q*eta)**2+c*(n+q*eta)
                b = 2*n*n+c*n
                exact_delta_numerator = a-b
                assert exact_delta_numerator % (126*c*c) == 0
                exact_turns = exact_delta_numerator // (126*c*c)
                assert exact_turns == 4*eta*n+252*c*c*eta*eta+c*eta
                checks += 1
        rows.append({"electric_normalization_E_equals_N_over_c": c,
                     "integer_flux_shift_Q": q,
                     "exact_integer_phase_tests": checks,
                     "failures": 0})
    return {"identity": "h((N+126*c^2*eta)/c)-h(N/c) = "
                        "126*(4*eta*N+252*c^2*eta^2+c*eta)",
            "phase_per_update": "pi/63", "conventions_checked": rows}


def verify_actual_cycles() -> dict:
    path = DATA / "L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz"
    with np.load(path, allow_pickle=False) as z:
        links = np.asarray(z["links"], int)
        faces = np.asarray(z["plaquette_edge_indices"], int)
        signs = np.asarray(z["plaquette_edge_sign"], int)
        starts = links[:, 1:]
        ends = starts.copy()
        ends[np.arange(len(links)), links[:, 0]] += 1
        max_divergence = 0
        for edges, coeff in zip(faces, signs):
            charges = {}
            for edge, value in zip(edges, coeff):
                start, end = tuple(starts[edge]), tuple(ends[edge])
                charges[start] = charges.get(start, 0)-int(value)
                charges[end] = charges.get(end, 0)+int(value)
            max_divergence = max(max_divergence, max(map(abs, charges.values())))
        assert max_divergence == 0 and len(faces) == 18252
        assert all(len(set(edges)) == 4 for edges in faces)
        # Actual coherent Wilson-loop multipliers T_eta = exp(i*126*eta.theta).
        np.savez_compressed(OUT / "L24_GAUSS_COMPATIBLE_FLUX_TRANSLATIONS.npz",
                            cycle_edge_indices=faces.astype(np.int32),
                            cycle_orientations=signs.astype(np.int8),
                            integer_flux_translation_coefficients=(126*signs).astype(np.int16),
                            first_cycle_edge_coordinates=links[faces[0]],
                            period=np.asarray(126, dtype=np.int64))
    return {"payload_sha256": sha(path), "gauge_links": int(len(links)),
            "gauss_compatible_closed_cycles_checked": int(len(faces)),
            "max_exact_boundary1_boundary2_coefficient": int(max_divergence),
            "first_cycle_edges": faces[0].tolist(),
            "first_cycle_orientations": signs[0].tolist(),
            "meaning": "Exact gauge-invariant high-flux Wilson translations "
                       "on actual archived plaquette cycles; no new coupling."}


def verify_loop_kernel() -> dict:
    # Independent single-plaquette calculation with the same electric and
    # magnetic couplings. Kernel periodicity is a mathematical control only.
    flux = np.arange(-85, 86, dtype=np.int64)
    m, n = flux[:, None], flux[None, :]
    alpha = 4*np.pi/63
    b = np.exp(-1j*alpha)*(1j**(m-n))*jv(m-n, alpha)
    # Actual L24 plaquette orientations (+,+,-,-) cancel the linear electric term.
    k = 8*n*n
    kp = 8*(n+63)**2
    assert np.all((kp-k) % 126 == 0)
    u = b*np.exp(-1j*np.pi*((k % 126)/63))
    translated = b*np.exp(-1j*np.pi*((kp % 126)/63))
    err = float(np.max(abs(u-translated)))
    assert err == 0
    roots = np.arange(63)[(8*np.arange(63)**2) % 126 == 0]
    assert roots.tolist() == [0, 21, 42]
    return {"kernel": "U_mn=exp(-4pi*i/63)*i^(m-n)*J_(m-n)(4pi/63)"
                       "*exp(-pi*i*8n^2/63)",
            "period_on_circulating_flux": 63,
            "sample_kernel_dimension": list(u.shape),
            "simultaneous_m_n_shift_63_max_error": err,
            "electric_eigenvalue_one_residues_modulo_63": roots.tolist(),
            "physical_scope": "Exact gauge-only loop operator; not full C10."}


def verify_seed_cutoff_states() -> dict:
    rows = []
    radii = [63, 126, 252, 504, 1008]
    counts = []
    for radius in radii:
        n = np.arange(-radius, radius+1, dtype=np.int64)
        mask = (8*n*n) % 126 == 0
        rank = int(mask.sum())
        counts.append(rank)
        rows.append({"flux_cutoff_radius": radius, "dimension": len(n),
                     "eigenvalue_one_riesz_rank": rank,
                     "normalized_cluster_purity": 1/rank,
                     "weight_on_fixed_flux_zero": 1/rank})
    distances = []
    for a, b in zip(rows[:-1], rows[1:]):
        # Nested diagonal projectors: D(Pa/ra,Pb/rb)=1-ra/rb.
        ra, rb = a["eigenvalue_one_riesz_rank"], b["eigenvalue_one_riesz_rank"]
        td = 1-ra/rb
        radius = b["flux_cutoff_radius"]
        n = np.arange(-radius, radius+1, dtype=np.int64)
        zero = (8*n*n) % 126 == 0
        pa = ((abs(n) <= a["flux_cutoff_radius"]) & zero).astype(float)/ra
        pb = zero.astype(float)/rb
        direct = .5*float(np.sum(abs(pa-pb)))
        assert abs(td-direct) < 3e-16
        distances.append({"radii": [a["flux_cutoff_radius"], radius],
                          "trace_distance": direct,
                          "exact_rational": f"{rb-ra}/{rb}"})
    assert counts == [7, 13, 25, 49, 97]
    np.savez_compressed(OUT / "ELECTRIC_SEED_NESTED_RIESZ_DATA.npz",
                        cutoff_radii=np.asarray(radii), ranks=np.asarray(counts),
                        trace_distances=np.asarray([x["trace_distance"] for x in distances]))
    return {"seed": "Gauss-invariant four-link U1 electric Floquet gate",
            "selection_rule": "normalized eigenvalue-one Riesz cluster P_R/r_R",
            "cutoffs": rows, "successive_trace_distances": distances,
            "rank_formula_at_R_63_m": "rank(P_R)=6*m+1",
            "doubling_trace_distance_limit": .5,
            "trace_norm_cauchy": False,
            "conclusion": "The source's normalized-cluster rule has no "
                          "trace-class regulator limit in this exact electric seed."}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    action = json.loads(ACTION.read_text())
    assert action['phi_star'] == 'pi/63'
    assert action['electric_generator']['U1'] == '2 E_Y^2 + E_Y'
    result = {
        "status": "EXACT_FLUX_SYMMETRY__FINITE_RANK_HAAR_RIESZ_SELECTION_OBSTRUCTED",
        "frozen_action_sha256": sha(ACTION),
        "integer_phase_identity": verify_integer_periods(),
        "actual_L24_cycles": verify_actual_cycles(),
        "independent_coherent_loop_kernel": verify_loop_kernel(),
        "nested_seed_riesz_states": verify_seed_cutoff_states(),
        "analytic_full_operator_result": {
            "carrier": "Gauss-constrained exact compact-link Haar L2 carrier "
                       "with a surviving nonzero closed Abelian cycle",
            "translation": "T_eta multiplication by exp(i*126*sum_e eta_e*theta_e)",
            "hypotheses": ["integer-normalized U1 generator E=-i d/dtheta",
                           "frozen electric exponential exp[-i*pi*(2E^2+E)/63]",
                           "magnetic multiplier on compact configurations",
                           "matter streams, cell and route gates decomposable in compact configurations",
                           "physical carrier is invariant under the gauge-invariant closed-loop multiplier"],
            "source_files": ["paper/sections/main/m004.tex",
                             "paper/sections/main/m098.tex",
                             "paper/sections/supplement/s23.tex",
                             "paper/sections/supplement/s44.tex"],
            "commutation": "[U_micro,T_eta]=0",
            "T_eta_point_spectrum": "empty on Haar L2",
            "nonzero_finite_rank_Riesz_projector_of_full_U_exists": False,
            "proof": "Every Riesz projector commutes with T_eta. A nonzero finite "
                     "dimensional invariant space of a unitary T_eta contains an "
                     "eigenvector, contradicting its atomless multiplication spectrum.",
            "scope": "This theorem applies to the exact Haar representation and "
                     "the factor types listed above. It does not forbid algebraic "
                     "states or state selection in a separately specified representation.",
        },
        "bounded_log_gibbs": {
            "principal_phase_generator_bound": "norm(K)<=pi",
            "partition_function_on_infinite_Haar_carrier": "infinite for every finite beta>0",
            "proof": "exp(-beta*K)>=exp(-beta*pi)*I, and Tr(I)=infinite",
            "finite_cutoff_gibbs_states_still_defined": True,
        },
        "requested_physical_chain": {
            "complete_charged_boundary_contraction_certified": False,
            "common_finite_rank_Kato_density_certified": False,
            "six_physical_GNS_sources_on_full_C10_certified": False,
            "failure_reason": "The finite-rank Floquet Riesz selector is incompatible "
                              "with the exact compact Haar endpoint under the stated factor hypotheses.",
            "source_authentic_alternative": "Supplement s09 already specifies "
                                            "the unwrapped primitive cost K_prim=K_E+K_B. "
                                            "Its ground state is not the finite-rank "
                                            "Floquet Riesz state obstructed here. "
                                            "Global state convergence and stationary "
                                            "selection remain separate tests.",
        },
    }
    for p in [OUT / "L24_GAUSS_COMPATIBLE_FLUX_TRANSLATIONS.npz",
              OUT / "ELECTRIC_SEED_NESTED_RIESZ_DATA.npz"]:
        result.setdefault("generated_data_sha256", {})[p.name] = sha(p)
    path = OUT / "FLUX_KATO_OBSTRUCTION_CERTIFICATE.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cycles_checked": result["actual_L24_cycles"]["gauss_compatible_closed_cycles_checked"],
                      "zero_phase_seed_ranks": [r["eigenvalue_one_riesz_rank"] for r in result["nested_seed_riesz_states"]["cutoffs"]],
                      "trace_distances": [r["trace_distance"] for r in result["nested_seed_riesz_states"]["successive_trace_distances"]],
                      "finite_rank_exact_Haar_Kato_possible": False}, indent=2))


if __name__ == "__main__":
    main()
