#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    r=Path(args.results)
    sec=json.loads((r/'SECTOR_PREDICTIONS.json').read_text())
    lep=json.loads((r/'REDUCED_LEPTON_INVARIANTS.json').read_text())
    su2=json.loads((r/'SU2_TENSOR_AUDIT.json').read_text())
    su3=json.loads((r/'SU3_VALIDATION.json').read_text())
    l24=json.loads((r/'L24_PRECONTRACTION_VALIDATION.json').read_text())
    prim=json.loads((r/'PRIMITIVE_DERIVATION_VALIDATION.json').read_text())
    dec=json.loads((r/'FLAVOR_COORDINATE_COMPARISON.json').read_text())
    pho=json.loads((r/'PHOTON_OBSERVATIONAL_BOUNDS.json').read_text())
    common=json.loads((r/'COMMON_CONTRACTION_CHARACTERIZATION.json').read_text())
    u1=json.loads((r/'U1_SECTOR_VALIDATION.json').read_text())
    masses=json.loads((r/'COMPLETE_MASS_COORDINATES.json').read_text())
    c11=json.loads((r/'C11_NUMERICAL_VALIDATION.json').read_text())
    pg2_path=r/'physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json'
    pg2=json.loads(pg2_path.read_text()) if pg2_path.exists() else None
    rows=[]
    def add(sector,observable,result,classification,provenance):
        rows.append(dict(sector=sector,observable=observable,result=result,classification=classification,provenance=provenance))

    fr=sec['family_receiver']; gc=sec['gauge_charges']
    add('primitive dynamics','oriented electric polynomial',prim['symbolic_oriented_polynomial'],'exact symbolic result','PRIMITIVE_DERIVATION_VALIDATION.json')
    add('primitive dynamics','route-cost identity','P_+(Delta)=T_{2Delta}','exact symbolic result under the declared primitive bridge','PRIMITIVE_DERIVATION_VALIDATION.json')
    add('primitive dynamics','SU(2) tangent rotor','H_rot=K_E+K_B; K_E=n(n+2); K_B=32I-16M_chi1/2','representation-theoretic result','PRIMITIVE_DERIVATION_VALIDATION.json')
    add('primitive dynamics','one-cell Schur coefficient alpha_fac',str(prim['alpha_fac']),'derived reduced observable','PRIMITIVE_DERIVATION_VALIDATION.json')
    add('receiver','primitive family multiplicity','3','exact structural result','SECTOR_PREDICTIONS.json')
    add('receiver','neutral Majorana completion rank',str(fr['neutral_majorana_completion_rank_q_nu']),'exact structural result','SECTOR_PREDICTIONS.json')
    add('receiver','route timing ratio',fr['timing_ratio'],'exact structural result','SECTOR_PREDICTIONS.json')
    add('receiver','integer timing structure',','.join(map(str,fr['integer_timing_structure']))+' times tau_*','exact structural result','SECTOR_PREDICTIONS.json')
    add('gauge','integer hypercharge ray',json.dumps(gc['integer_hypercharges_qY'],sort_keys=True),'exact structural result','SECTOR_PREDICTIONS.json')
    add('gauge','electric charges',json.dumps(gc['electric_charges_Q'],sort_keys=True),'exact structural result','SECTOR_PREDICTIONS.json')
    add('gauge','anomaly residuals',json.dumps(gc['anomaly_residuals'],sort_keys=True),'exact algebraic cancellation','SECTOR_PREDICTIONS.json')

    add('photon','quadratic high-symmetry pattern','9:21:25','exact tangent signature','PHOTON_SIGNATURE_MOMENTS.json')
    add('photon','quadratic LIV matching',pho['exact_LIV_matching']['matching'],'exact convention matching','PHOTON_OBSERVATIONAL_BOUNDS.json')
    s=pho['strongest_cited_time_of_flight_row']
    add('photon','E_tau observational lower interval',f"{s['weakest_orientation_E_tau_lower_GeV']:.12g}--{s['uniform_all_directions_E_tau_lower_GeV']:.12g} GeV",'observational translation','PHOTON_OBSERVATIONAL_BOUNDS.json')
    add('photon','tau_* observational upper interval',f"{s['uniform_all_directions_tau_star_upper_s']:.12g}--{s['weakest_orientation_tau_star_upper_s']:.12g} s",'observational translation','PHOTON_OBSERVATIONAL_BOUNDS.json')

    cm=masses['charged_lepton_reduced_coordinates']
    add('charged leptons','reduced mass coordinates',f"e=1; mu={cm['muon']:.15g}; tau={cm['tau']:.15g}",'reduced scale-free calculation','COMPLETE_MASS_COORDINATES.json')
    add('charged leptons','reduced Koide quotient',f"{cm['koide_Q']:.15g}",'reduced scale-free calculation','COMPLETE_MASS_COORDINATES.json')
    add('charged leptons','dimensional mass map','m_f=(hbar/(c^2 tau_*))*epsilon_f','common-regulator spectral definition','COMPLETE_MASS_COORDINATES.json')

    q=masses['quark_finite_receiver_coordinates']
    add('quarks','up-sector finite coordinates',json.dumps(q['up_normalized_to_heaviest']),'finite source-space calculation','COMPLETE_MASS_COORDINATES.json')
    add('quarks','down-sector finite coordinates',json.dumps(q['down_normalized_to_heaviest']),'finite source-space calculation','COMPLETE_MASS_COORDINATES.json')
    add('quarks','same-scale d/s coordinate ratio',str(q['same_scale_down_ratio_d_over_s']),'finite source-space comparison coordinate','COMPLETE_MASS_COORDINATES.json')
    add('quarks','short-distance mass definition','mbar_f(mu) from the scalar chirality-changing part of the common color-resolved renormalized inverse propagator','common-regulator matched observable','COMPLETE_MASS_COORDINATES.json')
    c=dec['ckm_finite_receiver_test']['finite']
    add('quarks','finite CKM coordinates',f"theta=({c['theta12_deg']:.9g},{c['theta13_deg']:.9g},{c['theta23_deg']:.9g}) deg; delta={c['delta_principal_deg']:.9g} deg",'finite source-space calculation','FLAVOR_COORDINATE_COMPARISON.json')
    add('quarks','CKM physical definition','residue-normalized standard-current G3 coefficient = left-mass-frame G2 mismatch in the common regulator limit','common-regulator matched observable','SECTOR_PREDICTIONS.json')

    n=masses['neutrino_finite_takagi_coordinates']
    add('neutrinos','finite normalized Takagi coordinates',json.dumps(n['normalized_to_heaviest']),'finite source-space calculation','COMPLETE_MASS_COORDINATES.json')
    add('neutrinos','simultaneous two-splitting projected masses',json.dumps(n['weighted_projected_masses_eV'])+' eV','finite-shape projection','COMPLETE_MASS_COORDINATES.json')
    add('neutrinos','simultaneous projected splittings',json.dumps(n['weighted_projected_splittings_eV2'])+' eV^2','finite-shape projection','COMPLETE_MASS_COORDINATES.json')
    add('neutrinos','simultaneous projected sum masses',str(n['weighted_projected_sum_masses_eV'])+' eV','finite-shape projection','COMPLETE_MASS_COORDINATES.json')
    add('neutrinos','simultaneous projected m_beta',str(n['weighted_projected_m_beta_eV'])+' eV','finite-shape projection','COMPLETE_MASS_COORDINATES.json')
    add('neutrinos','simultaneous projected m_beta_beta interval',json.dumps(n['weighted_projected_m_bb_phase_interval_eV'])+' eV','finite-shape projection','COMPLETE_MASS_COORDINATES.json')
    add('neutrinos','physical mass definition','Takagi singular values of the interacting 6x6 neutral Nambu kernel with the receiver-clock conversion','common-regulator spectral observable','COMPLETE_MASS_COORDINATES.json')
    p=dec['pmns_finite_receiver_test']['finite']
    add('neutrinos','finite PMNS coordinates',f"theta=({p['theta12_deg']:.9g},{p['theta13_deg']:.9g},{p['theta23_deg']:.9g}) deg; delta={p['delta_principal_deg']:.9g} deg",'finite source-space calculation','FLAVOR_COORDINATE_COMPARISON.json')
    add('neutrinos','PMNS physical definition','residue-normalized standard-current G3 coefficient = left-mass-frame G2 mismatch in the common regulator limit','common-regulator matched observable','SECTOR_PREDICTIONS.json')

    add('L24 weak tensors','sparse MPS nonzero coefficients',str(su2['sequential_SU2']['total_core_nonzero_coefficients']),'direct primary-tensor audit','SU2_TENSOR_AUDIT.json')
    add('L24 weak tensors','four-plaquette support',str(su2['four_plaquette_SU2']['support_direct']),'direct primary-tensor audit','SU2_TENSOR_AUDIT.json')
    st=l24['statistics']
    add('L24 rank-six precontraction','covering-lattice size',f"L={st['L']}, n={st['n']}; {st['links']} links; {st['plaquettes']} plaquettes",'exact primary-data validation','L24_PRECONTRACTION_VALIDATION.json')
    add('L24 rank-six precontraction','connected SU2 pair recouplings',f"{st['su2_connected_pairs']} pairs in {st['su2_translation_orientation_classes']} classes",'exact primary-data validation','L24_PRECONTRACTION_VALIDATION.json')
    add('L24 rank-six precontraction','C10 geometric graph',f"{st['c10_nodes']} nodes; {st['c10_edges']} directed edges",'exact primary-data validation','L24_PRECONTRACTION_VALIDATION.json')
    add('L24 rank-six precontraction','six-channel source order',','.join(st['source_order']),'exact primary-data validation','L24_PRECONTRACTION_VALIDATION.json')

    cc=common['coherence_resolved_c10_witness']
    add('common contraction','C10 row-stochastic residual',str(common['normalization_residuals']['c10_row_stochasticity_max_residual']),'direct contraction check','COMMON_CONTRACTION_CHARACTERIZATION.json')
    add('common contraction','SU2 branch-normalization residual',str(common['normalization_residuals']['su2_branch_probability_max_residual']),'direct contraction check','COMMON_CONTRACTION_CHARACTERIZATION.json')
    add('common contraction','C10 coherence witness trace distance',str(cc['coherent_state_output_trace_distance']),'quantum shell-map identifiability result','COMMON_CONTRACTION_CHARACTERIZATION.json')
    add('common contraction','strict Kato dimension and gap',f"dim={common['strict_kato_checkpoint']['dimension']}; gap={common['strict_kato_checkpoint']['gap']}",'finite sparse-state spectral calculation','COMMON_CONTRACTION_CHARACTERIZATION.json')
    add('common contraction','matrix-free Riesz numerical precision',json.dumps(common['matrix_free_riesz_validation'],sort_keys=True),'deterministic numerical validation','COMMON_CONTRACTION_CHARACTERIZATION.json')

    u1a=u1['components']['singleton_boundary_component']; u1b=u1['components']['four_plaquette_boundary_component']
    add('abelian gauge sector','U(1) singleton Riesz density',f"dim={u1a['dimension']}; trace={u1a['trace']:.16g}; contour gap={u1a['uniform_contour_lower_bound_min']:.12g}",'finite L24 boundary-sector validation','U1_SECTOR_VALIDATION.json')
    add('abelian gauge sector','U(1) four-plaquette Riesz density',f"dim={u1b['dimension']}; trace={u1b['trace']:.16g}; contour gap={u1b['uniform_contour_lower_bound_min']:.12g}",'finite L24 boundary-sector validation','U1_SECTOR_VALIDATION.json')

    add('color','local SU3 algebra validation','max residual '+str(su3['summary']['tensor_identity_max_error']),'finite-matrix validation','SU3_VALIDATION.json')
    add('spectral methods','Riesz/Toeplitz/reconstruction','deterministic validation suite','deterministic numerical validation','RIESZ_NUMERICAL_VALIDATION.json; SPECTRAL_METHOD_VALIDATION.json')
    add('spectral methods','C11 block-Toeplitz margin',str(c11['toeplitz_lambda_min_numeric']),'deterministic positive-matrix validation','C11_NUMERICAL_VALIDATION.json')
    add('spectral methods','C11 depth-18 block pencil',f"rank={c11['block_pencil']['spectral_numerical_rank']}; radial deviation={c11['block_pencil']['spectral_unit_circle_max_deviation']:.3e}",'deterministic matrix-pencil validation','C11_NUMERICAL_VALIDATION.json')
    add('spectral methods','exact matrix flat extension',c11['exact_flat_extension']['flat_extension_relation'],'exact symbolic result','C11_NUMERICAL_VALIDATION.json')
    if pg2 is not None:
        pc=pg2['precontraction']['statistics']
        add('physical G2 execution','L24 exact precontraction',f"L={pc['L']}, n={pc['n']}; {pc['links']} links; {pc['plaquettes']} plaquettes; {pc['su2_connected_pairs']} connected SU2 pairs",'exact physical-input certificate','physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json')
        add('physical G2 execution','physical input-chain status',pg2['status'],'execution-contract classification','physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json')
        add('physical G2 execution','continuum-sequence tensor inputs','; '.join(pg2.get('continuum_sequence_inputs',pg2['missing_physical_objects'])),'explicit continuum input set','physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json')
        add('physical G2 execution','continuum G2 stage','continuum value certified' if pg2['physical_G2_promoted'] else 'definition and L24 precontraction certified','physical classification','physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json')

    with open(args.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sector','observable','result','classification','provenance']); w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} result registry rows')

if __name__=='__main__': main()
