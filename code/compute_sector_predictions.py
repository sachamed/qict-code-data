#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from fractions import Fraction
from pathlib import Path


def frac_str(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def bcc_b(nx: Fraction, ny: Fraction, nz: Fraction) -> Fraction:
    s4 = nx**4 + ny**4 + nz**4
    return (Fraction(11) - 8*s4) / 32


def bcc_d_from_invariants(s4: Fraction, s6: Fraction, p: Fraction) -> Fraction:
    return (11520*p - 1088*s4*s4 + 5232*s4 - 1792*s6 - 2505) / 18432


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parameters', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    params = json.loads(Path(args.parameters).read_text())

    # Hypercharge convention q_Y = 6Y, using right-handed fields.
    q = {'Q':1, 'u':4, 'd':-2, 'L':-3, 'e':-6, 'nu':0, 'H':3}
    Y = {k: Fraction(v,6) for k,v in q.items()}
    charges = {
        'u_L': Fraction(1,2) + Y['Q'],
        'd_L': Fraction(-1,2) + Y['Q'],
        'u_R': Y['u'], 'd_R': Y['d'],
        'nu_L': Fraction(1,2) + Y['L'],
        'e_L': Fraction(-1,2) + Y['L'],
        'nu_R': Y['nu'], 'e_R': Y['e'],
        'H_plus': Fraction(1,2) + Y['H'],
        'H_zero': Fraction(-1,2) + Y['H'],
    }
    # Left-handed conjugate anomaly convention.
    qa = {'Q':1, 'uc':-4, 'dc':2, 'L':-3, 'ec':6, 'nuc':0}
    anomaly = {
        'SU3_SU3_U1': 2*qa['Q'] + qa['uc'] + qa['dc'],
        'SU2_SU2_U1': 3*qa['Q'] + qa['L'],
        'grav_grav_U1': 6*qa['Q'] + 3*qa['uc'] + 3*qa['dc'] + 2*qa['L'] + qa['ec'] + qa['nuc'],
        'U1_cubic': 6*qa['Q']**3 + 3*qa['uc']**3 + 3*qa['dc']**3 + 2*qa['L']**3 + qa['ec']**3 + qa['nuc']**3,
    }

    cycle = int(params['reduced_lepton_benchmark']['route_cycle_length'])
    routes = list(map(int, params['reduced_lepton_benchmark']['route_lengths']))
    defects = [cycle - 2*d for d in routes]
    r = 7
    q_charged = 2*r
    q_neutral = 3*r
    recurrence = int(params['receiver_central_order'])

    # BCC signatures.
    b_axis = Fraction(3,32)
    b_face = Fraction(7,32)
    b_body = Fraction(25,96)
    d_axis = bcc_d_from_invariants(Fraction(1), Fraction(1), Fraction(0))
    d_face = bcc_d_from_invariants(Fraction(1,2), Fraction(1,4), Fraction(0))
    d_body = bcc_d_from_invariants(Fraction(1,3), Fraction(1,9), Fraction(1,27))

    # Model-independent observable maps fixed by the QICT sector definitions.
    out = {
        'classification':'SECTOR_PREDICTION_MAP',
        'normalization':'q_Y = 6Y; electric charge Q = T3 + Y',
        'family_receiver':{
            'primitive_family_dimension':3,
            'pointed_receiver_rank_r':r,
            'charged_real_completion_rank_q':q_charged,
            'neutral_majorana_completion_rank_q_nu':q_neutral,
            'route_cycle_length':cycle,
            'route_lengths':routes,
            'oriented_defects':defects,
            'timing_ratio':':'.join(map(str, defects)),
            'integer_timing_structure':[1,3,5,q_charged,q_neutral,recurrence],
            'receiver_collision_phase_over_pi':frac_str(Fraction(1,63)),
            'receiver_recurrence_order':recurrence,
            'primitive_shell_dimensions':[3*m for m in range(1,8)],
        },
        'gauge_charges':{
            'integer_hypercharges_qY':q,
            'hypercharges_Y':{k:frac_str(v) for k,v in Y.items()},
            'electric_charges_Q':{k:frac_str(v) for k,v in charges.items()},
            'anomaly_residuals':anomaly,
            'color_representations':{
                'u_L,d_L,u_R,d_R':'fundamental 3',
                'nu_L,e_L,nu_R,e_R,H':'singlet 1'
            },
        },
        'photon_signatures':{
            'protected_zero':'omega_gamma(0)=0',
            'linear_energy_correction':'exactly absent in the BCC tangent expansion',
            'quadratic_law':'v_gamma/c = 1 - b(n) (E/E_tau)^2 + O((E/E_tau)^4)',
            'b_formula':'(11 - 8*(nx^4+ny^4+nz^4))/32',
            'b_axis':frac_str(b_axis), 'b_face':frac_str(b_face), 'b_body':frac_str(b_body),
            'quadratic_high_symmetry_ratio':[9,21,25],
            'b_min':frac_str(b_axis), 'b_max':frac_str(b_body),
            'quartic_law':'v_gamma/c = 1 - b e^2 + d e^4 + O(e^6)',
            'd_formula':'(11520 P -1088 S4^2 +5232 S4 -1792 S6 -2505)/18432',
            'd_axis':frac_str(d_axis), 'd_face':frac_str(d_face), 'd_body':frac_str(d_body),
            'd_axis_float':float(d_axis),'d_face_float':float(d_face),'d_body_float':float(d_body),
        },
        'charged_fermion_observable':{
            'source_rank':6,
            'source_order':params['source_order'],
            'mass_block':'M_f^ren = Z_L^(1/2) M_f(z_*,0) Z_R^(1/2)',
            'mass_eigenvalues':'singular values of the residue-normalized left-right block',
            'dimensionless_mass_map':'m_f = hbar * epsilon_f / (c^2 tau_*) after common relativistic normalization',
        },
        'quark_sector':{
            'electric_charges':{'up_type':frac_str(Fraction(2,3)),'down_type':frac_str(Fraction(-1,3))},
            'color_carrier_dimension':3,
            'weak_current_color_structure':'Gamma_ud^W = I_3(color) tensor Gamma_ud^W(family) for a color-singlet weak current',
            'short_distance_mass_observable':'scalar chirality-changing part of the renormalized inverse propagator in a declared scheme and scale',
            'mixing_definition':'V_CKM = U_uL^dagger U_dL, jointly matched to the normalized proper three-point standard-current coefficient divided by g_W',
        },
        'neutrino_sector':{
            'right_handed_hypercharge':0,
            'electric_charge':0,
            'neutral_receiver_rank':q_neutral,
            'active_family_rank':3,
            'nambu_dimension':6,
            'nambu_mass_matrix':'M_N(lambda) = [[0, lambda D_nuD],[lambda D_nuD^T, M_R]], with M_R symmetric',
            'mass_definition':'Takagi singular values of the interacting neutral Nambu kernel after active-branch continuation',
            'active_branch_gap':'g(lambda)=s_4(lambda)-s_3(lambda) > 0 fixes the three-dimensional active Takagi branch',
            'mixing_definition':'U_PMNS = U_eL^dagger U_nu, jointly matched to the normalized proper three-point standard-current coefficient divided by g_W',
            'beta_decay_mass':'m_beta^2 = sum_i |U_ei|^2 m_i^2',
            'majorana_0nubb_mass':'m_bb = |sum_i U_ei^2 m_i|',
            'mass_squared_splittings':'Delta m_ij^2 = (hbar/(c^2 tau_*))^2 (epsilon_i^2-epsilon_j^2)',
        },
        'mixing_and_cp':{
            'CKM':'left-handed up/down mass-frame mismatch with independent matched three-point current equality',
            'PMNS':'left-handed charged-lepton/neutral mass-frame mismatch with independent matched three-point current equality',
            'unitarity_target':'V^dagger V = I_3 in the common regulator limit',
            'jarlskog':'J = Im(V_ij V_kl V_il^* V_kj^*) for any standard distinct-index quartet',
        },
        'prediction_classes':{
            'numerically_evaluated_now':['receiver integer structure','hypercharge/electric-charge identities','anomaly cancellation','photon quadratic and quartic high-symmetry signatures'],
            'operator_predictions_requiring_common_interacting_contraction':['physical charged-fermion spectral masses','quark short-distance masses','neutrino Takagi masses','CKM','PMNS','m_beta','m_bb'],
        },
    }
    if any(v != 0 for v in anomaly.values()):
        raise SystemExit(f'Anomaly check failed: {anomaly}')
    if defects != [1,3,5]:
        raise SystemExit(f'Unexpected receiver defects: {defects}')
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'classification':'SECTOR_PREDICTION_MAP','defects':defects,'charges':{k:frac_str(v) for k,v in charges.items()},'anomalies':anomaly,'bcc_ratio':[9,21,25]}, indent=2))

if __name__ == '__main__':
    main()
