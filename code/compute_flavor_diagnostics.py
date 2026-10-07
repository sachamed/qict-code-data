#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def angles_from_moduli_and_absJ(B: np.ndarray, Jabs: float) -> dict:
    """Extract standard 3-angle parameters and the principal CP branch from |V_ij|^2 and |J|.

    The sign of J is not contained in |J|, so the conjugate branch 2*pi-delta is equally compatible.
    """
    s13 = math.sqrt(float(B[0,2])); c13 = math.sqrt(max(0.0, 1.0-s13*s13))
    s12 = math.sqrt(float(B[0,1]))/c13; c12 = math.sqrt(max(0.0,1.0-s12*s12))
    s23 = math.sqrt(float(B[1,2]))/c13; c23 = math.sqrt(max(0.0,1.0-s23*s23))
    den = c12*c23*c13*c13*s12*s23*s13
    sin_delta = max(-1.0,min(1.0,Jabs/den))
    cos_delta = (float(B[1,0]) - s12*s12*c23*c23 - c12*c12*s23*s23*s13*s13)/(2*s12*c12*s23*c23*s13)
    cos_delta = max(-1.0,min(1.0,cos_delta))
    delta = math.atan2(sin_delta,cos_delta)
    return {
        'theta12_deg': math.degrees(math.asin(s12)),
        'theta13_deg': math.degrees(math.asin(s13)),
        'theta23_deg': math.degrees(math.asin(s23)),
        'delta_principal_rad_from_absJ': delta,
        'delta_principal_deg_from_absJ': math.degrees(delta),
        'delta_conjugate_rad_from_absJ': 2*math.pi-delta,
        'J_denominator': den,
        'sin_delta_from_absJ': sin_delta,
        'cos_delta_from_moduli': cos_delta,
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmarks',required=True)
    ap.add_argument('--references',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    b=json.loads(Path(args.benchmarks).read_text())
    ref=json.loads(Path(args.references).read_text())
    Bq=np.asarray(b['B_ud'],float); Bl=np.asarray(b['B_enu'],float)
    aq=angles_from_moduli_and_absJ(Bq,float(b['abs_J_ud']))
    al=angles_from_moduli_and_absJ(Bl,float(b['abs_J_enu']))

    ckm=ref['ckm']
    ckm_angles={
        'theta12_deg': math.degrees(math.asin(ckm['sin_theta12'])),
        'theta13_deg': math.degrees(math.asin(ckm['sin_theta13'])),
        'theta23_deg': math.degrees(math.asin(ckm['sin_theta23'])),
        'delta_rad': ckm['delta_rad'],
        'delta_deg': math.degrees(ckm['delta_rad'])
    }
    pmns=ref['pmns_normal_ordering']

    s=np.asarray(b['neutral_bare_takagi_singular_values'],float)
    dm21=float(s[1]**2-s[0]**2); dm31=float(s[2]**2-s[0]**2); ratio=dm21/dm31
    obs31=float(pmns['delta_m3l_eV2'])
    scale=math.sqrt(obs31/dm31)
    m=(s*scale)
    dm21_pred=float(m[1]**2-m[0]**2)
    # Auxiliary beta-decay observable uses only the finite B_eν first-row moduli.
    m_beta=math.sqrt(float(np.sum(Bl[0,:]*m*m)))
    terms=Bl[0,:]*m
    m_bb_max=float(np.sum(terms)); m_bb_min=max(0.0,float(2*np.max(terms)-np.sum(terms)))

    su=np.asarray(b['receiver_singular_values']['u'],float)
    sd=np.asarray(b['receiver_singular_values']['d'],float)
    out={
      'classification':'FINITE_FLAVOR_DIAGNOSTIC_COORDINATES',
      'scope':'finite receiver coordinates evaluated alongside the separately defined common interacting mass and mixing maps',
      'finite_receiver_ckm_like': aq,
      'finite_receiver_pmns_like': al,
      'experimental_ckm_reference': ckm_angles,
      'experimental_pmns_reference': {k:pmns[k] for k in ['theta12_deg','theta13_deg','theta23_deg','delta_deg']},
      'angle_differences_deg': {
        'ckm': {k: aq[k]-ckm_angles[k] for k in ['theta12_deg','theta13_deg','theta23_deg']},
        'pmns': {k: al[k]-pmns[k] for k in ['theta12_deg','theta13_deg','theta23_deg']}
      },
      'finite_neutral_shape': {
        'takagi_singular_values_dimensionless': s.tolist(),
        'delta_s2_21': dm21,
        'delta_s2_31': dm31,
        'ratio_delta_s2_21_over_31': ratio,
        'NuFIT_ratio_delta_m2_21_over_3l': float(pmns['delta_m21_eV2']/pmns['delta_m3l_eV2']),
        'ratio_factor_relative_to_NuFIT': float(ratio/(pmns['delta_m21_eV2']/pmns['delta_m3l_eV2'])),
        'single_atmospheric_splitting_normalization': {
          'normalization_eV_per_dimensionless_unit': scale,
          'masses_eV': m.tolist(),
          'sum_masses_eV': float(m.sum()),
          'predicted_delta_m21_eV2': dm21_pred,
          'target_delta_m21_eV2': float(pmns['delta_m21_eV2']),
          'm_beta_aux_eV': m_beta,
          'm_bb_aux_phase_interval_eV': [m_bb_min,m_bb_max],
          'interpretation':'one-parameter finite-shape coordinate normalized to the atmospheric splitting and reported as a source-space comparison quantity'
        }
      },
      'finite_quark_receiver_hierarchy': {
        'up_singular_values':su.tolist(),
        'down_singular_values':sd.tolist(),
        'up_ratios': {'s2_over_s1':float(su[1]/su[0]),'s3_over_s2':float(su[2]/su[1]),'s3_over_s1':float(su[2]/su[0])},
        'down_ratios': {'s2_over_s1':float(sd[1]/sd[0]),'s3_over_s2':float(sd[2]/sd[1]),'s3_over_s1':float(sd[2]/sd[0])},
        'interpretation':'dimensionless finite-carrier diagnostics; quark masses are defined by the renormalized short-distance interacting two-point function'
      },
      'quark_reference_values':ref['quark_masses'],
      'absolute_neutrino_reference_constraints':ref['absolute_neutrino_constraints']
    }
    Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'classification':out['classification'],'ckm_like':aq,'pmns_like':al,'neutral_ratio':ratio,'normalized_neutrino_masses_eV':m.tolist()},indent=2))

if __name__=='__main__': main()
