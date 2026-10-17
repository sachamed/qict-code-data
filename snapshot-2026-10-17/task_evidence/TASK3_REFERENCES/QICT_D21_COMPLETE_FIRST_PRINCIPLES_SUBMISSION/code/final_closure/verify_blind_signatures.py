#!/usr/bin/env python3
from fractions import Fraction
from pathlib import Path
import json, math, hashlib
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'predictions'
OUT.mkdir(exist_ok=True)

# Receiver timing registry
receiver = {
    'deltaT_d10_over_tau': 1,
    'deltaT_d9_over_tau': 3,
    'deltaT_d8_over_tau': 5,
    'T_q_over_tau': 14,
    'T_nu_over_tau': 21,
    'T_126_over_tau': 126,
    'epsilon_min': math.pi/210.0,
    'ramsey_product_target': 1.0,
}
assert [receiver['deltaT_d10_over_tau'], receiver['deltaT_d9_over_tau'], receiver['deltaT_d8_over_tau']] == [1,3,5]

# BCC quadratic angular fingerprint b(n)=(11-8 S4)/32.
b_axis = Fraction(3,32)
b_face = Fraction(7,32)
b_body = Fraction(25,96)
ratio_scale = 96
high_ratio = [int(b_axis*ratio_scale), int(b_face*ratio_scale), int(b_body*ratio_scale)]
assert high_ratio == [9,21,25]
assert (b_face-b_axis)/(b_body-b_face) == 3

# Rotation-invariant exact shape data quoted/derived in the manuscript.
mean_b = Fraction(31,160)
var_b = Fraction(1,525)
power_ratio = Fraction(1024,20181)
skew = -18*math.sqrt(21)/143
kurt = Fraction(5607,2431)

# Quartic high-symmetry fingerprint.
d_axis = -Fraction(17,2048)
d_face = -Fraction(203,6144)
d_body = -Fraction(1963,55296)
quartic_common = 55296
quartic_integer_magnitudes = [int(abs(d_axis)*quartic_common), int(abs(d_face)*quartic_common), int(abs(d_body)*quartic_common)]
assert quartic_integer_magnitudes == [459,1827,1963]

# Seven-shell structural signature from 21=3x7.
krylov_ranks = [3,6,9,12,15,18,21]
assert krylov_ranks[-1] == 21

# Exact Gauss-reduced charged-channel susceptibility.
chi_e = float(mp.gamma(mp.mpf(1)/4)**4/(28*mp.pi**2))
assert abs(chi_e - 0.625268461493279) < 5e-15

registry = {
  'release_date': '2026-09-16',
  'registry_type': 'prospective_data_independent_signature_lock',
  'historical_blinding_note': 'This registry freezes data-independent QICT signatures for future tests. It is not claimed to predate already-published PDG or gamma-ray data.',
  'signatures': [
    {
      'id':'QICT-TIME-001',
      'class':'direct prospective receiver experiment',
      'statement':'Primitive receiver timing ratios are exactly 1:3:5 with the extended integer register 1:3:5:14:21:126.',
      'exact': {'timing_vector':[1,3,5,14,21,126], 'epsilon_min_symbolic':'pi/210'},
      'numeric': {'epsilon_min': receiver['epsilon_min']},
      'fit_parameters':0,
      'falsifier':'No common tau_* fits the six timing observables within their covariance.'
    },
    {
      'id':'QICT-PHOTON-002',
      'class':'direct astrophysical signature',
      'statement':'The BCC tangent has no linear-in-energy photon-speed term; the leading correction is quadratic, subluminal, inversion-even, and has only monopole plus cubic l=4 anisotropy.',
      'exact': {
        'b_formula':'(11-8*(nx^4+ny^4+nz^4))/32',
        'b_range':['3/32','25/96'],
        'high_symmetry_ratio':[9,21,25],
        'critical_points':[6,12,8],
        'gap_ratio':3,
        'mean':'31/160',
        'variance':'1/525',
        'l4_power_over_monopole_power':'1024/20181',
        'centered_skewness':'-18*sqrt(21)/143',
        'centered_kurtosis':'5607/2431'
      },
      'numeric': {
        'b_axis':float(b_axis),'b_face':float(b_face),'b_body':float(b_body),
        'mean':float(mean_b),'variance':float(var_b),'l4_power_ratio':float(power_ratio),
        'skewness':skew,'kurtosis':float(kurt)
      },
      'fit_parameters':0,
      'falsifier':'A robust linear LIV term, a superluminal leading quadratic coefficient in the same mapping, or a normalized quadratic sky map inconsistent with the locked cubic l=4 fingerprint.'
    },
    {
      'id':'QICT-PHOTON-004',
      'class':'direct higher-order astrophysical signature',
      'statement':'The quartic photon angular map is locked once the quadratic BCC frame is fixed; it has cubic-singlet support only in l=0,4,6,8.',
      'exact': {
        'd_axis':'-17/2048','d_face':'-203/6144','d_body':'-1963/55296',
        'integer_magnitude_ratio_common_denominator_55296':quartic_integer_magnitudes,
        'allowed_harmonics':[0,4,6,8]
      },
      'numeric': {'d_axis':float(d_axis),'d_face':float(d_face),'d_body':float(d_body)},
      'fit_parameters':0,
      'falsifier':'An independently measured quartic angular map requires angular coefficients outside the locked cubic-singlet pattern after the BCC orientation and overall scale are fixed by the quadratic map.'
    },
    {
      'id':'QICT-C21-007',
      'class':'internal structural signature',
      'statement':'For a rank-three source on the exact CRT receiver factorization C21=C3 x C7, the primitive block-Krylov dimensions are 3,6,9,12,15,18,21 and close after seven shells.',
      'exact': {'krylov_ranks':krylov_ranks,'shell_count':7},
      'fit_parameters':0,
      'falsifier':'The declared microscopic receiver operator and source fail to reproduce the seven-shell cyclic dimension sequence.'
    },
    {
      'id':'QICT-FAM-003',
      'class':'internal spectral signature',
      'statement':'A rank-three receiver Feshbach ancestry defines exactly three primary family branches; complementary finite-volume poles are satellites unless they are connected to the receiver roots by the locked homotopy.',
      'exact': {'receiver_rank':3,'primary_branch_count':3},
      'fit_parameters':0,
      'falsifier':'The primary Riesz cluster loses separation irreparably in the regulator limit or its dimension differs from three.'
    },
    {
      'id':'QICT-MASS-006',
      'class':'internal chiral mass signature',
      'statement':'A charged Dirac mass is carried by the residue-normalized L<->R block of a rank-six projected inverse propagator, not by a rank-three family-only propagation block.',
      'exact': {'source_rank':6,'chiral_blocks':['LL','LR','RL','RR'],'family_mass_matrix_dimension':[3,3]},
      'fit_parameters':0,
      'falsifier':'A regulator-stable charged mass is generated with a vanishing renormalized LR block, or the full electroweak contraction fails the rank-six Schur definition.'
    },
    {
      'id':'QICT-WEYL-000',
      'class':'internal symmetry no-go',
      'statement':'Exact invariance under the irreducible family Weyl pair X,Z forces the family mass coefficient to be proportional to I3 and hence degenerate.',
      'exact': {'commutant_complex_dimension':1,'consequence':'Y_e = y I_3'},
      'fit_parameters':0,
      'falsifier':'A nonscalar 3x3 family coefficient commutes exactly with both irreducible Weyl generators.'
    },
    {
      'id':'QICT-GAUSS-CHI',
      'class':'internal reduced-channel signature',
      'statement':'The Gauss-reduced tangent C11 electric edge half-curvature is chi_q^E=q^2 Gamma(1/4)^4/(1008 pi^2); for the electron normalization chi_e^E=Gamma(1/4)^4/(28 pi^2). It cancels from e-mu-tau splittings because their electromagnetic charge is common.',
      'exact': {'chi_e_symbolic':'Gamma(1/4)^4/(28*pi^2)'},
      'numeric': {'chi_e':chi_e},
      'fit_parameters':0,
      'falsifier':'The specified Gauss-reduced tangent channel fails the exact q^2 curvature law.'
    }
  ],
  'non_predictions': {
    'charged_lepton_mass_ratios':'NOT_LOCKED_NUMERICALLY',
    'absolute_tau_star':'NOT_LOCKED_NUMERICALLY',
    'physical_CKM_PMNS':'NOT_LOCKED_NUMERICALLY'
  }
}

lock_path = OUT / 'PROSPECTIVE_BLIND_PREDICTION_LOCK.json'
lock_path.write_text(json.dumps(registry, indent=2) + '\n')
lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
(OUT/'PROSPECTIVE_BLIND_PREDICTION_LOCK.sha256').write_text(f'{lock_sha}  {lock_path.name}\n')

# Existing external context, explicitly separate from the locked registry.
comparison = {
  'comparison_date':'2026-09-16',
  'lock_sha256':lock_sha,
  'scope_note':'Existing public measurements are used only as context. The normalized BCC angular fingerprint and receiver timing register remain prospective tests.',
  'existing_context': {
    'LHAASO_2024': {
      'result':'No significant energy-dependent vacuum speed in GRB 221009A; 95% CL E_QG,1 > 10 E_Pl and E_QG,2 > 6e-8 E_Pl in the collaboration LIV parametrization.',
      'relation_to_QICT':'Consistent with a zero linear coefficient; constrains the quadratic amplitude under an explicit mapping, but does not test the locked normalized BCC angular fingerprint.'
    },
    'PDG_2026': {
      'role':'External comparison only for already-computed negative mass proxies; no PDG mass is inserted into the locked microscopic predictions.',
      'charged_lepton_mass_prediction_status':'not yet available from the full 3D electroweak rank-six contraction'
    }
  }
}
(OUT/'POST_LOCK_EXISTING_DATA_CONTEXT.json').write_text(json.dumps(comparison, indent=2)+'\n')

print(json.dumps({'status':'PASS','prediction_lock_sha256':lock_sha,'signature_count':len(registry['signatures'])}, indent=2))
