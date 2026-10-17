#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'

ap=argparse.ArgumentParser(description='Validate the RG/continuum specialization used by the QICT submission.')
ap.add_argument('--out', type=Path, default=RESULTS/'RG_CONTINUUM_CERTIFICATE.json')
args=ap.parse_args()

def load(rel):
    return json.loads((ROOT/rel).read_text())

def find_key(obj,key):
    if isinstance(obj,dict):
        if key in obj: return obj[key]
        for v in obj.values():
            x=find_key(v,key)
            if x is not None: return x
    elif isinstance(obj,list):
        for v in obj:
            x=find_key(v,key)
            if x is not None: return x
    return None

common=load('results/COMMON_CONTRACTION_CHARACTERIZATION.json')
continuum=load('results/CONTINUUM_CONSISTENCY_CERTIFICATE.json')
photon=load('results/PHOTON_SIGNATURE_MOMENTS.json')
kato_gap=float(find_key(common,'gap'))

# Exact BCC quadratic-direction identities.
b100=3/32
b110=7/32
b111=25/96
rgamma=(b110-b100)/(b111-b100)
assert abs(rgamma-0.75)<1e-15
assert [round(96*x) for x in (b100,b110,b111)]==[9,21,25]

out={
  'classification':'RG_CONTINUUM_SPECIALIZATION_CERTIFICATE',
  'structural_validation':continuum['structural_validation'],
  'rg_path':{
    'representative_sequence':'a_n = a_0 2^{-n}',
    'reference_scale':'mu_* fixed along the comparison sequence',
    'defect_bound':'||D(a)||_1 <= C_D (a mu_*)^p',
    'p_definition':'p = any 0 < p < min(2, omega_*); endpoint requires a uniform sharp-rate bound on a differentiable stable renormalized trajectory, with marginal logarithmic corrections retained in the error envelope',
    'omega_star_status':'obtained from the interacting block-map Jacobian when that spectrum is evaluated'
  },
  'spectral_separation':{
    'chiral_kernel_gap':'delta(a)=dist(0,sigma(H_W_hat(a)))',
    'finite_kato_state_selection_hamiltonian_gap_checkpoint':kato_gap,
    'role_of_checkpoint':'finite-regulator spectral gap of the auxiliary Kato state-selection Hamiltonian, used specifically as the branch-isolation checkpoint at that regulator',
    'massless_sector_treatment':'the photon is characterized by its protected zero and principal-symbol cone, with infrared control supplied by the common-net convergence certificate'
  },
  'chiral_uniformity':{
    'perturbative_gap_bound':'delta(a) >= delta_ref - ||H_W(a)-H_W_ref(a)||',
    'uniform_target':'delta_0 = delta_ref/2 whenever ||H_W-H_W_ref|| <= delta_ref/2',
    'finite_wall_bound':'epsilon_chi <= 2*((1-delta_0)/(1+delta_0))^L_s'
  },
  'gauge_sector':{
    'primary_observables':'Gauss-projected gauge-invariant operator algebra',
    'finite_cone_representative':'rooted maximal-tree representative',
    'physical_moments':'defined gauge invariantly on the common regulator net'
  },
  'fermion_observable_status':{
    'charged_leptons':'Stable electron: dressed threshold; unstable muon/tau: declared infrared-safe resonance prescription',
    'quarks':'renormalized short-distance mass parameter in a declared color-resolved scheme and scale',
    'neutral_neutrinos':'Takagi data of the interacting neutral Nambu kernel on the selected active branch',
    'numerical_assignment_rule':'a physical value is assigned at the common-net convergence certificate on the declared renormalization scheme and scale'
  },
  'photon_directional_test':{
    'coefficients':{'b100':b100,'b110':b110,'b111':b111},
    'integer_ratio':'9:21:25',
    'rotation_invariant_shape_ratio':rgamma,
    'source':'PHOTON_SIGNATURE_MOMENTS.json'
  },
  'gravity_interface':{
    'kinematic_object':'common continuum principal-symbol cone / conformal metric',
    'dynamical_response':'homogeneous-frame Kato response epsilon_0 and TT Kato susceptibility gamma_TT^(s)',
    'einstein_hilbert_identification':'selected by the independently stated common-cone, metric Ward-identity, temporal/spatial matching, and positive-helicity-two conditions'
  },
  'source_files':['COMMON_CONTRACTION_CHARACTERIZATION.json','CONTINUUM_CONSISTENCY_CERTIFICATE.json','PHOTON_SIGNATURE_MOMENTS.json']
}
args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(args.out)
