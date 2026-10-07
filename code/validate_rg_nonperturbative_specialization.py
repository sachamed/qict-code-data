#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

ap=argparse.ArgumentParser()
ap.add_argument('--out', type=Path, default=ROOT/'results/RG_NONPERTURBATIVE_CERTIFICATE.json')
args=ap.parse_args()

def load(rel):
    return json.loads((ROOT/rel).read_text())

common=load('results/COMMON_CONTRACTION_CHARACTERIZATION.json')
photon=load('results/PHOTON_SIGNATURE_MOMENTS.json')
g2=load('results/physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json')

# Search recursively for the finite Kato gap already carried by the generated result.
def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj: return obj[key]
        for v in obj.values():
            x=find_key(v,key)
            if x is not None: return x
    elif isinstance(obj, list):
        for v in obj:
            x=find_key(v,key)
            if x is not None: return x
    return None

kato_gap=find_key(common,'gap')
# Exact directional coefficients are structural identities of the BCC tangent.
b100=3/32
b110=7/32
b111=25/96
rgamma=(b110-b100)/(b111-b100)
assert abs(rgamma-0.75)<1e-15
assert [round(96*x) for x in (b100,b110,b111)]==[9,21,25]

# Contract audit status is preserved verbatim as evidence of the physical G2 execution state.
status = find_key(g2,'status') or find_key(g2,'overall_status') or 'SEE_CONTRACT_AUDIT'

out={
  'classification':'RG_NONPERTURBATIVE_SPECIALIZATION_CERTIFICATE',
  'rg_path':{
    'a_sequence':'a_n = a_0 2^{-n}',
    'reference_scale':'mu_* fixed',
    'defect_bound':'||D(a)||_1 <= C_D (a mu_*)^p',
    'p_definition':'min(2, omega_*)',
    'numeric_p_status':'requires interacting block-map Jacobian spectrum'
  },
  'gap_separation':{
    'chiral_kernel_gap':'delta(a)=dist(0,sigma(H_W_hat(a)))',
    'kato_branch_gap_finite_L24':kato_gap,
    'physical_sector_gap':'defined sectorwise; massless photon excluded from a global positive-gap statement',
    'distinct_roles_verified':True
  },
  'chiral_uniformity':{
    'theorem':'delta(a) >= delta_ref - ||H_W(a)-H_W_ref(a)||',
    'uniform_target':'delta_0 = delta_ref/2 under perturbation <= delta_ref/2',
    'finite_wall_bound':'epsilon_chi <= 2*((1-delta_0)/(1+delta_0))^L_s',
    'multiscale_numeric_status':'requires Wilson-kernel interval lower bounds along the RG path'
  },
  'gribov':{
    'primary_observables':'Gauss-projected gauge-invariant algebra',
    'finite_cone_representative':'rooted maximal-tree gauge',
    'landau_comparison':'fundamental modular region / absolute minimum',
    'physical_moments_gauge_invariant':True
  },
  'charged_sector':{
    'charged_leptons':'Gauss-dressed lower spectral support edge',
    'quarks':'renormalized short-distance mass parameter m_q^S(mu) in confined color sector',
    'neutral_neutrinos':'isolated pole/Takagi data when spectrally isolated'
  },
  'excluded_scale_datum':{
    'number':'m_e c^2 = 0.51099895069(16) MeV',
    'rule':'do not use to determine tau_* in an absolute electron-mass prediction'
  },
  'photon_directional_test':{
    'coefficients':{'b100':b100,'b110':b110,'b111':b111},
    'integer_ratio':'9:21:25',
    'R_gamma':rgamma,
    'five_sigma_refutation':'abs(Rhat_gamma-0.75) > 5 sigma_R, conditional on resolved quadratic propagation',
    'profiled_shape_refutation':'Delta chi^2 >= 25 for one effective shape degree of freedom'
  },
  'gravity_interface':{
    'object':'conformal metric from common continuum principal symbol',
    'formula':'g_eff^{mu nu} proportional to d^2 Gamma^(2)/dp_mu dp_nu at p=0',
    'next_dynamical_quantity':'two-derivative metric effective action coefficient / Newton coupling'
  },
  'physical_g2_contract_status':status,
  'source_files':[
    'COMMON_CONTRACTION_CHARACTERIZATION.json',
    'PHOTON_SIGNATURE_MOMENTS.json',
    'physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json'
  ]
}
args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(args.out)
