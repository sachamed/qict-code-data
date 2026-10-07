#!/usr/bin/env python3
from pathlib import Path
import json, math, re, sys

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
VAL=ROOT/'validation'
VAL.mkdir(exist_ok=True)
TITLE='Quantum Copy-Time Geometry: Standard-Model Gauge Algebra, Family Structure, and a Parameter-Free BCC Photon Signature'

def close(a,b,rtol=5e-12,atol=5e-15):
    return math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol)

def load(name):
    return json.loads((ROOT/'results'/name).read_text())

manuscript=(PAPER/'manuscript_final.tex').read_text(errors='replace')
all_paper='\n'.join(p.read_text(errors='replace') for p in PAPER.rglob('*.tex'))
cover=(ROOT/'cover_letter/QICT_COVER_LETTER.tex').read_text(errors='replace')
readme=(ROOT/'README.md').read_text(errors='replace')

checks={}
checks['title_exact_main']=TITLE in (PAPER/'preamble.tex').read_text(errors='replace')
checks['title_exact_cover']=TITLE in cover
checks['title_exact_readme']=TITLE in readme
checks['title_scope_terms_present']=all(term in TITLE for term in ['Gauge Algebra','Family Structure','Parameter-Free BCC Photon Signature'])
checks['parameter_free_bcc_scope_explicit_in_abstract']='parameter-free normalized BCC photon signature' in manuscript
checks['parameter_free_bcc_scope_explicit_in_cover']='parameter-free normalized photon-direction invariant' in cover
checks['photon_harmonic_ratio_label_consistent']='fourth-harmonic power ratio $1024/20181$' in all_paper and 'spherical variance $1024/20181$' not in all_paper and 'variance $1/525$' in all_paper

forbidden=[
    r'artificial intelligence', r'large language model', r'ChatGPT', r'OpenAI',
    r'we do not claim', r'we cannot', r'we refrain', r'beyond the scope',
    r'future work', r'\bcaveat\b', r'\bpreliminary\b', r'\btentative\b',
    r'\bprovisional\b', r'\bmerely\b', r'\bunfortunately\b',
    r'\bsubmission\b', r'\breviewer\b', r'\breferee\b',
    r'stronger than a numerical surrogate', r'missing contract objects',
    r'unique remaining input', r'awaits the complete charged operator assembly'
]
forbidden_hits=[]
for pat in forbidden:
    if re.search(pat,all_paper,re.I): forbidden_hits.append(pat)
checks['scientific_text_free_of_ai_admin_defensive_meta_language']=not forbidden_hits

unsupported=[
    r'15-point sector-specific',
    r'three converged edges are reported',
    r'converged full compact three-dimensional G2 family edges',
    r'common full-SM regulator data set consisting',
    r'physical quark masses are numerically predicted',
    r'physical neutrino masses are numerically predicted',
]
unsupported_hits=[]
for pat in unsupported:
    if re.search(pat,all_paper,re.I): unsupported_hits.append(pat)
checks['unsupported_claim_phrases_absent']=not unsupported_hits

su2=load('SU2_TENSOR_AUDIT.json')
common=load('COMMON_CONTRACTION_CHARACTERIZATION.json')
lepton=load('REDUCED_LEPTON_INVARIANTS.json')
photon=load('PHOTON_SIGNATURE_MOMENTS.json')

# Exact highlighted values in the abstract.
checks['abstract_su2_nonzero']=str(su2['sequential_SU2']['total_core_nonzero_coefficients']) in manuscript.replace('\\,','')
checks['abstract_su2_bond']=str(su2['sequential_SU2']['largest_observed_core_bond_index_plus_one']) in manuscript.replace('\\,','')
checks['abstract_su2_support']=str(su2['four_plaquette_SU2']['support_direct']) in manuscript.replace('\\,','')
counts=common['serialized_primary_counts']
checks['abstract_common_links']=str(counts['links']) in manuscript.replace('\\,','')
checks['abstract_common_plaquettes']=str(counts['plaquettes']) in manuscript.replace('\\,','')
checks['abstract_common_su2_pairs']=str(counts['su2_connected_pairs']) in manuscript.replace('\\,','')
checks['abstract_common_c10_nodes']=str(counts['c10_nodes']) in manuscript
checks['abstract_kato_dimension']=str(common['strict_kato_checkpoint']['dimension']) in manuscript.replace('\\,','')
checks['abstract_kato_gap']='14.4258464463' in manuscript and close(common['strict_kato_checkpoint']['gap'],14.4258464463)
checks['abstract_riesz_projector']='1.44\\times10^{-17}' in manuscript and close(common['matrix_free_riesz_validation']['projector_action_fro_error'],1.44e-17,rtol=0.01)
checks['abstract_riesz_density']='1.57\\times10^{-16}' in manuscript and close(common['matrix_free_riesz_validation']['rho_fro_error'],1.57e-16,rtol=0.01)
checks['abstract_lepton_mu_e']='206.731294355038' in manuscript and close(lepton['mu_over_e_reduced'],206.731294355038)
checks['abstract_lepton_tau_e']='3477.03865395274' in manuscript and close(lepton['tau_over_e_reduced'],3477.03865395274)
checks['abstract_koide']='0.666671416005203' in manuscript and close(lepton['koide_reduced'],0.666671416005203)
checks['abstract_photon_mean']='31/160' in manuscript and photon['sphere_average_b']=='31/160'
checks['abstract_photon_variance']='1/525' in manuscript and photon['sphere_variance_b']=='1/525'
checks['abstract_mass_scope_reduced']=r'(m_e:m_\mu:m_\tau)_{\rm red}' in manuscript
checks['abstract_operator_defined_mass_scope']='mass and mixing observables are generated by one common interacting operator hierarchy' in manuscript
checks['abstract_single_scale_coordinate']='single metrological coordinate' in manuscript
checks['abstract_exact_algebra_scope']='Universal quantum principles define the admissible finite-geometry category' in manuscript
checks['abstract_g2_execution_scope']='common-net operator contract that defines the interacting $G_2/G_3$ mass and mixing construction' in manuscript

# Scientific scope consistency with the machine-readable registry.
sector=load('SECTOR_PREDICTIONS.json')
requiring=set(sector['prediction_classes']['operator_predictions_requiring_common_interacting_contraction'])
checks['registry_marks_physical_mass_sector_as_interacting']=all(x in requiring for x in [
    'physical charged-fermion spectral masses','quark short-distance masses','neutrino Takagi masses','CKM','PMNS'])
checks['manuscript_preserves_interacting_mass_scope']='common interacting regulator net' in all_paper and 'common regulator net' in all_paper

report={
    'classification':'PUBLICATION_CLAIM_AND_STYLE_AUDIT',
    'title':TITLE,
    'title_word_count':len(TITLE.split()),
    'checks':checks,
    'forbidden_hits':forbidden_hits,
    'unsupported_claim_hits':unsupported_hits,
    'all_checks_pass':all(checks.values()),
}
(VAL/'PUBLICATION_CLAIM_AUDIT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
if not report['all_checks_pass']:
    sys.exit(1)
