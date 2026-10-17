#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parent/'paper'; RESULTS=ROOT/'results'
THEOREMS={
 'thm:s92-universal-scope':'scope of universal kinematics',
 'thm:s92-receiver-algebra':'receiver-class algebra uniqueness',
 'thm:s92-three-generation':'primitive Weyl-carrier multiplicity',
 'thm:s92-car-unique':'strict CAR uniqueness',
 'thm:s92-kato-vacuum':'thermodynamic state selection',
 'thm:s92-mass-scale':'interacting fermion spectrum map',
 'thm:s92-mixing':'charged-current mixing map',
 'thm:s92-scheme':'regulator universality criterion',
 'thm:s92-continuum-reconstruction':'continuum reconstruction criterion',
 'thm:s92-spectral-stability':'spectral stability',
 'thm:s92-perturbative-recovery':'perturbative recovery',
}
BIB={'HaagKastler1964','LiebRobinson1972','HastingsKoma2006','BachmannEtAl2012','BuchholzInfra','Jarlskog','NielsenNinomiya1981','GinspargWilson1982','Neuberger1998','HJL1999','LuscherAbelian1999','LuscherAllOrders2000','OsterwalderSchrader1973','tHooftYM1971','tHooftVeltman1972'}
def load(p): return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default=str(RESULTS/'CONTINUUM_CONSISTENCY_CERTIFICATE.json')); ap.add_argument('--matrix',default=str(RESULTS/'CONTINUUM_CONSISTENCY_MATRIX.csv')); a=ap.parse_args()
 supp=(PAPER/'sections/supplement/s92.tex').read_text(errors='replace'); main=(PAPER/'manuscript_final.tex').read_text(errors='replace'); bib=(PAPER/'references.bib').read_text(errors='replace')
 labels=set(re.findall(r'\\label\{([^}]+)\}',supp)); bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
 theorem_checks={k:k in labels for k in THEOREMS}; bib_checks={k:k in bibkeys for k in sorted(BIB)}
 riesz=load(RESULTS/'RIESZ_NUMERICAL_VALIDATION.json'); common=load(RESULTS/'COMMON_CONTRACTION_CHARACTERIZATION.json'); l24=load(RESULTS/'L24_PRECONTRACTION_VALIDATION.json')
 exact={
  'l24_rank6_precontraction_certified': l24.get('manifest_consistency')=='verified' and l24.get('array_shape_consistency')=='verified',
  'riesz_projector_fro_error':float(riesz['projector_fro_error']), 'riesz_density_fro_error':float(riesz['density_fro_error']), 'riesz_uniform_contour_gap':float(riesz['uniform_contour_gap']),
  'finite_kato_state_selection_hamiltonian_gap':float(common['strict_kato_checkpoint']['gap']), 'finite_kato_state_norm':float(common['strict_kato_checkpoint']['state_norm'])}
 stages=[
 ('1','Receiver algebra','MODEL_CLASS_THEOREM','thm:s92-receiver-algebra'),('2','Primitive Weyl-carrier multiplicity','MODEL_CLASS_THEOREM','thm:s92-three-generation'),('3','CAR matter lift','THEOREM','thm:s92-car-unique'),('4','State selection','CONDITIONAL_CONTINUUM_THEOREM','thm:s92-kato-vacuum'),('5','Fermion spectrum map','OPERATOR_DEFINED','thm:s92-mass-scale'),('6','Mixing/current map','OPERATOR_DEFINED','thm:s92-mixing'),('7','Regulator universality','CONDITIONAL_CONTINUUM_THEOREM','thm:s92-scheme'),('8','Continuum reconstruction','CONDITIONAL_RECONSTRUCTION_THEOREM','thm:s92-continuum-reconstruction'),('9','Spectral stability','THEOREM','thm:s92-spectral-stability'),('10','Perturbative recovery','CONDITIONAL_STANDARD_CLASS','thm:s92-perturbative-recovery')]
 structural=all(theorem_checks.values()) and all(bib_checks.values()) and 'sec:architecture-continuum' in main
 out={'validation_scope':'Source-label, bibliography and finite-result inventory; continuum existence and theorem validity are separate mathematical questions.','classification':'CONTINUUM_CONSISTENCY_CERTIFICATE','structural_validation':'PASS' if structural else 'FAIL','theorem_labels':theorem_checks,'bibliographic_anchors':bib_checks,'finite_regulator_certificates':exact,'common_net_status':{'operator_map':'SPECIFIED','numerical_role':'Finite-regulator certificates validate components; physical masses and mixing require the declared common-net interacting limits.','no_external_mass_fit_in_definition':True},'theorem_chain':[{'stage':a,'subject':b,'status':c,'label':d} for a,b,c,d in stages],'master_defect_vector':['hierarchy_cauchy','vacuum_boundary','cross_regulator','positivity','covariance','spectrum','locality','cone','infrared','spectral_edge','slavnov_taylor'],'implication':'Vanishing of the declared defect vector under the hypotheses of the reconstruction theorem yields a regulator-independent local continuum realization with stable spectral observables.'}
 Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with Path(a.matrix).open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['stage','subject','status','theorem_label']); w.writerows(stages)
 print(json.dumps(out,indent=2,sort_keys=True)); return 0 if structural else 1
if __name__=='__main__': raise SystemExit(main())
