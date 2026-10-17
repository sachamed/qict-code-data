#!/usr/bin/env python3
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[2]
main=(ROOT/'manuscript/QICT_MAIN_MANUSCRIPT.tex').read_text(errors='ignore')
A=(ROOT/'supplements/QICT_Supplement_A_Finite_Spectral_and_C7_Proofs.tex').read_text(errors='ignore')
W=(ROOT/'supplements/QICT_Supplement_W_Operatorial_Interacting_Chiral_QCA.tex').read_text(errors='ignore')
X=(ROOT/'supplements/QICT_Supplement_X_Assumptions_Global_Gauge_Consistency_and_Convergence.tex').read_text(errors='ignore')
alltex='\n'.join(p.read_text(errors='ignore') for p in [ROOT/'manuscript/QICT_MAIN_MANUSCRIPT.tex', *sorted((ROOT/'supplements').glob('QICT_Supplement_*.tex'))])
checks={
 'qnu_upstream_rank_derivation': all(t in A for t in [r'r=7', r'q=2r=14', r'q_\nu=2r+r=3r=21']),
 'totient_is_certification_not_selector': ('cyclotomic certification' in A.lower() and 'after' in A.lower() and r'\varphi(21)=12' in A),
 'old_degree12_selector_removed': not bool(re.search(r'C3\s+posits.{0,160}degree\s+twelve',A,re.I|re.S)),
 'finite_algebra_local_labels_F1_F5': all(f'(F{i})' in main for i in range(1,6)),
 'global_A1_A8_explicit': all(re.search(rf'\bA{i}\b',X) for i in range(1,9)),
 'local_LP1_LP2_explicit': all(re.search(rf'\bLP{i}\b',X) for i in (1,2)),
 'thermodynamic_qca_automorphism_present': ('quasi-local' in W.lower() and '*-automorphism' in W.replace(r'\ast','*').lower() or 'automorphism' in W.lower()),
 'c11_spectral_measure_uniqueness_present': ('unique positive' in alltex.lower() and 'spectral measure' in alltex.lower()),
 'previous_qict_relation_explicit': ('Relation to the 2026 QICT article' in main),
 'bcc_prior_art_explicit': ('not claimed as a new lattice classification result' in main.lower() and 'DArianoErba2017' in main and 'Raynal2017' in main),
 'sealed_specification_naming': ((ROOT/'preregistration/QICT_BCC_SEALED_PREDICTION.md').exists() and not (ROOT/'preregistration/QICT_BCC_BLIND_PREDICTION.md').exists()),
 'canonical_section_registry_exists': (ROOT/'data/canonical_95_section_titles.json').exists(),
 'integrity_report_exists': (ROOT/'validation/INTEGRITY_REPORT.json').exists(),
}
status='PASS' if all(checks.values()) else 'FAIL'
out={'status':status,'checks':checks}
(ROOT/'certificates'/'DEPENDENCY_SYNCHRONIZATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if status=='PASS' else 1)
