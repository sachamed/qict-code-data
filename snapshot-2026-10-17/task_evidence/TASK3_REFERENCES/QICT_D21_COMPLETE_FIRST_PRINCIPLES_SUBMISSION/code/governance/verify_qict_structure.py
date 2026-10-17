#!/usr/bin/env python3
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[2]
main=ROOT/'manuscript/QICT_MAIN_MANUSCRIPT.tex'
truth=json.loads((ROOT/'DOCUMENT_INDEX.json').read_text())
expected_title='Quantum Information Copy Time (QICT): Operator Foundations of Discrete Quantum Time, Chiral Gauge Dynamics, and Intrinsic Spectral Observables'
expected=list('ABCDEFGHIJKLMNO')+list('RSTUVWX')
text=main.read_text(errors='ignore')
sections=re.findall(r'(?m)^\\section\{',text)
supps=sorted((ROOT/'supplements').glob('QICT_Supplement_*.tex'))
letters=[]
for p in supps:
    m=re.match(r'QICT_Supplement_([A-Z])_',p.name)
    if m: letters.append(m.group(1))
active_tex=[main]+supps
forbidden=[]
patterns=[r'QICT'+r'[-_ ]?D21',r'\bV'+r'4R?\b',r'Referee'+r'-Hardened',r'referee'+r'-hardened',r'V'+r'3S',r'Supplements?\s+A.{0,3}Q']
for p in active_tex:
    t=p.read_text(errors='ignore')
    for pat in patterns:
        if re.search(pat,t,re.I): forbidden.append({'file':str(p.relative_to(ROOT)),'pattern':pat})
refaudit=json.loads((ROOT/'references/REFERENCE_REPORT.json').read_text())
checks={
 'title_exact': expected_title in text and truth.get('article_title')==expected_title,
 'section_count_95': len(sections)==95,
 'supplement_set_exact': letters==expected,
 'no_active_PQ': 'P' not in letters and 'Q' not in letters,
 'no_release_branding_in_active_tex': not forbidden,
 'bibliography_missing_zero': len(refaudit.get('missing_keys',[]))==0,
 'main_paths_exist': all((ROOT/truth[k]).exists() for k in ['main_tex','main_pdf','complete_tex','complete_pdf','bibliography','reference_library']),
}
status='PASS' if all(checks.values()) else 'FAIL'
out={'status':status,'checks':checks,'section_count':len(sections),'supplement_letters':letters,'forbidden_hits':forbidden,'reference_audit':refaudit}
(ROOT/'certificates/structure_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if status=='PASS' else 1)
