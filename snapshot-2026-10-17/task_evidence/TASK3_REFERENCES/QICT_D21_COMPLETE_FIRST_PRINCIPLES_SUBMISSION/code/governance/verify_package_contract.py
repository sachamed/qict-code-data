#!/usr/bin/env python3
from pathlib import Path
import json,re,sys
try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader
ROOT=Path(__file__).resolve().parents[2]
truth=json.loads((ROOT/'DOCUMENT_INDEX.json').read_text())
expected=set(list('ABCDEFGHIJKLMNO')+list('RSTUVWX'))
checks={}
# PDF checks
def pdf_a4(path):
    r=PdfReader(str(path)); ok=not r.is_encrypted and len(r.pages)>0
    dims=[]
    for pg in r.pages:
        w=float(pg.mediabox.width); h=float(pg.mediabox.height); dims.append((w,h))
        ok &= abs(w-595.276)<1.5 and abs(h-841.89)<1.5
    return ok,len(r.pages),dims[:2]
main=ROOT/truth['main_pdf']; complete=ROOT/truth['complete_pdf']; monograph=ROOT/truth['monograph_pdf']
checks['main_pdf_a4'],main_pages,_=pdf_a4(main)
checks['complete_pdf_a4'],complete_pages,_=pdf_a4(complete)
checks['monograph_pdf_a4'],monograph_pages,_=pdf_a4(monograph)
letters=set(); supp_meta={}
for p in sorted((ROOT/'supplements').glob('QICT_Supplement_*.pdf')):
    m=re.match(r'QICT_Supplement_([A-Z])_',p.name)
    if not m: continue
    L=m.group(1); letters.add(L); ok,n,_=pdf_a4(p); supp_meta[L]={'file':p.name,'pages':n,'a4':ok}
checks['supplement_set_exact']=letters==expected
checks['all_supplements_a4']=all(v['a4'] for v in supp_meta.values()) and len(supp_meta)==22
checks['no_active_PQ']=not ({'P','Q'} & letters)
checks['entrypoint_exists']=(ROOT/'reproduce_qict.py').exists()
checks['references_exist']=(ROOT/'references/QICT_CITED_BIBLIOGRAPHY.tex').exists() and (ROOT/'references/QICT_REFERENCE_LIBRARY.tex').exists()
checks['figures_exist']=len(list((ROOT/'figures').glob('*')))>0 and len(list((ROOT/'figures_hd').glob('*')))>0
checks['prediction_lock_exists']=(ROOT/'preregistration/QICT_PREDICTION_LOCK.json').exists() and (ROOT/'preregistration/QICT_PREDICTION_LOCK.sha256').exists()
checks['bcc_sealed_prediction_lock_exists']=(ROOT/'preregistration/QICT_BCC_SEALED_PREDICTION_LOCK.json').exists() and (ROOT/'preregistration/QICT_BCC_SEALED_PREDICTION_LOCK.sha256').exists()
checks['scope_documents_present']=(ROOT/'supplements/QICT_Supplement_N_Logical_Status_and_Domains_of_Validity.tex').exists() and (ROOT/'supplements/QICT_Supplement_X_Assumptions_Global_Gauge_Consistency_and_Convergence.tex').exists()
checks['no_journal_specific_cover_letters']=not (ROOT/'cover_letters').exists() or len(list((ROOT/'cover_letters').glob('*')))==0
status='PASS' if all(checks.values()) else 'FAIL'
out={'status':status,'checks':checks,'main_pages':main_pages,'complete_pages':complete_pages,'monograph_pages':monograph_pages,'supplements':supp_meta}
(ROOT/'certificates/package_contract_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if status=='PASS' else 1)
