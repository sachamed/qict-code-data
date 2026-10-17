#!/usr/bin/env python3
"""Whole-package readability and source-integrity audit for the QICT submission.

Reads every package file byte-for-byte.  In addition, parses all JSON, CSV and
Python sources, decodes authored text sources as UTF-8, and checks the active
source tree for stale paths/editorial residue relevant to the publication package.
"""
from __future__ import annotations
from pathlib import Path
import ast, csv, hashlib, json, re, sys
try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader

ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'/'FULL_SUBMISSION_SOURCE_AUDIT.json'
TEXT_EXT={'.tex','.md','.txt','.bib','.py','.json','.csv'}
IGNORE_DIRS={'.git','__pycache__'}
rows=[]; errors=[]; parsed={'json':0,'csv':0,'python':0,'pdf':0,'text':0}; total_bytes=0
active_text=[]
for p in sorted(x for x in ROOT.rglob('*') if x.is_file()):
    rel=p.relative_to(ROOT).as_posix()
    if any(part in IGNORE_DIRS for part in p.relative_to(ROOT).parts):
        continue
    try:
        b=p.read_bytes()
    except Exception as e:
        errors.append({'file':rel,'kind':'read','error':repr(e)}); continue
    total_bytes += len(b)
    rows.append({'file':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    suf=p.suffix.lower()
    if suf in TEXT_EXT:
        try:
            t=b.decode('utf-8'); parsed['text']+=1
            if suf not in {'.log','.aux','.out'}: active_text.append((rel,t))
        except Exception as e:
            errors.append({'file':rel,'kind':'utf8','error':repr(e)}); continue
        if suf=='.json':
            try: json.loads(t); parsed['json']+=1
            except Exception as e: errors.append({'file':rel,'kind':'json','error':repr(e)})
        elif suf=='.csv':
            try:
                list(csv.reader(t.splitlines())); parsed['csv']+=1
            except Exception as e: errors.append({'file':rel,'kind':'csv','error':repr(e)})
        elif suf=='.py':
            try: ast.parse(t,filename=rel); parsed['python']+=1
            except Exception as e: errors.append({'file':rel,'kind':'python_ast','error':repr(e)})
    if suf=='.pdf':
        try:
            r=PdfReader(str(p)); _=len(r.pages); parsed['pdf']+=1
        except Exception as e: errors.append({'file':rel,'kind':'pdf','error':repr(e)})

# Active-source publication checks.  Generated .log/.aux files are excluded.
forbidden={
    'stale_c10_path':r'c10_bcc_gauss_coulomb_R1_R24\.csv',
    'stale_c11_name':r'(tangent_mass_response|TANGENT_MASS_RESPONSE)',
    'editorial_todo':r'(?i)\b(TODO|TBD|FIXME|lorem ipsum)\b',
    'meta_generation_marker':r'(?i)(assistant-generated|machine-generated prose)',
}
hits={k:[] for k in forbidden}
for rel,t in active_text:
    # Exclude the audit script's own regex literals from self-detection.
    if rel.endswith('audit_full_submission.py'): continue
    for k,pat in forbidden.items():
        if re.search(pat,t): hits[k].append(rel)

# Required new closure artifacts and exact constants.
required=[
 'code/closure/verify_c11_gauss_reduced_edge_curvature.py',
 'data/c11_gauss_reduced_edge_curvature_convergence.csv',
 'certificates/C11_GAUSS_REDUCED_EDGE_CURVATURE.json',
 'code/closure/verify_c11_full_family_projector_invariant.py',
 'data/c11_full_family_projector_overlap.csv',
 'certificates/C11_FULL_FAMILY_PROJECTOR_INVARIANT.json',
 'code/closure/verify_c11_family_current_observable.py',
 'data/c11_family_current_observables.csv',
 'certificates/C11_FAMILY_CURRENT_OBSERVABLE.json',
 'manuscript/QICT_MAIN_MANUSCRIPT.tex',
 'supplements/QICT_Supplement_L_C11_Intrinsic_Observable_Completion.tex',
]
required_present={x:(ROOT/x).is_file() for x in required}
main=(ROOT/'manuscript/QICT_MAIN_MANUSCRIPT.tex').read_text()
suppL=(ROOT/'supplements/QICT_Supplement_L_C11_Intrinsic_Observable_Completion.tex').read_text()
closure_checks={
 'exact_formula_in_main':'\\Gamma(1/4)^4}{28\\pi^2}' in main,
 'spectral_translation_theorem':'Exact C11 spectral translation in the Gauss-reduced tangent channel' in suppL,
 'noncommuting_scope_guard':'unreduced C8J Floquet product' in suppL,
 'R48_bound':'8.147\\times10^{-3}' in suppL,
 'full_c11_family_reduction_theorem':'Exact family reduction of the full compact gauged-Julia dynamics' in suppL,
 'full_c11_zero_leakage':'\\mathcal L_{f,N}:=' in suppL and '=0' in suppL,
 'full_c11_quark_purity':'2.54781958277858' in suppL,
 'full_c11_lepton_purity':'2.40076284729224' in suppL,
 'full_c11_ckm_pmns_scope_guard':'They do not identify $L_u^\\dagger L_d$ or $L_e^\\dagger L_\\nu$ with standard CKM or PMNS matrices' in suppL,
 'c11_inclusive_current_theorem':'Inclusive C11 family-current fraction' in suppL and '\\mathfrak B_{fg}^{ab}' in suppL,
 'c11_phase_interference_prediction':'Operational family-current prediction from the full C11 channel geometry' in suppL and '3.7127907325\\times10^{-5}' in suppL and '1.5455105450\\times10^{-4}' in suppL,
 'c11_standard_mixing_scope_guard':r'Failure of either $\Delta_{\rm AH}=0$ or $h_{\min}>0$ falsifies this direct promotion' in suppL,
 'c11_source_vertex_matching':'C11 source--response amputation identity' in suppL and 'Exact polar matching criterion' in suppL,
 'c11_family_mass_ratio_theorem':'Family-resolved C11/G2 spectral-edge ratio theorem' in (ROOT/'supplements/QICT_Supplement_V_Physical_Mass_and_Mixing_Definitions.tex').read_text(errors='ignore'),
}
status='PASS' if not errors and not any(hits.values()) and all(required_present.values()) and all(closure_checks.values()) else 'FAIL'
out={
 'status':status,
 'files_read_byte_for_byte':len(rows),
 'total_bytes_read':total_bytes,
 'parsed_counts':parsed,
 'required_present':required_present,
 'closure_checks':closure_checks,
 'forbidden_source_hits':hits,
 'errors':errors,
 'note':'Every package file was read byte-for-byte; authored text, JSON, CSV, Python and PDF containers received additional format checks. This is a package/source integrity audit, not an empirical validation of the physical theory.'
}
CERT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
sys.exit(0 if status=='PASS' else 1)
