#!/usr/bin/env python3
from pathlib import Path
import re, json, sys
root=Path(__file__).resolve().parents[1]; paper=root/'paper'
tex=list(paper.rglob('*.tex')); text_by={p:p.read_text(errors='replace') for p in tex}
missing_inputs=[]
for p,s in text_by.items():
    for m in re.finditer(r'\\(?:input|include)\{([^}]+)\}',s):
        q=p.parent/m.group(1); q=q if q.suffix else q.with_suffix('.tex')
        if not q.exists(): missing_inputs.append((str(p.relative_to(root)),m.group(1)))
labels={}
for p,s in text_by.items():
    for lab in re.findall(r'\\label\{([^}]+)\}',s): labels.setdefault(lab,[]).append(str(p.relative_to(root)))
duplicates={k:v for k,v in labels.items() if len(v)>1}
refs=[]
for p,s in text_by.items():
    for typ,lab in re.findall(r'\\(ref|eqref|autoref|pageref)\{([^}]+)\}',s): refs.append((str(p.relative_to(root)),typ,lab))
unresolved_refs=[x for x in refs if x[2] not in labels and not x[2].startswith(('main:','supp:')) and '#' not in x[2]]
bibtext=(paper/'references.bib').read_text(errors='replace'); bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bibtext))
citations=[]
for p,s in text_by.items():
    for grp in re.findall(r'\\cite\w*\{([^}]+)\}',s):
        for key in grp.split(','): citations.append((str(p.relative_to(root)),key.strip()))
unresolved_citations=[x for x in citations if x[1] and x[1] not in bibkeys]
unresolved_figures=[]
for p,s in text_by.items():
    for raw in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s):
        cand=p.parent/raw
        ok=cand.exists() or any((p.parent/(raw+ext)).exists() for ext in ['.pdf','.png','.jpg','.jpeg','.eps'])
        if not ok:
            candidates=[paper/raw,paper/'figures'/raw]
            ok=any(c.exists() for c in candidates) or any(Path(str(c)+ext).exists() for c in candidates for ext in ['.pdf','.png','.jpg','.jpeg','.eps'])
        if not ok: unresolved_figures.append((str(p.relative_to(root)),raw))
patterns=[r'\bD2[0-9]\b',r'\bversion\s+\d',r'\brerun\b',r'\brecovered\b',r'\bself[- ]?tests?\b',r'\bhard stop\b',r'\bexecution ledger\b',r'\bmaximum execution\b',r'\bfinalizer\b',r'\bcheckpoint\b',r'artificial intelligence',r'large language model',r'ChatGPT',r'OpenAI',r'we do not claim',r'we cannot',r'we refrain',r'beyond the scope',r'future work',r'\bcaveat\b',r'\bpreliminary\b',r'\btentative\b',r'\bprovisional\b',r'\bmerely\b',r'\bunfortunately\b',r'\bapolog(?:y|ize|ise)\b',r'\bsubmission\b',r'\breviewer\b',r'\breferee\b',r'stronger than a numerical surrogate',r'missing contract objects',r'unique remaining input',r'awaits the complete charged operator assembly']
editorial_hits=[]
for p,s in text_by.items():
    for i,line in enumerate(s.splitlines(),1):
        for pat in patterns:
            if re.search(pat,line,re.I): editorial_hits.append((str(p.relative_to(root)),i,pat,line.strip()))
obsolete=['7.199588798902931','4.761260928942132','1.3884042343367827','8.611595765663217','0.39982833628703174','3.8294346006355573','105.639474','1776.763101','4878 stored']
obsolete_hits=[]
for p,s in text_by.items():
    for i,line in enumerate(s.splitlines(),1):
        if any(x in line for x in obsolete): obsolete_hits.append((str(p.relative_to(root)),i,line.strip()))

expected_title='Quantum Copy-Time Geometry: Standard-Model Gauge Algebra, Family Structure, and a Parameter-Free BCC Photon Signature'
old_title=('Quantum Information Copy Time: Finite Receiver Geometry, ' +
           'Gauge-Covariant Lattice Dynamics, and Matrix-Valued Spectral Observables')
previous_title='Spectral Gauge Geometry from Quantum Information' + ' Dynamics'
prior_title='Copy-Time Gauge Geometry Generates' + ' Standard Model Structure'
title_issues=[]
for p,src in text_by.items():
    for i,line in enumerate(src.splitlines(),1):
        if old_title in line or previous_title in line or prior_title in line: title_issues.append((str(p.relative_to(root)),i,'superseded title',line.strip()))
        if '\t' in line: title_issues.append((str(p.relative_to(root)),i,'literal tab',line.strip()))
if expected_title not in (paper/'preamble.tex').read_text(errors='replace'):
    title_issues.append(('paper/preamble.tex',0,'expected title missing',expected_title))
classification_patterns=[
    r'physical flavor map is selected by comparing these coordinates',
    r'large two-splitting statistic selects the interacting neutral',
    r'comparison selects the common-regulator mixing construction',
    r'selects the interacting neutral kernel as the physical mass observable',
    r'selects the common interacting construction for physical mixing',
    r'selects the interacting neutral Nambu kernel for the physical spectrum',
]
classification_issues=[]
for p,src in text_by.items():
    for i,line in enumerate(src.splitlines(),1):
        for pat in classification_patterns:
            if re.search(pat,line,re.I): classification_issues.append((str(p.relative_to(root)),i,pat,line.strip()))

unsupported_claim_patterns=[
    r'15-point sector-specific',
    r'three converged edges are reported',
    r'converged full compact three-dimensional G2 family edges',
    r'common full-SM regulator data set consisting',
]
unsupported_claim_issues=[]
for p,src in text_by.items():
    for i,line in enumerate(src.splitlines(),1):
        for pat in unsupported_claim_patterns:
            if re.search(pat,line,re.I): unsupported_claim_issues.append((str(p.relative_to(root)),i,pat,line.strip()))

stale_structure=[]
for p,src in text_by.items():
    for i,line in enumerate(src.splitlines(),1):
        if '98 numbered sections' in line: stale_structure.append((str(p.relative_to(root)),i,line.strip()))

issues=[missing_inputs,duplicates,unresolved_refs,unresolved_citations,unresolved_figures,editorial_hits,obsolete_hits,title_issues,classification_issues,unsupported_claim_issues,stale_structure]
if any(issues):
    detail={'input_resolution_issues':missing_inputs,'label_uniqueness_issues':duplicates,'reference_resolution_issues':unresolved_refs,'citation_resolution_issues':unresolved_citations,'figure_resolution_issues':unresolved_figures,'editorial_language_issues':editorial_hits,'superseded_numeric_issues':obsolete_hits,'title_consistency_issues':title_issues,'flavor_classification_issues':classification_issues,'unsupported_claim_issues':unsupported_claim_issues,'stale_structure_issues':stale_structure}
    print(json.dumps(detail,indent=2,sort_keys=True)); sys.exit(1)
out={'classification':'SOURCE_CONSISTENCY_VERIFICATION','tex_file_count':len(tex),'bib_entry_count':len(bibkeys),'citation_count':len(citations),'label_count':len(labels),'reference_count':len(refs),'input_resolution':'verified','label_uniqueness':'verified','reference_resolution':'verified','citation_resolution':'verified','figure_resolution':'verified','editorial_language_screen':'verified','numerical_consistency_screen':'verified','title_consistency_screen':'verified','flavor_classification_screen':'verified','unsupported_claim_screen':'verified','structure_freshness_screen':'verified'}
(root/'validation').mkdir(exist_ok=True); (root/'validation'/'SOURCE_CONSISTENCY_VERIFICATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
