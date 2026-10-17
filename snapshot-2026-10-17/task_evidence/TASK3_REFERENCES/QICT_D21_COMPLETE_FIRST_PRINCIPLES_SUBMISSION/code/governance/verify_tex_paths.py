#!/usr/bin/env python3
from pathlib import Path
import re, json, sys
ROOT=Path(__file__).resolve().parents[2]
tex=(list((ROOT/'manuscript').glob('*.tex'))+
     list((ROOT/'supplements').glob('*.tex'))+
     list((ROOT/'references').glob('*.tex'))+
     list((ROOT/'monograph').glob('*.tex')))
missing=[]; ambiguous=[]; checked=[]
path_ext=re.compile(r'\.(?:py|json|csv|md|tex|pdf|txt|sha256)$',re.I)

def normalize(raw):
    return (raw.replace(r'\_','_').replace(r'\#','#').replace(r'\%','%').strip())

def resolve(source,v):
    # URLs and symbolic/math snippets are not filesystem references.
    if v.startswith(('http://','https://','doi:')) or '$' in v or '...' in v:
        return None,'skip'
    pathlike=('/' in v or path_ext.search(v))
    if not pathlike:
        return None,'skip'
    candidates=[]
    for q in (ROOT/v, source.parent/v):
        try:q=q.resolve()
        except Exception:continue
        if q.exists() and q not in candidates:candidates.append(q)
    if not candidates and '/' not in v and path_ext.search(v):
        # Bare basename: accept only a unique source-tree match.
        candidates=[q.resolve() for q in ROOT.rglob(v) if q.is_file()]
    if len(candidates)==1:return candidates[0],'ok'
    if len(candidates)>1:return candidates,'ambiguous'
    return None,'missing'

patterns=[]
for cmd in ('path','nolinkurl','texttt'):
    patterns.append((cmd,re.compile(r'\\'+cmd+r'\{([^{}]+)\}')))
for p in tex:
    s=p.read_text(errors='ignore')
    for cmd,pat in patterns:
        for m in pat.finditer(s):
            raw=m.group(1); v=normalize(raw)
            target,status=resolve(p,v)
            if status=='skip':continue
            rec={'source':str(p.relative_to(ROOT)),'command':cmd,'reference':v}
            if status=='ok':rec['resolved']=str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)
            checked.append(rec)
            if status=='missing':missing.append(rec)
            elif status=='ambiguous':
                ambiguous.append({**rec,'matches':[str(q.relative_to(ROOT)) if q.is_relative_to(ROOT) else str(q) for q in target]})
out={'status':'PASS' if not missing and not ambiguous else 'FAIL',
     'checked_references':len(checked),'missing':missing,'ambiguous':ambiguous,'checked':checked}
(ROOT/'certificates/tex_path_integrity_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if out['status']=='PASS' else 1)
