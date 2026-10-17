#!/usr/bin/env python3
"""Index every current LaTeX source and check labels, citations and syntax.

This is a structural audit, not an independent mathematical proof check.
Run after editing sources and before build_catalogue.py and build_documents.py.
"""
from pathlib import Path
import ast, collections, csv, hashlib, json, re
ROOT=Path(__file__).resolve().parent
P=ROOT.parent
A=ROOT/'audit'
FORMAL=['theorem','proposition','lemma','corollary','criterion','assumption','definition','test']
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def save_csv(path,rows,fields):
 with path.open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows)
def clean(text):return re.sub(r'(?<!\\)%[^\n]*','',text)
def main():
 A.mkdir(exist_ok=True);claims=[];tables=[];sources=[];headings=[];labels={};refs={};citations={};active={}
 for folder,doc,root in [('01_Main_Manuscript','main','manuscript_final.tex'),('02_Supplementary_Information','supp','supplement_final.tex')]:
  d=P/folder;lab=collections.defaultdict(list);ref=[];cit=set();seen=set()
  def traverse(f):
   if f in seen:return
   seen.add(f)
   for name in re.findall(r'\\input\{([^}]+)\}',clean(f.read_text())):
    child=d/(name if name.endswith('.tex') else name+'.tex')
    if child.exists():traverse(child)
  traverse(d/root)
  active[doc]={'source_modules':sum('sections' in f.parts for f in seen),'input_files':len(seen)}
  for f in sorted(d.rglob('*.tex')):
   txt=f.read_text();rel=str(f.relative_to(P));src=clean(txt)
   sources.append({'path':rel,'sha256':sha(f),'bytes':f.stat().st_size,'active':f in seen})
   for m in re.finditer(r'\\begin\{('+ '|'.join(FORMAL)+r')\}(?:\[([^\]]*)\])?(.*?)\\end\{\1\}',src,re.S):
    labs=re.findall(r'\\label\{([^}]+)\}',m[3]);claims.append({'id':f'C{len(claims)+1:04d}','path':rel,'line':src.count('\n',0,m.start())+1,'type':m[1],'title':m[2] or '', 'labels':';'.join(labs),'statement':m[3].strip()})
   for m in re.finditer(r'\\begin\{(?:table\*?|longtable)\}(.*?)\\end\{(?:table\*?|longtable)\}',src,re.S):
    tables.append({'document':folder,'source':rel,'line':src.count('\n',0,m.start())+1,'labels':';'.join(re.findall(r'\\label\{([^}]+)\}',m[1])),'sha256':hashlib.sha256(m[0].encode()).hexdigest(),'reproduction':'LaTeX table source; numerical data and certificates cited locally. A typesetting check is distinct from a numerical recomputation.'})
   for m in re.finditer(r'\\(subsubsection|subsection|section|paragraph)\*?\{([^\n]*)',src):
    headings.append({'path':rel,'line':src.count('\n',0,m.start())+1,'level':m[1],'heading':m[2]})
   if f not in seen:continue
   for m in re.finditer(r'\\label\{([^}]+)\}',src):lab[m[1]].append(rel)
   for m in re.finditer(r'\\(?:eqref|ref|pageref|autoref|[Cc]ref)\{([^}]+)\}',src):
    for key in m[1].split(','):ref.append((rel,key.strip()))
   for m in re.finditer(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',src):cit.update(x.strip() for x in m[1].split(','))
  labels[doc]=lab;refs[doc]=ref;citations[doc]=cit
 bibs=[P/'01_Main_Manuscript/references.bib',P/'02_Supplementary_Information/references.bib']
 bibkeys=re.findall(r'@\w+\s*\{([^,]+),',bibs[0].read_text())
 missing=[]
 for doc,rr in refs.items():
  for f,key in rr:
   target=doc
   if '#' in key or key in labels[doc]:continue
   if key.startswith('main:'):target='main';key=key[5:]
   elif key.startswith('supp:'):target='supp';key=key[5:]
   if key not in labels[target]:missing.append({'source':f,'label':key,'target':target})
 parse_errors=[];pycount=0;jsoncount=0
 for f in sorted(ROOT.rglob('*')):
  if not f.is_file() or any(x in f.parts for x in ['paper','checkpoints','__pycache__']):continue
  try:
   if f.suffix=='.py':ast.parse(f.read_text());pycount+=1
   elif f.suffix=='.json':json.loads(f.read_text());jsoncount+=1
  except Exception as e:parse_errors.append({'path':str(f.relative_to(ROOT)),'error':str(e)})
 duplicates={doc:{key:value for key,value in ll.items() if len(value)>1} for doc,ll in labels.items()}
 missing_cites={doc:sorted(cc-set(bibkeys)) for doc,cc in citations.items()}
 report={'scope':'Structural source audit; no independent proof validation claim.','active':active,'all_tex_source_files':len(sources),'formal_statement_index_entries':len(claims),'table_environments':len(tables),'bibliography_entries':len(bibkeys),'bibliographies_identical':sha(bibs[0])==sha(bibs[1]),'duplicate_bibliography_keys':[k for k,v in collections.Counter(bibkeys).items() if v>1],'duplicate_labels':duplicates,'missing_references':missing,'missing_citations':missing_cites,'python_files_parsed':pycount,'json_files_parsed':jsoncount,'parse_errors':parse_errors}
 report['status']='PASS' if not (missing or parse_errors or any(duplicates.values()) or any(missing_cites.values()) or report['duplicate_bibliography_keys'] or not report['bibliographies_identical']) else 'FAIL'
 save_csv(A/'source_claim_index.csv',claims,['id','path','line','type','title','labels','statement'])
 save_csv(P/'04_Theoretical_Predictions/formal_statement_index.csv',claims,['id','path','line','type','title','labels','statement'])
 save_csv(A/'source_headings.csv',headings,['path','line','level','heading'])
 save_csv(A/'CURRENT_SOURCE_INVENTORY.csv',sources,['path','sha256','bytes','active'])
 save_csv(ROOT/'TABLE_PROVENANCE.csv',tables,['document','source','line','labels','sha256','reproduction'])
 (A/'STRUCTURAL_RELEASE_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
 if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
