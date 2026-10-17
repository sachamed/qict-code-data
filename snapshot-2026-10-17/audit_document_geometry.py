#!/usr/bin/env python3
"""Check every delivered PDF page and font inventory; no proof-validity claim."""
from pathlib import Path
import json,subprocess
import fitz
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent
rows=[]
for folder,name in [('01_Main_Manuscript','manuscript'),('02_Supplementary_Information','supplement'),('04_Theoretical_Predictions','predictions')]:
    path=PACKAGE/folder/(name+'.pdf')
    if b'%%EOF' not in path.read_bytes()[-2048:]:raise RuntimeError('Incomplete PDF export: '+str(path))
    doc=fitz.open(path);outside=[];unresolved=[];blank=[]
    for i,page in enumerate(doc):
        txt=page.get_text()
        if '??' in txt:unresolved.append(i+1)
        if len(txt.strip())<20:blank.append(i+1)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:
                        outside.append([i+1,span['text'][:80]])
    lines=subprocess.run(['pdffonts',str(path)],capture_output=True,text=True,check=True).stdout.splitlines()[2:]
    badfonts=[line for line in lines if len(line.split())>6 and line.split()[-5]!='yes']
    rows.append({'document':str(path.relative_to(PACKAGE)),'pages':len(doc),'unresolved_reference_candidates':unresolved,
                 'nearly_blank_pages':blank,'text_outside_physical_page':outside,'unembedded_fonts':badfonts})
out=ROOT/'audit/PDF_TEXT_GEOMETRY.json'
out.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
if any(row[k] for row in rows for k in ['unresolved_reference_candidates','nearly_blank_pages','text_outside_physical_page','unembedded_fonts']):
    raise SystemExit(1)
