#!/usr/bin/env python3
"""Build the article and global supplement with their shared references."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,tempfile

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--revtex',action='store_true')
 args=parser.parse_args()
 if args.revtex and not subprocess.check_output(['kpsewhich','revtex4-2.cls'],text=True).strip():
  raise SystemExit('Install the genuine REVTeX 4.2 class to build that layout.')
 main_input=r'\def\QICTRevtex{1}\input{manuscript_final.tex}' if args.revtex else 'manuscript_final.tex'
 commands=[
  ['pdflatex','-interaction=nonstopmode','-halt-on-error',main_input],
  ['pdflatex','-interaction=nonstopmode','-halt-on-error','supplement_final.tex'],
  [shutil.which('bibtex') or shutil.which('bibtex8') or 'bibtex','manuscript_final'],
  [shutil.which('bibtex') or shutil.which('bibtex8') or 'bibtex','supplement_final'],
 ]
 for _ in range(2):
  commands.extend([
   ['pdflatex','-interaction=nonstopmode','-halt-on-error',main_input],
   ['pdflatex','-interaction=nonstopmode','-halt-on-error','supplement_final.tex']])
 with tempfile.TemporaryDirectory(prefix='qict-publication-build-') as tmp:
  build=Path(tmp)/'paper'
  shutil.copytree(PAPER,build,ignore=shutil.ignore_patterns('*.aux','*.log','*.out','*.toc','*.lot','*.blg'))
  logs=[]
  for index,cmd in enumerate(commands):
   r=subprocess.run(cmd,cwd=build,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   if r.returncode:raise SystemExit(r.stdout[-6500:])
   logs.append(r.stdout);print('completed',index+1,cmd[-1],flush=True)
  final=logs[-2:]
  undefined=[l for s in final for l in s.splitlines() if 'undefined references' in l or 'undefined on input' in l]
  overfull=[l for s in final for l in s.splitlines() if 'Overfull \\hbox' in l]
  if undefined or overfull:raise SystemExit(json.dumps({'undefined':undefined,'overfull':overfull},indent=2))
  outputs={}
  for name in ['manuscript_final.pdf','supplement_final.pdf']:
   p=build/name;info=subprocess.check_output(['pdfinfo',str(p)],text=True)
   pages=int(next(l.split(':',1)[1] for l in info.splitlines() if l.startswith('Pages:')))
   outputs['paper/'+name]={'pages':pages,'bytes':p.stat().st_size,'sha256':sha(p)}
  inputs={str(p.relative_to(PAPER)):sha(p) for p in sorted(PAPER.rglob('*'))
   if p.is_file() and (p.suffix in {'.tex','.bib','.sty','.bst'} or 'figures' in p.relative_to(PAPER).parts)}
  for name in ['manuscript_final.pdf','supplement_final.pdf','manuscript_final.bbl','supplement_final.bbl']:
   target=PAPER/name;temporary=target.with_suffix(target.suffix+'.complete')
   shutil.copyfile(build/name,temporary);os.replace(temporary,target)
  certificate={'status':'PASS_PUBLICATION_BUILD','main_class':'revtex4-2' if args.revtex else 'article',
   'supplement_class':'article','revtex_layout_executed':args.revtex,
   'undefined_references':0,'overfull_hboxes':0,'outputs':outputs,'commands':commands}
  certificate['publication_inputs_sha256']=inputs
  out=ROOT/'closure/certificates/PDF_BUILD_CERTIFICATE.json';out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(json.dumps(certificate,indent=2)+'\n')
  print(json.dumps(certificate,indent=2))

if __name__=='__main__':main()
