#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PAPER=ROOT/'paper'
COVER=ROOT/'cover_letter'
VAL=ROOT/'validation'
VAL.mkdir(exist_ok=True)

def run(cmd,cwd=None,log=None):
    p=subprocess.run(cmd,cwd=cwd or ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if log: (VAL/log).write_text(p.stdout)
    if p.returncode:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(map(str,cmd))}\n{p.stdout[-4000:]}")
    return p.stdout

def bibtex_exe():
    for name in ('bibtex.original','bibtex','bibtex8'):
        candidate=shutil.which(name)
        if candidate and Path(candidate).exists():
            return candidate
    return None

def compile_tex(stem,passes_after_bib=2):
    tex=f'{stem}.tex'
    run(['pdflatex','-interaction=nonstopmode','-halt-on-error',tex],cwd=PAPER,log=f'{stem}_pdflatex_1.txt')
    bib=bibtex_exe()
    if not bib: raise SystemExit('BibTeX executable not found')
    run([bib,stem],cwd=PAPER,log=f'{stem}_bibtex.txt')
    for i in range(2,2+passes_after_bib):
        run(['pdflatex','-interaction=nonstopmode','-halt-on-error',tex],cwd=PAPER,log=f'{stem}_pdflatex_{i}.txt')


def compile_cover():
    run(['pdflatex','-interaction=nonstopmode','-halt-on-error','QICT_COVER_LETTER.tex'],cwd=COVER,log='cover_letter_pdflatex.txt')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def build_manifest():
    manifest=ROOT/'MANIFEST_SHA256.txt'
    rows=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==manifest: continue
        if p.suffix in {'.aux','.blg','.fdb_latexmk','.fls','.log','.out','.toc'}: continue
        if 'validation/renders_' in str(p.relative_to(ROOT)): continue
        rows.append(f'{sha256(p)}  {p.relative_to(ROOT)}')
    manifest.write_text('\n'.join(rows)+'\n')

ap=argparse.ArgumentParser(description='Regenerate all numerical results and, optionally, the publication PDFs.')
ap.add_argument('--compile-pdfs',action='store_true',help='also compile manuscript and Supplement using the local TeX toolchain')
args=ap.parse_args()

run([sys.executable,str(ROOT/'code/run_all.py')],log='reproduce_numerics.txt')
run([sys.executable,str(ROOT/'code/audit_latex_submission.py')],log='reproduce_source_audit.txt')
run([sys.executable,str(ROOT/'code/audit_publication_claims.py')],log='reproduce_publication_claim_audit.txt')
if args.compile_pdfs:
    if not shutil.which('pdflatex') or not bibtex_exe():
        raise SystemExit('TeX toolchain unavailable')
    # Bootstrap main labels, compile the Supplement with those labels, then settle the main cross-document references.
    compile_tex('manuscript_final',passes_after_bib=2)
    compile_tex('supplement_final',passes_after_bib=2)
    run(['pdflatex','-interaction=nonstopmode','-halt-on-error','manuscript_final.tex'],cwd=PAPER,log='manuscript_final_pdflatex_final1.txt')
    run(['pdflatex','-interaction=nonstopmode','-halt-on-error','manuscript_final.tex'],cwd=PAPER,log='manuscript_final_pdflatex_final2.txt')
    compile_cover()
    shutil.copy2(PAPER/'manuscript_final.pdf', ROOT/'QICT_MAIN_MANUSCRIPT.pdf')
    shutil.copy2(PAPER/'supplement_final.pdf', ROOT/'QICT_SUPPLEMENTARY_INFORMATION.pdf')
    shutil.copy2(COVER/'QICT_COVER_LETTER.pdf', ROOT/'QICT_COVER_LETTER.pdf')
build_manifest()
print('All numerical results regenerated successfully.' + (' PDFs compiled.' if args.compile_pdfs else ''))
