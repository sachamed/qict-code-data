#!/usr/bin/env python3
"""Compile all three documents in an isolated directory, including crossrefs."""
from pathlib import Path
import argparse,json,re,shutil,subprocess,tempfile,sys,os

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--keep-work',type=Path);args=ap.parse_args()
    main=PACKAGE/'01_Main_Manuscript';supp=PACKAGE/'02_Supplementary_Information';pred=PACKAGE/'04_Theoretical_Predictions'
    subprocess.run([sys.executable,str(ROOT/'normalize_figure_fonts.py')],check=True)
    def publish(source,dest):
        staged=dest.with_name(dest.name+'.staged')
        shutil.copy2(source,staged)
        os.replace(staged,dest)
    report={'commands':[],'warnings':{},'scope':'LaTeX compilation and reference/layout diagnostics; no scientific-proof certification.'}
    with tempfile.TemporaryDirectory(prefix='qict-publication-') as td:
        work=Path(td)
        for src in [main,supp]:
            shutil.copytree(src,work,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.aux','*.log','*.out','*.toc','*.fls','*.fdb_latexmk','manuscript.pdf','supplement.pdf','README.md','build.py'))
        def run(cmd):
            p=subprocess.run(cmd,cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            report['commands'].append(cmd)
            if p.returncode:
                print(p.stdout[-12000:]);raise RuntimeError('Build command failed: '+' '.join(cmd))
        def latex(name):
            source='manuscript_final.tex' if name=='manuscript' else 'supplement_final.tex'
            run(['pdflatex','-jobname='+name,'-recorder','-interaction=nonstopmode','-halt-on-error',source])
        for name in ['manuscript','supplement']:latex(name)
        for name in ['manuscript','supplement']:run(['bibtex',name])
        for i in range(4):
            for name in ['manuscript','supplement']:latex(name)
        for name,dest in [('manuscript',main),('supplement',supp)]:
            log=(work/(name+'.log')).read_text(errors='replace')
            issues=[x for x in log.splitlines() if re.search(r'undefined|multiply defined|Overfull|^!|Fatal error',x,re.I)]
            report['warnings'][name]=issues
            publish(work/(name+'.pdf'),dest/(name+'.pdf'))
            shutil.copy2(work/(name+'.bbl'),dest/(name+'.bbl'))
            other='supplement' if name=='manuscript' else 'manuscript'
            shutil.copy2(work/(other+'.aux'),dest/(other+'.aux'))
            logdir=ROOT/'audit/latex_logs';logdir.mkdir(parents=True,exist_ok=True)
            shutil.copy2(work/(name+'.log'),logdir/(name+'.log'))
            shutil.copy2(work/(name+'.aux'),logdir/(name+'.aux'))
        if (pred/'predictions.tex').exists():
            shutil.copy2(pred/'predictions.tex',work/'predictions.tex')
            for other in ['manuscript','supplement']:shutil.copy2(work/(other+'.aux'),pred/(other+'.aux'))
            for i in range(3):run(['pdflatex','-interaction=nonstopmode','-halt-on-error','predictions.tex'])
            publish(work/'predictions.pdf',pred/'predictions.pdf')
            log=(work/'predictions.log').read_text(errors='replace')
            report['warnings']['predictions']=[x for x in log.splitlines() if re.search(r'undefined|multiply defined|Overfull|^!|Fatal error',x,re.I)]
            shutil.copy2(work/'predictions.log',ROOT/'audit/latex_logs/predictions.log')
        if args.keep_work:shutil.copytree(work,args.keep_work,dirs_exist_ok=True)
    (ROOT/'audit/BUILD_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['warnings'],indent=2))
    if any(report['warnings'].values()):raise SystemExit(1)

if __name__=='__main__':main()
