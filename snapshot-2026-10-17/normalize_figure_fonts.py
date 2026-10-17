#!/usr/bin/env python3
"""Embed any retained base fonts while preserving figure vectors and data."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile,os

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent

def missing_fonts(path):
    lines=subprocess.run(['pdffonts',str(path)],check=True,capture_output=True,text=True).stdout.splitlines()[2:]
    return [line for line in lines if len(line.split())>6 and line.split()[-5]!='yes']

def main():
    report_path=ROOT/'audit/FONT_NORMALIZATION.json'
    records=json.loads(report_path.read_text()) if report_path.exists() else []
    changed=0
    for folder in ['01_Main_Manuscript','02_Supplementary_Information']:
        for path in sorted((PACKAGE/folder/'figures').glob('*.pdf')):
            fonts=missing_fonts(path)
            if not fonts:continue
            if not shutil.which('gs'):raise RuntimeError('Ghostscript is required to embed a retained figure base font.')
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            with tempfile.TemporaryDirectory(prefix='qict-fonts-') as td:
                out=Path(td)/path.name
                args=['gs','-q','-dNOPAUSE','-dBATCH','-sDEVICE=pdfwrite','-dCompatibilityLevel=1.7','-dPDFSETTINGS=/prepress','-dEmbedAllFonts=true','-dSubsetFonts=true','-sOutputFile='+str(out),str(path)]
                subprocess.run(args,check=True)
                assert not missing_fonts(out)
                staged=path.with_name(path.name+".font-staged")
                shutil.copy2(out,staged)
                os.replace(staged,path)
            records.append({'figure':str(path.relative_to(PACKAGE)),'input_sha256':before,'embedded_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'original_unembedded_fonts':fonts,'method':'Ghostscript pdfwrite/prepress with EmbedAllFonts and SubsetFonts; no numerical-data transformation.'})
            changed+=1
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'figure_fonts_embedded_this_run':changed}))

if __name__=='__main__':main()
