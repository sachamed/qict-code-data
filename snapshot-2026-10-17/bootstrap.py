#!/usr/bin/env python3
"""Restore exact aliases and create the documented legacy source view."""
from pathlib import Path
import hashlib,json,os,shutil

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent

def prepare():
    restored=0
    manifest=ROOT/'EXACT_FILE_ALIASES.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text()):
            src=ROOT/row['canonical'];dst=ROOT/row['alias']
            assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256']
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():
                shutil.copy2(src,dst)
                restored+=1
    # These are source-layout compatibility links, created on demand rather
    # than duplicate scientific sources distributed in a second location.
    paper=ROOT/'paper';paper.mkdir(exist_ok=True)
    main=PACKAGE/'01_Main_Manuscript';supp=PACKAGE/'02_Supplementary_Information'
    targets={
        'manuscript_final.tex':main/'manuscript_final.tex',
        'supplement_final.tex':supp/'supplement_final.tex',
        'preamble.tex':main/'preamble.tex',
        'references.bib':main/'references.bib',
        'sections/main':main/'sections/main',
        'sections/supplement':supp/'sections/supplement',
    }
    for name,src in targets.items():
        dst=paper/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            try:dst.symlink_to(os.path.relpath(src,dst.parent),target_is_directory=src.is_dir())
            except OSError:
                if src.is_dir():shutil.copytree(src,dst)
                else:shutil.copy2(src,dst)
    fig=paper/'figures';fig.mkdir(exist_ok=True)
    for src in [*main.glob('figures/*'),*supp.glob('figures/*')]:
        dst=fig/src.name
        if not dst.exists():
            try:dst.symlink_to(os.path.relpath(src,dst.parent))
            except OSError:shutil.copy2(src,dst)
    return {'exact_aliases_restored':restored,'legacy_source_view':'paper/'}

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
