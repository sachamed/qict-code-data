#!/usr/bin/env python3
"""Compile and assemble the active QICT-D21 manuscript and all supplements."""
from __future__ import annotations
from pathlib import Path
import re, subprocess, sys, tempfile
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
MAIN_TEX = ROOT / "manuscript" / "QICT_D21_MAIN_MANUSCRIPT_FINAL.tex"
SUPP_DIR = ROOT / "supplements"
OUT_COMPLETE = ROOT / "QICT_D21_COMPLETE_SUBMISSION_FINAL.pdf"
OUT_SUPP = SUPP_DIR / "QICT_D21_SUPPLEMENTAL_MATERIAL_FINAL.pdf"

def code_value(code: str) -> int:
    v=0
    for ch in code:
        v=v*26+(ord(ch)-64)
    return v

def supplement_code(p: Path) -> str:
    m=re.match(r"QICT_Supplement_([A-Z]+)_",p.name)
    if not m: raise ValueError(p.name)
    return m.group(1)

def compile_tex(tex: Path, passes: int=2) -> Path:
    dst=tex.with_suffix(".pdf")
    if dst.exists() and dst.stat().st_mtime >= tex.stat().st_mtime:
        return dst
    with tempfile.TemporaryDirectory(prefix="qict_tex_") as td:
        td=Path(td)
        for _ in range(passes):
            r=subprocess.run([
                "pdflatex","-interaction=nonstopmode","-halt-on-error",
                f"-output-directory={td}",str(tex)
            ],cwd=tex.parent,text=True,capture_output=True,encoding="utf-8",errors="replace")
            if r.returncode:
                sys.stderr.write(r.stdout[-6000:]+"\n"+r.stderr[-2000:]+"\n")
                raise SystemExit(f"LaTeX failure: {tex}")
        src=td/(tex.stem+".pdf")
        dst.write_bytes(src.read_bytes())
        return dst

def merge(paths: list[Path], out: Path, title: str) -> int:
    w=PdfWriter(); page=0
    w.add_metadata({"/Title":title,"/Author":"Mohamed Sacha","/Subject":"Quantum Information Copy Time (QICT)"})
    for p in paths:
        r=PdfReader(str(p))
        try: w.add_outline_item(p.stem.replace("QICT_Supplement_","Supplement ").replace("_"," "),page)
        except Exception: pass
        for pg in r.pages: w.add_page(pg)
        page += len(r.pages)
    with out.open("wb") as f: w.write(f)
    return page

def main() -> int:
    main_pdf=compile_tex(MAIN_TEX,2)
    supp_tex=sorted(SUPP_DIR.glob("QICT_Supplement_*.tex"),key=lambda p:code_value(supplement_code(p)))
    supp_pdf=[compile_tex(p,2) for p in supp_tex]
    supp_pages=merge(supp_pdf,OUT_SUPP,"QICT-D21 Supplemental Material")
    complete_pages=merge([main_pdf,*supp_pdf],OUT_COMPLETE,"QICT-D21 Complete Submission")
    print(f"supplements={len(supp_pdf)} supplement_pages={supp_pages} complete_pages={complete_pages}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
