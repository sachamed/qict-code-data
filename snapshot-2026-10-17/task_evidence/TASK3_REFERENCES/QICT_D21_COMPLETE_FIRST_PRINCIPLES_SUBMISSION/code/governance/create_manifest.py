#!/usr/bin/env python3
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2]
EXCLUDE={'MANIFEST_SHA256.txt'}
rows=[]
for p in sorted(x for x in ROOT.rglob('*') if x.is_file()):
    rel=p.relative_to(ROOT).as_posix()
    if rel in EXCLUDE: continue
    if any(part in {'__pycache__','.git'} for part in p.relative_to(ROOT).parts): continue
    if p.suffix in {'.aux','.out','.toc','.fls','.fdb_latexmk','.synctex.gz'}: continue
    rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
(ROOT/'MANIFEST_SHA256.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8')
print(f'WROTE MANIFEST_SHA256.txt with {len(rows)} files')
