#!/usr/bin/env python3
from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parent
ok=True;n=0
for line in (root/'MANIFEST_SHA256.txt').read_text().splitlines():
    if not line.strip(): continue
    want,rel=line.split('  ',1); p=root/rel; n+=1
    if not p.exists(): print('MISSING',rel); ok=False; continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    if h.hexdigest()!=want: print('MISMATCH',rel); ok=False
print({'status':'PASS' if ok else 'FAIL','files_checked':n})
sys.exit(0 if ok else 2)
