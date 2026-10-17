#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parents[2]
manifest=ROOT/'MANIFEST_SHA256.txt'
errors=[]; checked=0
if not manifest.exists():
    print(json.dumps({'status':'FAIL','error':'MANIFEST_SHA256.txt missing'},indent=2)); sys.exit(1)
for lineno,line in enumerate(manifest.read_text(encoding='utf-8').splitlines(),1):
    if not line.strip(): continue
    try: expected,rel=line.split('  ',1)
    except ValueError:
        errors.append({'line':lineno,'reason':'malformed'}); continue
    p=ROOT/rel
    if not p.is_file():
        errors.append({'path':rel,'reason':'missing'}); continue
    got=hashlib.sha256(p.read_bytes()).hexdigest(); checked+=1
    if got!=expected:
        errors.append({'path':rel,'reason':'sha256_mismatch','expected':expected,'actual':got})
out={'status':'PASS' if not errors else 'FAIL','checked_files':checked,'errors':errors}
print(json.dumps(out,indent=2)); sys.exit(0 if not errors else 1)
