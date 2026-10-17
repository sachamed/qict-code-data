#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
LOCK=ROOT/'preregistration'/'QICT_BCC_SEALED_PREDICTION_LOCK.json'
obj=json.loads(LOCK.read_text()) if LOCK.exists() else {}
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for ent in obj.get('files',[]):
    p=ROOT/ent['path']
    checks.append({'path':ent['path'],'pass':p.exists() and sha256(p)==ent['sha256'] and p.stat().st_size==ent['bytes']})
canonical=''.join(f"{x['path']}\0{x['sha256']}\n" for x in obj.get('files',[])).encode()
payload=hashlib.sha256(canonical).hexdigest()
ok=bool(checks) and all(x['pass'] for x in checks) and payload==obj.get('payload_sha256')
out={'status':'PASS' if ok else 'FAIL','payload_sha256':payload,'checks':checks,'experimental_data_used':False}
(ROOT/'certificates'/'BCC_SEALED_SPECIFICATION_LOCK.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if ok else 1)
