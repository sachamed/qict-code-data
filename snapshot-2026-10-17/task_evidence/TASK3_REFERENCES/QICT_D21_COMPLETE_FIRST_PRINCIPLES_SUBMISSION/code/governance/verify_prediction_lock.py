#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
LOCK=ROOT/'preregistration'/'QICT_PREDICTION_LOCK.json'
if not LOCK.exists():
    print(json.dumps({'status':'FAIL','reason':'prediction lock absent'},indent=2)); sys.exit(1)
obj=json.loads(LOCK.read_text())
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for ent in obj['files']:
    p=ROOT/ent['path']; ok=p.exists() and sha256(p)==ent['sha256']
    checks.append({'path':ent['path'],'pass':ok})
canonical=''.join(f"{x['path']}\0{x['sha256']}\n" for x in obj['files']).encode()
payload=hashlib.sha256(canonical).hexdigest()
ok=all(x['pass'] for x in checks) and payload==obj['payload_sha256']
out={'status':'PASS' if ok else 'FAIL','claim_status':'COMPUTED','payload_sha256':payload,'external_timestamp_status':'INDEPENDENT_IMMUTABLE_TIMESTAMP_REQUIRED_FOR_PREREGISTERED_COMPARISON','checks':checks}
(ROOT/'certificates'/'PREDICTION_LOCK_INTEGRITY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2)); sys.exit(0 if ok else 1)
