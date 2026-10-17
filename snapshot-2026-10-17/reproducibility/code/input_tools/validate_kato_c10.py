#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, zipfile, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'input'; OUT=ROOT/'results/PREFLIGHT_STATUS.json'
ACTION_EXPECT='c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7'
TASK3_EXPECT='5e88cc98b7412862b790a387125836366b7cfe49fb72855759614623a18f2b79'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

out={'status':'FAIL','action':{},'kato':{},'task3':{},'mass_gate':'BLOCKED'}
# action freeze
ap=IN/'QICT_LATTICE_ACTION_FREEZE_V3.canonical.json'; ah=sha(ap)
out['action']={'sha256':ah,'expected':ACTION_EXPECT,'pass':ah==ACTION_EXPECT}
if ah!=ACTION_EXPECT: raise SystemExit('action freeze hash mismatch')
# Kato checkpoint
kp=IN/'KATO_STRICT_K0_K12_48_CP.npz'; kh=sha(kp)
z=np.load(kp,allow_pickle=False); meta=json.loads(str(z['meta'].item())); v=z['v']
kn=float(np.linalg.norm(v)); kato_pass=(v.shape==(122677,) and meta.get('dim')==122677 and meta.get('steps')==48 and float(meta.get('H_gap',0))>0 and np.isfinite(kn))
out['kato']={'sha256':kh,'vector_shape':list(v.shape),'vector_norm':kn,'next_idx':int(z['next_idx']),'meta':meta,'pass':bool(kato_pass)}
if not kato_pass: raise SystemExit('Kato checkpoint validation failed')
# Task3/C10 full-pass archive
zp=IN/'TASK3_PHYSICAL_FULL_PASS_CONSOLIDATED_2026-09-28.zip'; zh=sha(zp)
if zh!=TASK3_EXPECT: raise SystemExit('Task3 full-pass zip hash mismatch')
with zipfile.ZipFile(zp) as q:
 bad=q.testzip()
 prefix='TASK3_PHYSICAL_FULL_PASS_CONSOLIDATED_2026-09-28/'
 man=json.loads(q.read(prefix+'certificates/FULL_PASS_MANIFEST.json'))
 cert=json.loads(q.read(prefix+'certificates/TASK3_FINAL_THREE_INEQUALITIES_CERTIFICATE_2026-09-28.json'))
 status=json.loads(q.read(prefix+'provenance/STATUS.json'))
 tpass=(bad is None and man.get('status')=='TASK3_PHYSICAL_FULL_PASS' and cert.get('status','').startswith('PASS_3_OF_3') and status.get('status')=='PASS_TASK3_PHYSICAL_FULL_PASS_CONSOLIDATED')
 out['task3']={'sha256':zh,'zip_test':bad is None,'manifest_status':man.get('status'),'coarse_fine':man.get('coarse_fine_pass'),'max_certified_total_bound':man.get('maximum_certified_total_bound'),'provenance_status':status.get('status'),'pass':bool(tpass)}
 if not tpass: raise SystemExit('Task3/C10 full-pass validation failed')
# payload gate
payload=IN/'G2_NINE_FLAVOR_PAYLOAD.json'
if payload.exists():
 out['mass_gate']='PAYLOAD_PRESENT_RUN_EXTRACTOR'
else:
 out['mass_gate']='AWAITING_G2_NINE_FLAVOR_PAYLOAD_L4_L8_L12'
out['status']='PASS_KATO_C10_PREFLIGHT'
OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
