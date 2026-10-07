#!/usr/bin/env python3
from pathlib import Path
import json, hashlib, math, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PAY=ROOT/'input/G2_NINE_FLAVOR_PAYLOAD.json'; OUT=ROOT/'results/NINE_MASS_RESULT.json'; LOCK=ROOT/'results/NINE_MASS_PRECOMPARISON.sha256'
ACTION='c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7'
FLAVORS=['e','mu','tau','u','c','t','d','s','b']; VOLS=[4,8,12]

def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def fit(y,s,power):
 L=np.array(VOLS,float); y=np.array(y,float); s=np.array(s,float)
 X=np.column_stack([np.ones(3),1/(L**power)]); W=np.diag(1/np.maximum(s,1e-300)**2)
 A=X.T@W@X; cov=np.linalg.pinv(A,rcond=1e-13); b=cov@X.T@W@y
 r=y-X@b; chi=float(r.T@W@r); return float(b[0]),float(math.sqrt(max(cov[0,0],0))),chi
if not PAY.exists(): raise SystemExit('Missing input/G2_NINE_FLAVOR_PAYLOAD.json; copy and complete the template after physical G2 extraction.')
p=json.loads(PAY.read_text())
if p.get('status')!='FROZEN_QICT_KATO_C10_G2_NINE_MASS_PAYLOAD_V1': raise SystemExit('bad payload status')
if p.get('action_sha256')!=ACTION: raise SystemExit('action hash mismatch')
if p.get('external_targets_used') is not False: raise SystemExit('external targets were used: blind mass extraction rejected')
if p.get('volumes')!=VOLS: raise SystemExit('exact volumes [4,8,12] required')
res={}
for f in FLAVORS:
 block=p.get('flavors',{}).get(f,{}).get('points',{})
 vals=[]; sig=[]
 for L in VOLS:
  q=block.get(str(L),{}); v=q.get('value'); e=q.get('sigma')
  if v is None or e is None: raise SystemExit(f'missing physical G2 value/sigma for {f}, L={L}')
  v=float(v); e=float(e)
  if not (np.isfinite(v) and np.isfinite(e) and v>0 and e>0): raise SystemExit(f'invalid value/sigma for {f}, L={L}')
  vals.append(v); sig.append(e)
 q1,s1,c1=fit(vals,sig,1); q2,s2,c2=fit(vals,sig,2)
 central=(q1+q2)/2; spread=abs(q1-q2)/2; err=math.hypot(max(s1,s2),spread)
 if central<=0: raise SystemExit(f'nonpositive infinite-volume estimate for {f}')
 res[f]={'L_values':dict(zip(map(str,VOLS),vals)),'L_sigmas':dict(zip(map(str,VOLS),sig)),
         'model_1_over_L':{'mass_inf':q1,'sigma':s1,'chi2':c1},'model_1_over_L2':{'mass_inf':q2,'sigma':s2,'chi2':c2},
         'mass_clock_units':central,'absolute_error_clock_units':err,'relative_error':err/central}
lock_scope={'status':p['status'],'action_sha256':p['action_sha256'],'external_targets_used':p['external_targets_used'],'volumes':p['volumes'],'mass_scheme':p.get('mass_scheme'),'flavors':p['flavors']}
digest=hashlib.sha256(canonical(lock_scope)).hexdigest(); LOCK.write_text(digest+'  G2_NINE_FLAVOR_PRECOMPARISON\n')
scale=p.get('clock_energy_GeV'); source=p.get('clock_energy_source')
if scale is not None:
 scale=float(scale)
 if not (np.isfinite(scale) and scale>0 and source and source!='FIRST_PRINCIPLES_ONLY_OR_NULL'):
  raise SystemExit('absolute scale must be positive and have an explicit first-principles source label')
 for f in FLAVORS:
  res[f]['mass_GeV']=res[f]['mass_clock_units']*scale
  res[f]['absolute_error_GeV']=res[f]['absolute_error_clock_units']*scale
out={'status':'PASS_NINE_MASS_EXTRACTION_PRECOMPARISON_LOCKED','blind':True,'action_sha256':ACTION,'precomparison_sha256':digest,
     'volumes':VOLS,'mass_scheme':p.get('mass_scheme'),'clock_energy_GeV':scale,'clock_energy_source':source,'masses':res,
     'ratios':{'mu_over_e':res['mu']['mass_clock_units']/res['e']['mass_clock_units'],'tau_over_e':res['tau']['mass_clock_units']/res['e']['mass_clock_units'],
               'c_over_u':res['c']['mass_clock_units']/res['u']['mass_clock_units'],'t_over_c':res['t']['mass_clock_units']/res['c']['mass_clock_units'],
               's_over_d':res['s']['mass_clock_units']/res['d']['mass_clock_units'],'b_over_s':res['b']['mass_clock_units']/res['s']['mass_clock_units']},
     'comparison_to_1_3_5_permitted_only_after_this_file_is_written':True}
OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
