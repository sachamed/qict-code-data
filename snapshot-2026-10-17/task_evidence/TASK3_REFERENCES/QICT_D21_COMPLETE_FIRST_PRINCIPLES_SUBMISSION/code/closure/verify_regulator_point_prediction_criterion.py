#!/usr/bin/env python3
"""QICT point-observable convergence criterion.
Reads an optional CSV with columns observable, scheme, regulator_scale, value, error.
When the interacting regulator payload is absent, the observable remains DEFINED_OBSERVABLE.
"""
from pathlib import Path
import csv,json,sys,math
ROOT=Path(__file__).resolve().parents[2]
payload=ROOT/'data/g2_physical_regulator_scan.csv'
out={'criterion':'G2_POINT_OBSERVABLE','status':'DEFINED_OBSERVABLE','reason':'Numerical evaluation requires a full 3D compact interacting regulator scan.'}
if payload.exists():
    rows=list(csv.DictReader(payload.open()))
    by={}
    for r in rows:
        by.setdefault((r['observable'],r['scheme']),[]).append((float(r['regulator_scale']),float(r['value']),float(r['error'])))
    failures=[]; limits={}
    for key,pts in by.items():
        pts=sorted(pts)
        if len(pts)<4: failures.append(f'{key}: fewer than 4 regulator points'); continue
        tail=pts[-3:]
        vals=[p[1] for p in tail]; errs=[p[2] for p in tail]
        spread=max(vals)-min(vals); tol=max(errs)*3
        if spread>tol: failures.append(f'{key}: tail spread {spread} > 3sigma {tol}')
        limits[key]=sum(vals)/len(vals)
    # scheme stability per observable
    obs={}
    for (o,s),v in limits.items(): obs.setdefault(o,[]).append(v)
    for o,vs in obs.items():
        if len(vs)<2: failures.append(f'{o}: fewer than 2 regulator schemes')
        elif max(vs)-min(vs)>1e-3*max(1.0,abs(sum(vs)/len(vs))): failures.append(f'{o}: scheme spread too large')
    out={'criterion':'G2_POINT_OBSERVABLE','status':'PASS' if not failures else 'FAIL','failures':failures,'limits':{str(k):v for k,v in limits.items()}}
(ROOT/'validation/G2_PHYSICAL_PROMOTION_GATE.json').write_text(json.dumps(out,indent=2,sort_keys=True))
print(json.dumps(out,indent=2))
sys.exit(0 if out['status'] in ('PASS','DEFINED_OBSERVABLE') else 1)
