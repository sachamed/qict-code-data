#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'stage_b'/'results'/'PEPO3D_H_ROUTE_SCALARS.json'
required=[10,9,8]
if not p.exists():
    out={
      'status':'BLOCKED_PHYSICAL_PROMOTION',
      'reason':'PEPO3D_H_ROUTE_SCALARS.json is absent: the repeated-time SU(2) 24-link recoupling with the common matter path has not been executed.',
      'required_objects':['h_10','h_9','h_8'],
      'forbidden_substitutes':['static common-link control','one-link SU2 route','three-independent-link SU2 route','K-cutoff endpoint','Abelian current compression'],
      'all_pass':False
    }
    print(json.dumps(out,indent=2)); sys.exit(4)
d=json.load(open(p))
checks={}
for r in required:
    k=str(r)
    try:
      c=d['coarse'][k]; f=d['fine'][k]
      hc=complex(c['real'],c['imag']); hf=complex(f['real'],f['imag'])
      ec=float(c['eta_total']); ef=float(f['eta_total']); eps=float(d['epsilon_h'])
      checks[k]={'lhs':abs(hf-hc)+ec+ef,'rhs':eps,'pass':abs(hf-hc)+ec+ef<eps}
    except Exception as e:
      checks[k]={'pass':False,'error':str(e)}
allpass=all(x.get('pass',False) for x in checks.values())
out={'status':'PASS_TASK3_PHYSICAL_PROMOTION' if allpass else 'FAIL_TASK3_PHYSICAL_PROMOTION',
     'checks':checks,'all_pass':allpass,
     'definition':'h_d=<Omega0,s,h0|A^d|Omega0,s,h0> from the same repeated-time 24-link U(1)xSU(2) PEPO with electric kernel and C10-dressed source/sink.'}
print(json.dumps(out,indent=2)); sys.exit(0 if allpass else 3)
