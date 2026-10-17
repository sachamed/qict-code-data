#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'stage_b'/'results'
required_structural={
 'C10_SPLITSTEP_CORE_SECTOR_MAP.json':'PASS_C10_SPLITSTEP_CORE_AFFINE_SECTOR_MAP',
 'RELATIVE_ELECTRIC_DEFECT_REDUCTION.json':'PASS_RELATIVE_ELECTRIC_DEFECT_LOCALIZATION',
 'LOCAL_DEFECT_FRONTIER_WIDTH.json':'PASS_LOCAL_DEFECT_FRONTIER_WIDTH_14',
 'CONDITIONAL_PRODUCT_GROUP_INFLUENCE.json':'PASS_EXACT_CONDITIONAL_PRODUCT_GROUP_INFLUENCE_DECOMPOSITION',
 'C8_L2_INTERNAL_REDUCTION.json':'PASS_EXACT_L2_C8_POSITION_ELIMINATION',
 'PATHWIDTH_EXACT_CERTIFICATE.json':'PASS_EXACT_LINK_SWEEP_PATHWIDTH_14',
 'PHYSICAL_ERROR_BUDGET.json':'PASS_FROZEN_PHYSICAL_ERROR_BUDGET',
 'U1_WIDTH14_FRONTIER_CROSSCHECK.json':'PASS_U1_WIDTH14_FRONTIER_INDEPENDENT_3D_CROSSCHECK',
 'FRONTIER_ENGINE_ARCHITECTURE_SELECTION.json':'PASS_FRONTIER_ENGINE_ARCHITECTURE_SELECTION',
}
missing=[]
for fn,st in required_structural.items():
    p=R/fn
    try: ok=p.exists() and json.load(open(p)).get('status')==st
    except Exception: ok=False
    if not ok: missing.append(fn)
forbidden=[
 'probabilistic C10 thin-string superposition',
 'neutral periodic vacuum return factor used as the charged observable',
 'static common-link control',
 'one-link SU2 route',
 'three-independent-link SU2 route',
 'K-cutoff endpoint',
 'S24-only auxiliary cutoff result',
 'Abelian current compression used as the SU2 holonomy',
 'post-summed U1 times SU2 route amplitudes'
]
if missing:
    out={'status':'BLOCKED_PHYSICAL_PROMOTION','reason':'One or more exact structural reduction certificates are absent or failing.','missing_structural_certificates':missing,'forbidden_substitutes':forbidden,'all_pass':False}
    print(json.dumps(out,indent=2));sys.exit(4)

p=R/'PEPO3D_H_ROUTE_SCALARS.json'
if not p.exists():
    out={
      'status':'BLOCKED_PHYSICAL_PROMOTION',
      'reason':'The source/environment reduction is structurally closed, but PEPO3D_H_ROUTE_SCALARS.json is absent: the neutral-Kato width-14 influence-frontier contraction with common C8 history and local electric defects has not produced coarse/fine h_8,h_9,h_10.',
      'required_objects':['h_10(coarse,fine)','h_9(coarse,fine)','h_8(coarse,fine)','frontier discarded-norm certificate'],
      'forbidden_substitutes':forbidden,'all_pass':False
    }
    print(json.dumps(out,indent=2));sys.exit(4)

d=json.load(open(p));checks={}
eps=1.0e-6
if abs(float(d.get('epsilon_h',eps))-eps)>1e-15:
    print(json.dumps({'status':'FAIL_TASK3_PHYSICAL_PROMOTION','reason':'epsilon_h must equal the frozen absolute target 1e-6','all_pass':False},indent=2)); sys.exit(3)
for rr in (10,9,8):
    k=str(rr)
    try:
      c=d['coarse'][k]; f=d['fine'][k]
      hc=complex(c['real'],c['imag']); hf=complex(f['real'],f['imag'])
      ec=float(c['eta_total']); ef=float(f['eta_total'])
      lhs=abs(hf-hc)+ec+ef
      checks[k]={'lhs':lhs,'rhs':eps,'pass':lhs<eps}
    except Exception as e: checks[k]={'pass':False,'error':str(e)}
frontier=bool(d.get('frontier_convergence_pass',False))
allpass=frontier and all(x.get('pass',False) for x in checks.values())
out={'status':'PASS_TASK3_PHYSICAL_PROMOTION' if allpass else 'FAIL_TASK3_PHYSICAL_PROMOTION','checks':checks,'frontier_convergence_pass':frontier,'all_pass':allpass,
     'definition':'h_d from the repeated-time relative charged-lepton U(1)xSU(2)+CAR network after exact SU3 cancellation, with C10 fixing the affine Gauss sector and charged dependence represented as branch-conditioned local electric defects on the neutral Kato width-14 frontier.'}
print(json.dumps(out,indent=2));sys.exit(0 if allpass else 3)
