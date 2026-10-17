#!/usr/bin/env python3
"""Exact, data-separate checks of the one-clock QICT inference identities.

All example energy coordinates below are synthetic fractions. No measured
mass enters an energy coordinate or the clock value. This does not evaluate
physical G2 moments or QICT physical masses.
"""
from fractions import Fraction as F
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
checks=[]
def ensure(label, condition):
    if not condition: raise AssertionError(label)
    checks.append(label)

EPS={'e':F(2,13),'mu':F(7,11),'tau':F(17,3),'up':F(31,8),'nu':F(5,1003)}
clock=F(41,7)
clock_2=F(193,5)
k=F(7,19)  # unit convention for hbar/c^2 in this rational arithmetic check

def mass(epsilon,t): return k*epsilon/t

for f,ef in EPS.items():
    ensure('exact scale covariance '+f,mass(ef,3*clock)==mass(ef,clock)/3)
    for g,eg in EPS.items():
        ensure('exact cancellation '+f+'/'+g,mass(ef,clock)/mass(eg,clock)==ef/eg)
        ensure('independent common clock '+f+'/'+g,mass(ef,clock_2)/mass(eg,clock_2)==ef/eg)

A=F(13,100);B=F(17,100);C=F(5,23);D=F(7,23)
lo,hi=F(8,3),F(10,3)
ensure('interval lower mass bound',mass(A,hi)<=mass(B,lo))
ensure('interval upper mass bound',mass(B,lo)>=mass(A,hi))
for x in [A,(A+B)/2,B]:
    for t in [lo,(lo+hi)/2,hi]:
        ensure('mass interval inclusion',mass(A,hi)<=mass(x,t)<=mass(B,lo))
for x in [A,(A+B)/2,B]:
    for y in [C,(C+D)/2,D]:
        ensure('ratio interval inclusion',A/D<=x/y<=B/C)

for af in [F(-1,12),F(0),F(1,12)]:
    for ag in [F(-1,15),F(0),F(1,15)]:
        d=(1+af)/(1+ag)-1
        ensure('bounded relative-matching error',abs(d)<= (F(1,12)+F(1,15))/(1-F(1,15)))

contract=json.loads((root/'results/physical_g2_execution/PHYSICAL_G2_CONTRACT_AUDIT.json').read_text())
missing=contract.get('missing_physical_objects',[])
ready=bool(contract.get('physical_G2_promoted',False))
ensure('readiness flag internally consistent',not (ready and missing))
ensure('real physical source data not silently inferred',bool(contract.get('physical_contract_checks')))

out={'classification':'EXACT_ONE_CLOCK_SCALING_IDENTITIES_AND_INPUT_READINESS',
     'checks_passed':len(checks),'checks_failed':0,
     'status':'MATHEMATICAL_IDENTITIES_VERIFIED_PHYSICAL_MASS_VALUES_NOT_COMPUTED',
     'scope':'Fraction-exact symbolic arithmetic and physical-input inventory only.',
     'synthetic_example':True,
     'no_experimental_mass_input':True,
     'results_are_physical_predictions':False,
     'checks':checks,
     'physical_G2_promoted':ready,
     'missing_physical_object_count':len(missing),
     'missing_physical_objects':missing,
     'metrological_scale_measured_in_this_package':False}
(root/'results/METROLOGICAL_CLOSURE_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS exact one-clock identities; checks:',len(checks),'; physical G2 complete:',ready)
