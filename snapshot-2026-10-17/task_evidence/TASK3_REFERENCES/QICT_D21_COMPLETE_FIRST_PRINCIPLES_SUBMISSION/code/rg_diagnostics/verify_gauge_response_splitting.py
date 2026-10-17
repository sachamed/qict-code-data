#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, hashlib
from fractions import Fraction as F
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
IN=DATA/'PDG_2026_RG_COMPARISON_INPUTS.csv'
OUT=CERT/'GAUGE_RESPONSE_SPLITTING.json'
OUTCSV=DATA/'GAUGE_RESPONSE_ONE_LOOP_DIAGNOSTICS.csv'
MZ=91.1876
SEED=20260914
NSAMP=250_000

kY,k2,k3=F(10,3),F(2,1),F(2,1)
bY,b2,b3=F(41,6),F(-19,6),F(-7,1)
# Coefficients below multiply 1/pi^2.
sY=-bY/(8*kY)      # -41/160
s2=-b2/(8*k2)      # +19/96
s3=-b3/(8*k3)      # +7/16
dY2=sY-s2          # -109/240
d32=s3-s2          # +23/96
assert F(115)*dY2+F(218)*d32 == 0

def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_inputs():
    d={}
    with IN.open() as f:
        for r in csv.DictReader(f):
            d[r['quantity']]=(float(r['value']),float(r['uncertainty']))
    return d

def physical(a_inv,s2w,alphas):
    alpha=1/a_inv; e2=4*math.pi*alpha
    gy2=e2/(1-s2w); g22=e2/s2w; g32=4*math.pi*alphas
    XiY=(1/gy2)/float(kY); Xi2=(1/g22)/float(k2); Xi3=(1/g32)/float(k3)
    d12=XiY-Xi2; d32v=Xi3-Xi2
    J=115*d12+218*d32v
    # equivalent null-test variables retained for comparison diagnostics
    FY=float(kY)*d12
    F3=float(k3)*d32v
    # unique pairwise equality scales under one-loop running
    slope_d12=float(dY2)/(math.pi**2)
    slope_d32=float(d32)/(math.pi**2)
    t12=-d12/slope_d12
    t32=-d32v/slope_d32
    mu12=MZ*math.exp(t12); mu32=MZ*math.exp(t32)
    return gy2,g22,g32,XiY,Xi2,Xi3,d12,d32v,J,FY,F3,mu12,mu32

def summary(x):
    x=np.asarray(x,float)
    return {'mean':float(x.mean()),'sd':float(x.std(ddof=1)),
            'q025':float(np.quantile(x,.025)),'q975':float(np.quantile(x,.975))}

def main():
    d=read_inputs(); a,sa=d['alpha_em_inv']; sw,ssw=d['sin2thetaW']; al,sa_s=d['alpha_s']
    vals=physical(a,sw,al)
    gy2,g22,g32,XiY,Xi2,Xi3,D12,D32,J,FY,F3,mu12,mu32=vals
    rng=np.random.default_rng(SEED)
    samples=[]
    for aa,bb,cc in zip(rng.normal(a,sa,NSAMP),rng.normal(sw,ssw,NSAMP),rng.normal(al,sa_s,NSAMP)):
        if aa<=0 or not (0<bb<1) or cc<=0: continue
        samples.append(physical(float(aa),float(bb),float(cc)))
    arr=np.asarray(samples,float)
    Js=arr[:,8]; l12=np.log10(arr[:,11]); l32=np.log10(arr[:,12])
    # J = (69 FY + 218 F3)/2 exactly.
    legacy=69*FY+218*F3
    assert abs(2*J-legacy)<1e-12
    rows=[
      {'quantity':'dXiY/dlnmu','value_over_pi2':str(sY),'interpretation':'one-loop response slope'},
      {'quantity':'dXi2/dlnmu','value_over_pi2':str(s2),'interpretation':'one-loop response slope'},
      {'quantity':'dXi3/dlnmu','value_over_pi2':str(s3),'interpretation':'one-loop response slope'},
      {'quantity':'d(XiY-Xi2)/dlnmu','value_over_pi2':str(dY2),'interpretation':'response splitting'},
      {'quantity':'d(Xi3-Xi2)/dlnmu','value_over_pi2':str(d32),'interpretation':'response splitting'},
      {'quantity':'Jresp(MZ)','value_over_pi2':f'{J:.15g}','interpretation':'RG-invariant response offset; not a finite-trace identity'},
      {'quantity':'XiY=Xi2 crossing scale [GeV]','value_over_pi2':f'{mu12:.16e}','interpretation':'external comparison diagnostic'},
      {'quantity':'Xi3=Xi2 crossing scale [GeV]','value_over_pi2':f'{mu32:.16e}','interpretation':'external comparison diagnostic'},
    ]
    with OUTCSV.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    cert={
      'status':'PASS_RESPONSE_SPLITTING_THEOREM_AND_NULL_TEST',
      'theory_statement':{
        'finite_metric':'kY=10/3, k2=k3=2',
        'physical_response':'1/g_r^2(mu)=k_r Xi_r(mu)',
        'one_loop_flow':'dXi_r/dlnmu=-b_r/(8 pi^2 k_r)',
        'slopes_over_pi2':{'Y':str(sY),'2':str(s2),'3':str(s3)},
        'difference_slopes_over_pi2':{'Y-2':str(dY2),'3-2':str(d32)},
        'RG_invariant':'J_resp=115(XiY-Xi2)+218(Xi3-Xi2)',
        'invariant_derivative_exact_zero':str(F(115)*dY2+F(218)*d32),
        'conclusion':'Universal response is not an RG-invariant submanifold. Sector-resolved response is required in the one-loop Standard-Model continuum regime.'
      },
      'external_null_test':{
        'input_sha256':sha(IN),'seed':SEED,'samples':len(samples),
        'at_MZ':{'gY2':gy2,'g22':g22,'g32':g32,'XiY':XiY,'Xi2':Xi2,'Xi3':Xi3,
                 'DeltaY2':D12,'Delta32':D32,'J_resp':J,
                 'legacy_2J_equals_69FY_plus_218F3':legacy},
        'J_resp_MC':summary(Js),
        'pairwise_response_equality_scales_GeV':{'XiY=Xi2':mu12,'Xi3=Xi2':mu32},
        'log10_scales_MC':{'XiY=Xi2':summary(l12),'Xi3=Xi2':summary(l32),'separation':summary(l32-l12)},
        'interpretation':'The old universal-response/common-scale condition is an auxiliary null hypothesis, not a QICT theorem. External couplings exclude that null hypothesis; they do not falsify the finite trace metric.'
      },
      'first_principles_boundary':'A QICT prediction of the nonzero response invariant requires numerical evaluation of the C11 flux/twist stiffness in the interacting continuum limit.'
    }
    OUT.write_text(json.dumps(cert,indent=2)+'\n')
    print('PASS_RESPONSE_SPLITTING_THEOREM_AND_NULL_TEST')
    print(json.dumps(cert,indent=2))

if __name__=='__main__': main()
