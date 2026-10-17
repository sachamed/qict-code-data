#!/usr/bin/env python3
from __future__ import annotations
import json, math, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import su2_charged_openline_seed as ch
from fast_su2_cg import cg_fast
ch.cg=cg_fast
ROOT=HERE.parent; OUT=ROOT/'results'/'SU2_CHARGED_KRYLOV_D2.json'

def inner(a,b):
    if len(a)>len(b): a,b=b,a
    return sum(np.conjugate(v)*b.get(k,0j) for k,v in a.items())
def norm(a): return math.sqrt(max(0.0,float(inner(a,a).real)))
def axpy(y,a,x):
    for k,v in x.items():
        z=y.get(k,0j)+a*v
        if abs(z)<1e-14: y.pop(k,None)
        else: y[k]=z

def Hmat(v,j2max):
    out={}; calls=0
    for st,amp in v.items():
        if abs(amp)<1e-15: continue
        for pid in range(24):
            nb=ch.neighbors(st,pid,1,j2max); calls+=1
            for q,z in nb.items(): out[q]=out.get(q,0j)+amp*z
    out={k:v for k,v in out.items() if abs(v)>2e-13}
    return out,calls

def main():
    t=time.time(); s0=ch.seed((2,1,0)); q0={s0:1+0j}
    H0,c0=Hmat(q0,2); alpha0=inner(q0,H0); beta1=norm(H0)
    q1={k:v/beta1 for k,v in H0.items()}
    t1=time.time(); H1,c1=Hmat(q1,3)
    alpha1=inner(q1,H1)
    r=dict(H1); axpy(r,-alpha1,q1); axpy(r,-beta1,q0); beta2=norm(r)
    q2={k:v/beta2 for k,v in r.items()} if beta2 else {}
    # H^2|0> and m4 from H0 -> H(H0)=beta1 H q1
    H2v0={k:beta1*v for k,v in H1.items()}
    m2=float(inner(H0,H0).real); m4=float(inner(H2v0,H2v0).real)
    herm_res=abs(inner(q0,H1)-np.conjugate(inner(H0,q1)))
    checks={'alpha0_zero':bool(abs(alpha0)<2e-12),'alpha1_real':bool(abs(alpha1.imag)<2e-12),'beta1_sq_24':bool(abs(beta1*beta1-24)<2e-10),'beta2_positive':bool(beta2>1e-8),'q1_normalized':bool(abs(norm(q1)-1)<2e-12),'q2_normalized':bool(abs(norm(q2)-1)<2e-12),'m4_positive':bool(m4>m2*m2),'finite_hermiticity_regression':bool(herm_res<2e-11)}
    out={'status':'PASS_SU2_CHARGED_KRYLOV_D2' if all(checks.values()) else 'FAIL','support':{'q0':1,'Hq0':len(H0),'q1':len(q1),'Hq1':len(H1),'q2':len(q2)},'lanczos':{'alpha0':[float(alpha0.real),float(alpha0.imag)],'beta1':beta1,'alpha1':[float(alpha1.real),float(alpha1.imag)],'beta2':beta2},'moments':{'m0':1.0,'m1':float(alpha0.real),'m2':m2,'m4':m4},'matvec_calls':{'Hq0_plaquette_calls':c0,'Hq1_plaquette_calls':c1},'hermiticity_regression':float(herm_res),'seconds':time.time()-t,'seconds_second_matvec':time.time()-t1,'checks':checks,'scope':'Exact charged SU(2) fusion-tree Krylov through q2 with j support sufficient for two fundamental plaquette actions. This is a magnetic-sector Kato primitive, not yet the full electric+matter h_d contraction.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
