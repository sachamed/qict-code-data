#!/usr/bin/env python3
from functools import lru_cache
import math, json, random
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan

def _fact(n:int)->float:
    if n < 0: return math.inf
    return math.gamma(n+1)

@lru_cache(maxsize=None)
def cg_fast(a2,ma2,b2,mb2,c2,mc2):
    if ma2 + mb2 != mc2: return 0.0
    if min(a2,b2,c2) < 0: return 0.0
    if abs(ma2)>a2 or abs(mb2)>b2 or abs(mc2)>c2: return 0.0
    if ((a2-ma2)|(b2-mb2)|(c2-mc2)) & 1: return 0.0
    if c2 < abs(a2-b2) or c2 > a2+b2 or ((a2+b2-c2)&1): return 0.0
    # All factorial arguments below are integers in doubled-spin notation.
    A=(a2+b2-c2)//2; B=(a2-b2+c2)//2; C=(-a2+b2+c2)//2; D=(a2+b2+c2)//2 + 1
    cm=(c2+mc2)//2; cmm=(c2-mc2)//2
    ap=(a2+ma2)//2; am=(a2-ma2)//2; bp=(b2+mb2)//2; bm=(b2-mb2)//2
    pref=math.sqrt((c2+1)*_fact(A)*_fact(B)*_fact(C)/_fact(D))
    pref*=math.sqrt(_fact(cm)*_fact(cmm)*_fact(ap)*_fact(am)*_fact(bp)*_fact(bm))
    # k denominators for the standard Condon-Shortley CG formula.
    kmin=max(0, (b2-c2-ma2)//2, (a2+mb2-c2)//2)
    kmax=min(A, am, bp)
    s=0.0
    for k in range(kmin,kmax+1):
        den=(_fact(k)*_fact(A-k)*_fact(am-k)*_fact(bp-k)*
             _fact((c2-b2+ma2)//2+k)*_fact((c2-a2-mb2)//2+k))
        if math.isfinite(den) and den!=0:
            s += (-1.0 if k&1 else 1.0)/den
    return pref*s

def cg_sym(a2,ma2,b2,mb2,c2,mc2):
    if ma2+mb2!=mc2:return 0.0
    return float(clebsch_gordan(Rational(a2,2),Rational(b2,2),Rational(c2,2),Rational(ma2,2),Rational(mb2,2),Rational(mc2,2)))

def self_test(j2max=8):
    maxerr=0.0; tests=0
    for a in range(j2max+1):
      for b in range(j2max+1):
       for c in range(abs(a-b),a+b+1,2):
        for ma in range(-a,a+1,2):
         for mb in range(-b,b+1,2):
          mc=ma+mb
          if abs(mc)>c or ((c-mc)&1): continue
          x=cg_fast(a,ma,b,mb,c,mc); y=cg_sym(a,ma,b,mb,c,mc)
          maxerr=max(maxerr,abs(x-y)); tests+=1
    if maxerr>2e-12: raise AssertionError(maxerr)
    return {'status':'PASS_FAST_RACAH_CG_COMPILER','twice_j_test_max':j2max,'coefficients_compared':tests,'max_abs_residual':maxerr}
if __name__=='__main__': print(json.dumps(self_test(),indent=2))
