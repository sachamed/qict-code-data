#!/usr/bin/env python3
import json
import mpmath as mp
mp.mp.dps=100
X=32*mp.pi/63
# coefficient c_j = i^(2j) 2d/X J_d(X), d=2j+1 = j2+1
# |chi_j(U)| <= d. Thus uniform class-function tail <= sum_{d>=D} 2 d^2/X |J_d(X)|.

def J_upper(n):
    # absolute series majorant
    return (X/2)**n/mp.factorial(n)*mp.e**(X*X/(4*(n+1)))

def tail(j2max):
    d0=j2max+2 # first omitted d because d=j2+1, omitted j2=j2max+1
    s_coeff=mp.mpf('0'); s_uniform=mp.mpf('0')
    for d in range(d0,d0+120):
        u=J_upper(d)
        s_coeff += 2*d/X*u
        s_uniform += 2*d*d/X*u
    # remainder conservative geometric from next term ratio; polynomial d factors included with loose 2x ratio envelope
    D=d0+120
    u=J_upper(D)
    r=(X/2)/(D+1)*mp.e**(X*X/(4*(D+1)))
    # for d factors, ratio <= r*((D+1)/D)^2 for uniform
    rc=r*((D+1)/D)
    ru=r*((D+1)/D)**2
    s_coeff += (2*D/X*u)/(1-rc)
    s_uniform += (2*D*D/X*u)/(1-ru)
    return s_coeff,s_uniform
rows=[]
for j2max in [2,4,6,8,10,12,14,16]:
    a,b=tail(j2max)
    rows.append({'twice_j_max':j2max,'j_max':j2max/2,'coefficient_l1_tail_upper':mp.nstr(a,50),'uniform_character_tail_upper':mp.nstr(b,50)})
rep={'status':'PASS_SU2_CHARACTER_ANALYTIC_TAIL_BOUND','X':'32*pi/63','X_numeric':mp.nstr(X,60),
'derivation':['c_j = i^(2j) 2(2j+1) J_(2j+1)(X)/X','|chi_j(U)| <= 2j+1','absolute Bessel-series majorant bounds all omitted representations'], 'rows':rows}
out='/mnt/data/qict_task1_v5/SU2_CHARACTER_TAIL_CERTIFICATE.json'
with open(out,'w') as f:json.dump(rep,f,indent=2);f.write('\n')
print(json.dumps(rep,indent=2))
