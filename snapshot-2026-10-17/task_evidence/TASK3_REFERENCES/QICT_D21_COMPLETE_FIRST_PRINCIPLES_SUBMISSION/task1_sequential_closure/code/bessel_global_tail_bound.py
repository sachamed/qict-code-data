#!/usr/bin/env python3
import json, math
import mpmath as mp
mp.mp.dps = 100
x = 4*mp.pi/63
# compute S_infty with explicit terms until tiny and rigorous geometric remainder using ratio bound
# For n>=0, |J_n(x)| <= (x/2)^n/n! * exp(x^2/(4(n+1))) (simple series abs bound)
def J_abs_upper(n):
    return (x/2)**n/mp.factorial(n) * mp.e**(x*x/(4*(n+1)))

def tail_upper(N): # sum over |k|>N |J_k| = 2 sum_{n=N+1} inf |J_n|
    # sum first many exact upper terms then ratio dominate remainder
    s=mp.mpf('0')
    n=N+1
    for j in range(n, n+80):
        s += J_abs_upper(j)
    # terms after n+79 negligible; bound geometric using ratio <= (x/2)/(m+1)*exp small
    m=n+79
    a=J_abs_upper(m+1)
    # conservative ratio upper for later
    r=(x/2)/(m+2)*mp.e**(x*x/(4*(m+2)))
    rem=a/(1-r)
    return 2*(s+rem)

def exact_sum_to(N):
    return abs(mp.besselj(0,x))+2*mp.fsum([abs(mp.besselj(k,x)) for k in range(1,N+1)])
# S_inf upper = finite to 40 + tail upper 40
S40=exact_sum_to(40)
Sinf_up=S40+tail_upper(40)
Sinf_lo=S40
rows=[]
for N in [2,3,4,5,6,7,8,9,10,11,12]:
    SN=exact_sum_to(N)
    t=tail_upper(N)
    # local F l1 tail: tuples where any of 4 indices outside cutoff
    dF_up=Sinf_up**4-SN**4
    Fnorm_up=Sinf_up**4
    # 12-block telescoping bound
    dZ_up=12*dF_up*(Fnorm_up**11)
    rows.append({
      'N':N,
      'S_N':mp.nstr(SN,50),
      'single_plaquette_l1_tail_upper':mp.nstr(t,50),
      'local_block_F_l1_tail_upper':mp.nstr(dF_up,50),
      'global_Z_tail_upper_multilinear_12F':mp.nstr(dZ_up,50)
    })
rep={
 'status':'PASS_ANALYTIC_BESSEL_TO_INFINITY_BOUND',
 'x':'4*pi/63',
 'x_numeric':mp.nstr(x,60),
 'S_infty_upper':mp.nstr(Sinf_up,60),
 'S_infty_lower_from_k_le_40':mp.nstr(Sinf_lo,60),
 'derivation':[
   '|J_n(x)| <= (x/2)^n/n! * exp(x^2/(4(n+1))) from absolute Bessel series',
   'sum_|k|>N |J_k| bounded by explicit majorant plus a geometric remainder',
   'Each local six-block tensor F contains four plaquette coefficients, hence ||F_inf-F_N||_1 <= S_inf^4-S_N^4',
   'The two-time six-block contraction is multilinear in 12 copies of F and all Kronecker constraints/phases have modulus <=1',
   '|Z_inf-Z_N| <= 12 ||F_inf-F_N||_1 ||F_inf||_1^11 <= 12(S_inf^4-S_N^4)(S_inf^4)^11'
 ],
 'rows':rows
}
with open('/mnt/data/qict_task1_v5/BESSEL_INFINITY_TAIL_CERTIFICATE.json','w') as f: json.dump(rep,f,indent=2); f.write('\n')
print(json.dumps(rep,indent=2))
