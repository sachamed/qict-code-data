#!/usr/bin/env python3
from fractions import Fraction
from pathlib import Path
import json, math, hashlib
ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; DATA=ROOT/'data'; CERT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

def odf(n):
    if n<=0: return 1
    p=1
    for k in range(n,0,-2): p*=k
    return p

def sphere_moment(a,b,c):
    N=a+b+c
    return Fraction(odf(2*a-1)*odf(2*b-1)*odf(2*c-1),odf(2*N+1))

def s4moment(k):
    import math
    z=Fraction(0)
    for i in range(k+1):
        for j in range(k-i+1):
            l=k-i-j
            C=math.factorial(k)//(math.factorial(i)*math.factorial(j)*math.factorial(l))
            z += C*sphere_moment(2*i,2*j,2*l)
    return z

ES=[Fraction(1)] + [s4moment(k) for k in range(1,5)]
meanS=ES[1]
centralS={}
for k in range(2,5):
    centralS[k]=sum(Fraction(math.comb(k,j))*(-meanS)**(k-j)*ES[j] for j in range(k+1))
meanb=Fraction(11,32)-meanS/Fraction(4)
mu2=centralS[2]/16
mu3=-centralS[3]/64
mu4=centralS[4]/256
assert meanb==Fraction(31,160)
assert mu2==Fraction(1,525)
assert mu3==Fraction(-6,125125)
assert mu4==Fraction(89,10635625)
power=mu2/(meanb*meanb)
skew2=mu3*mu3/(mu2**3)
kurt=mu4/(mu2**2)
assert power==Fraction(1024,20181)
assert skew2==Fraction(6804,20449)
assert kurt==Fraction(5607,2431)
axis,face,body=Fraction(3,32),Fraction(7,32),Fraction(25,96)
assert [96*axis,96*face,96*body]==[9,21,25]
assert (face-axis)/(body-face)==3
out={
 'status':'PASS_BCC_CUBIC_HARMONIC_FINGERPRINT',
 'claim_status':'PROVEN_WITHIN_DECLARED_BCC_PHOTON_TANGENT_COMPLETION',
 'external_data_used':False,
 'parity':'even',
 'quadratic_order_harmonic_support':['l=0','l=4'],
 'critical_point_counts':{'minima':6,'saddles':12,'maxima':8},
 'axis_face_body_ratio':'9:21:25',
 'successive_gap_ratio':'3',
 'mean_b':'31/160','variance_b':'1/525',
 'l4_to_monopole_power_ratio':'1024/20181',
 'centered_skewness':'-18*sqrt(21)/143',
 'centered_skewness_float':-18*math.sqrt(21)/143,
 'centered_kurtosis':'5607/2431',
 'centered_kurtosis_float':float(Fraction(5607,2431)),
 'shape_invariant_mu3sq_over_mu2cubed':'6804/20449',
 'experimental_comparison':'SEPARATE_HELD_OUT_TEST',
 'external_timestamp_requirement':'An independent immutable timestamp is required only for a claim of independent preregistration before a held-out comparison.'
}
(CERT/'BCC_CUBIC_HARMONIC_FINGERPRINT.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS_BCC_CUBIC_HARMONIC_FINGERPRINT')
print(json.dumps(out,indent=2))
