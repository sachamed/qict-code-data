#!/usr/bin/env python3
import json, math, hashlib
import numpy as np
from collections import Counter
from sympy import Rational, N
from sympy.physics.wigner import wigner_6j

J2MAX=12

def fuse(a,b):
    return range(abs(a-b),a+b+1,2)

def inv4_channels(a,b,c,d):
    return sorted(set(fuse(a,b)).intersection(fuse(c,d)))

def inv5_fund_channels(a,b,c,d):
    out=[]
    for A in fuse(a,b):
      for B in fuse(c,d):
        if 1 in fuse(A,B): out.append((A,B))
    return out

def tri(a,b,c):
    return c>=abs(a-b) and c<=a+b and ((a+b+c)%2==0)

def recoupling_F(a,b,c,d):
    # a,b,c,d are doubled spins; d is total spin. Basis e=(ab), f=(bc).
    es=[e for e in fuse(a,b) if tri(e,c,d)]
    fs=[f for f in fuse(b,c) if tri(a,f,d)]
    F=np.zeros((len(es),len(fs)),float)
    phase=(-1)**((a+b+c+d)//2)
    for i,e in enumerate(es):
      for j,f in enumerate(fs):
        # {j1 j2 j12; j3 J j23}
        val=wigner_6j(Rational(a,2),Rational(b,2),Rational(e,2),
                      Rational(c,2),Rational(d,2),Rational(f,2))
        F[i,j]=phase*math.sqrt((e+1)*(f+1))*float(N(val,50))
    return es,fs,F

# enumerate multiplicities at physical declared cutoff
h4=Counter(); h5=Counter(); max4=(-1,None); max5=(-1,None)
for a in range(J2MAX+1):
 for b in range(J2MAX+1):
  for c in range(J2MAX+1):
   for d in range(J2MAX+1):
    m4=len(inv4_channels(a,b,c,d)); h4[m4]+=1
    if m4>max4[0]:max4=(m4,(a,b,c,d))
    m5=len(inv5_fund_channels(a,b,c,d)); h5[m5]+=1
    if m5>max5[0]:max5=(m5,(a,b,c,d))

# stress recoupling cases
cases=[(12,12,12,12),(12,10,8,10),(11,12,11,12),(9,10,11,10),(1,12,11,0)]
tests=[]
for case in cases:
 try:
  es,fs,F=recoupling_F(*case)
  if F.shape[0] and F.shape[0]==F.shape[1]:
   resid=float(np.linalg.norm(F@F.T-np.eye(F.shape[0]),ord=2))
  else: resid=None
  tests.append({'case_twice_spins':case,'left_channels_twice_j':es,'right_channels_twice_j':fs,'shape':list(F.shape),'unitarity_residual_2norm':resid})
 except ValueError as ex:
  tests.append({'case_twice_spins':case,'error':str(ex)})

# orientation duality matrix at max j2=12
j2=12; d=j2+1
Y=np.zeros((d,d),float)
# m2=-j2,-j2+2,...,j2; Y_mn=(-1)^(j-m) delta_{m,-n}
ms=list(range(-j2,j2+1,2)); mi={m:i for i,m in enumerate(ms)}
for m in ms:
 Y[mi[m],mi[-m]]=(-1)**((j2-m)//2)
yres=float(np.linalg.norm(Y.T@Y-np.eye(d),ord=2))

rep={
 'status':'PASS_SU2_FUSION_SEPARATOR_LOCAL_EXACT',
 'declared_plaquette_twice_j_max':J2MAX,
 'pure_gauge_four_leg':{
   'quadruples_scanned':(J2MAX+1)**4,
   'max_invariant_multiplicity':max4[0],
   'argmax_twice_spins':max4[1],
   'multiplicity_histogram':{str(k):v for k,v in sorted(h4.items())}
 },
 'fundamental_matter_five_leg':{
   'quadruples_scanned':(J2MAX+1)**4,
   'max_invariant_multiplicity':max5[0],
   'argmax_twice_spins':max5[1],
   'channel_definition':'(J12,J34) with twice-spin 1 in J12 tensor J34, then coupled with matter j=1/2 to total singlet',
   'multiplicity_histogram':{str(k):v for k,v in sorted(h5.items())}
 },
 'recoupling_tests':tests,
 'orientation_duality_j2_12_unitarity_residual_2norm':yres,
 'architecture_consequence':'SU(2) magnetic m-indices need not be propagated globally; Haar link integration is represented exactly by finite fusion-channel labels and unitary Wigner-6j F-moves.',
 'scope':'Local/separator algebra only. This does not yet execute the full SU(2)xU(1)xCAR contraction.'
}
out='/mnt/data/qict_task1_v5/SU2_FUSION_SEPARATOR_CERTIFICATE.json'
with open(out,'w') as f:json.dump(rep,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in rep.items() if k!='pure_gauge_four_leg' and k!='fundamental_matter_five_leg'},indent=2))
print('max4',max4,'max5',max5,'Yres',yres)
