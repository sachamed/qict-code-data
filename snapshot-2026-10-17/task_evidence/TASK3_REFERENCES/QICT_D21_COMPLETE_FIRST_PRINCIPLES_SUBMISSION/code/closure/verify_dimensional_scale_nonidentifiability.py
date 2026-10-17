#!/usr/bin/env python3
"""Exact dimensional-analysis certificate for the QICT clock scale.

No experimental particle data are read.  The calculation asks whether a time
scale can be constructed from dimensionless microscopic QICT data together
with hbar and c.  It then shows how the obstruction disappears if an
independent gravitational constant G is supplied/derived.
"""
from fractions import Fraction
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)

# dimension vectors in (M,L,T)
HBAR=(Fraction(1),Fraction(2),Fraction(-1))
C=(Fraction(0),Fraction(1),Fraction(-1))
G=(Fraction(-1),Fraction(3),Fraction(-2))
TARGET_T=(Fraction(0),Fraction(0),Fraction(1))

def solve2(v1,v2,t):
    # exact brute solve from first two non-singular component equations and verify all
    for i in range(3):
      for j in range(i+1,3):
        det=v1[i]*v2[j]-v1[j]*v2[i]
        if det:
          a=(t[i]*v2[j]-t[j]*v2[i])/det
          b=(v1[i]*t[j]-v1[j]*t[i])/det
          if all(a*v1[k]+b*v2[k]==t[k] for k in range(3)):
            return a,b
    return None

def solve3(cols,t):
    # Gaussian elimination over Fractions
    A=[[cols[j][i] for j in range(3)]+[t[i]] for i in range(3)]
    for p in range(3):
        q=next(i for i in range(p,3) if A[i][p])
        A[p],A[q]=A[q],A[p]
        s=A[p][p]; A[p]=[x/s for x in A[p]]
        for i in range(3):
            if i!=p and A[i][p]:
                s=A[i][p]; A[i]=[A[i][j]-s*A[p][j] for j in range(4)]
    return tuple(A[i][3] for i in range(3))

hc=solve2(HBAR,C,TARGET_T)
hcg=solve3((HBAR,C,G),TARGET_T)
assert hc is None
assert hcg==(Fraction(1,2),Fraction(-5,2),Fraction(1,2))

out={
  'status':'PASS_DIMENSIONAL_SCALE_NONIDENTIFIABILITY',
  'claim_status':'PROVEN',
  'external_particle_data_used':False,
  'basis_dimensions':'(M,L,T)',
  'available_without_gravity':{'hbar':['1','2','-1'],'c':['0','1','-1'],'microscopic_QICT_numbers':'dimensionless'},
  'target_time_dimension':['0','0','1'],
  'solution_using_hbar_and_c':None,
  'theorem':'No monomial hbar^a c^b times a dimensionless QICT invariant has dimensions of time. Therefore the numerical value of tau_* cannot be derived from the present dimensionless microscopic structure plus hbar and c alone.',
  'gravity_extension_control':{
    'G_dimension':['-1','3','-2'],
    'solution_exponents':{'hbar':'1/2','c':'-5/2','G':'1/2'},
    'time_scale':'sqrt(hbar G/c^5)'
  },
  'consequence':'An absolute mass m_f=(hbar/(c^2 tau_*)) epsilon_f requires either independent copy-time metrology or a separately derived dimensionful sector such as gravity. Dimensionless mass ratios do not require tau_*.'
}
(CERT/'DIMENSIONAL_SCALE_NONIDENTIFIABILITY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
