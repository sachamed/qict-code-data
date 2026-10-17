#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from fractions import Fraction

# Exact spherical moments for a uniformly distributed unit vector in R^3:
# E[n_x^(2a)n_y^(2b)n_z^(2c)] = ((2a-1)!!(2b-1)!!(2c-1)!!)/( (2(a+b+c)+1)!! )
def dfact_odd(k:int)->int:
    if k <= 0: return 1
    out=1
    for j in range(k,0,-2): out*=j
    return out

def moment(a,b,c):
    return Fraction(dfact_odd(2*a-1)*dfact_odd(2*b-1)*dfact_odd(2*c-1), dfact_odd(2*(a+b+c)+1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    # S4 = sum n_i^4; b=(11-8S4)/32.
    ES4 = 3*moment(2,0,0)
    ES4sq = 3*moment(4,0,0) + 6*moment(2,2,0)
    Eb = (Fraction(11)-8*ES4)/32
    Eb2 = (Fraction(121)-176*ES4+64*ES4sq)/1024
    Varb = Eb2-Eb*Eb
    out={
      'classification':'PHOTON_DIRECTIONAL_MOMENTS',
      'sphere_average_b':str(Eb),
      'sphere_average_b_float':float(Eb),
      'sphere_variance_b':str(Varb),
      'sphere_variance_b_float':float(Varb),
      'sphere_moment_S4':str(ES4),
      'sphere_moment_S4_squared':str(ES4sq),
      'high_symmetry':{
        'axis':{'b':'3/32','d':'-17/2048'},
        'face_diagonal':{'b':'7/32','d':'-203/6144'},
        'body_diagonal':{'b':'25/96','d':'-1963/55296'},
      },
      'overconstraint_statement':'one BCC orientation and one energy scale fix both the quadratic and quartic directional maps'
    }
    assert Eb == Fraction(31,160)
    assert Varb == Fraction(1,525)
    Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out, indent=2))
if __name__=='__main__': main()
