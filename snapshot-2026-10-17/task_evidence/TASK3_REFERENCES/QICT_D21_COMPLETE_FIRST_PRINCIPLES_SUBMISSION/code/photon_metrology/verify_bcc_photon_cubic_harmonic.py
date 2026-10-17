#!/usr/bin/env python3
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; OUT=CERT/'BCC_PHOTON_CUBIC_HARMONIC.json'
SEED=20260914

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(n):
    n=np.asarray(n,float); n=n/np.linalg.norm(n); s4=float(np.sum(n**4)); return (11-8*s4)/32,s4

def main():
    # Define cubic l=4 harmonic H4 = sum n_i^4 - 3/5 on S^2.
    # Then b = 31/160 - H4/4 exactly.
    exact=[]
    for name,n in [('axis',[1,0,0]),('face',[1,1,0]),('body',[1,1,1])]:
        bv,s4=b(n); H4=s4-3/5; rhs=31/160-H4/4
        assert abs(bv-rhs)<1e-15
        exact.append({'direction':name,'S4':s4,'H4':H4,'b':bv})
    # On the unit sphere 1/3 <= S4 <= 1 by Cauchy and |n_i|<=1, hence 3/32 <= b <=25/96.
    rng=np.random.default_rng(SEED); x=rng.normal(size=(500000,3)); x/=np.linalg.norm(x,axis=1)[:,None]
    s4=np.sum(x**4,axis=1); bv=(11-8*s4)/32
    assert s4.min() >= 1/3-1e-12 and s4.max() <= 1+1e-12
    assert bv.min() >= 3/32-1e-12 and bv.max() <= 25/96+1e-12
    cert={
      'status':'PASS_BCC_PHOTON_CUBIC_HARMONIC_DECOMPOSITION',
      'taxonomy':'DEMONSTRATED',
      'identity':'b(n)=31/160-(1/4) H4(n), H4(n)=sum_i n_i^4-3/5',
      'interpretation':'The leading quadratic photon correction contains an isotropic coefficient 31/160 plus a pure cubic-symmetry l=4 anisotropy. It is not a fitted sky function.',
      'rigorous_bounds':{'S4_min':1/3,'S4_max':1,'b_min':3/32,'b_max':25/96},
      'high_symmetry_directions':exact,
      'monte_carlo_control':{'seed':SEED,'samples':len(x),'sample_b_min':float(bv.min()),'sample_b_max':float(bv.max())},
      'script_sha256':sha(__file__)
    }
    OUT.write_text(json.dumps(cert,indent=2)+'\n')
    print('PASS_BCC_PHOTON_CUBIC_HARMONIC_DECOMPOSITION')
    print(json.dumps(cert,indent=2))
if __name__=='__main__': main()
