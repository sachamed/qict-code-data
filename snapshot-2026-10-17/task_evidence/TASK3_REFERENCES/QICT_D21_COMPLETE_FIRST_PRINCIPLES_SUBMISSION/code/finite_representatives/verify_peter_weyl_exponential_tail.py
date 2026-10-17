#!/usr/bin/env python3
from pathlib import Path
import json,csv
import mpmath as mp
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True); DATA=ROOT/'data'
mp.mp.dps=80

def tail_norm(M,t=mp.mpf('1')):
    # exact identity sum_{m in Z} J_m(2t)^2=1; compute positive tail directly.
    return 2*mp.nsum(lambda m: mp.besselj(m,2*t)**2,[M+1, mp.inf])

def main():
    t=mp.mpf('1'); rows=[]
    for M in range(0,41):
        tail=tail_norm(M,t)
        rows.append({'M':M,'tail_norm':mp.nstr(tail,30),'captured_norm':mp.nstr(1-tail,30)})
    with open(DATA/'peter_weyl_exponential_tail.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    samples={str(m):mp.nstr(mp.besselj(m,2*t),30) for m in [0,1,2,5,10,20,30,40]}
    cert={
      'model':'U(1) one-link covariant-Laplacian counterexample',
      'K':'2-U-U^*',
      'time':'1',
      'analytic_expansion':'exp(-2it) sum_{m in Z} i^m J_m(2t) exp(i m phi)',
      'finite_Peter_Weyl_support':False,
      'sample_Bessel_coefficients':samples,
      'tail_after_M8':rows[8]['tail_norm'],
      'tail_after_M12':rows[12]['tail_norm'],
      'tail_after_M20':rows[20]['tail_norm'],
      'interpretation':'For generic nonzero time the Fourier/Peter-Weyl series is infinite. Real-space causal support is finite, but representation support is not. Finite representation calculations require a controlled numerical truncation rather than an exact reachability claim.',
      'status':'PASS_COUNTEREXAMPLE'
    }
    (CERT/'peter_weyl_exponential_tail_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps(cert,indent=2))
if __name__=='__main__': main()
