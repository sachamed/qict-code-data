#!/usr/bin/env python3
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'
OUT=CERT/'BCC_GAUGE_PHOTON_BRIDGE.json'
C={'U(1)':2.0,'SU(2)':4.0,'SU(3)':3.0}
N=2  # minimal integer giving a strict stability margin on the full BCC Brillouin zone

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def Wbcc(q):
    x,y,z=map(float,q); return 1-math.cos(x)*math.cos(y)*math.cos(z)
def lam(q): return 2*Wbcc(q)  # normalized so lambda(q)=|q|^2+O(|q|^4), lambda_max=4

def phase(q):
    L=lam(q); return 2*N*math.asin(math.sqrt(max(0.0,L))/(2*N))

def radial_speed(r,n,h=1e-6):
    n=np.asarray(n,float); n/=np.linalg.norm(n)
    return (phase((r+h)*n)-phase((r-h)*n))/(2*h)

def main():
    # Direct-dual reciprocity gives equal electric/magnetic Hessian coefficient c_r.
    # One split subcycle has cos(theta)=1-2(eta_r c_r)^2 lambda.
    # Common IR light cone: 2N eta_r c_r=1. Strict full-BZ stability: eta c sqrt(lambda_max)=1/N<1 => N>=2.
    eta={r:1/(2*N*c) for r,c in C.items()}
    stability={r:eta[r]*c*2 for r,c in C.items()} # sqrt(lambda_max)=2
    assert all(abs(2*N*eta[r]*C[r]-1)<1e-15 for r in C)
    assert max(stability.values())<1 and N==2
    # Expansion of lambda_BCC along q=r n:
    # lambda = r^2 + r^4(S4/6-1/4)+O(r^6)
    # omega=4 asin(sqrt(lambda)/4)=r + (8S4-11) r^3/96 + O(r^5)
    # v/c=1 - b(n) r^2+..., b=(11-8 S4)/32. Since E/E_tau=omega=r+O(r^3), same b at O(E^2).
    dirs={'axis':[1,0,0],'face_diagonal':[1,1,0],'body_diagonal':[1,1,1]}
    checks={}
    for name,v in dirs.items():
        n=np.asarray(v,float); n/=np.linalg.norm(n); S4=float(np.sum(n**4)); b=(11-8*S4)/32
        rs=[0.02,0.015,0.01,0.008]
        estimates=[(1-radial_speed(r,n))/r**2 for r in rs]
        checks[name]={'S4':S4,'b_exact':b,'numeric_b':estimates,'max_error':max(abs(x-b) for x in estimates)}
        assert checks[name]['max_error']<2e-4
    bmin=3/32; bmax=25/96
    cert={
      'status':'PASS_BCC_GAUGE_TANGENT_PHOTON_BRIDGE',
      'taxonomy':{'linearized_compact_gauge_tangent':'DEMONSTRATED','full_interacting_photon_dispersion':'PROTOCOLIZED','external_photon_data_used':False},
      'premises':[
        'C8 direct-dual reciprocity gives equal electric and curvature quadratic Hessian coefficient c_r within each gauge factor.',
        'The spatial gauge tangent uses the range-one BCC scalar lambda_BCC(q)=2[1-cos(qx)cos(qy)cos(qz)], so lambda=q^2+O(q^4) and lambda_max=4.',
        'The receiver collision phase pi/63 is not identified with the gauge Courant coefficient.'
      ],
      'selection':{'subcycles_N':N,'eta_r':eta,'quadratic_c_r':C,'strict_stability_arguments':stability,
                   'proof':'Common cone requires 2 N eta_r c_r=1. Full-BZ strict stability gives 1/N<1, hence the smallest integer is N=2 and eta_r=1/(4 c_r).'},
      'universal_BCC_tangent_dispersion':'omega_gamma(q)=4 asin( sqrt(2[1-cos(qx)cos(qy)cos(qz)])/4 )',
      'small_energy_law':'v_gamma(E,n)/c = 1 - b(n)(E/E_tau)^2 + O((E/E_tau)^4), b(n)=(11-8 sum_i n_i^4)/32, E_tau=hbar/tau_star',
      'directional_bounds':{'b_min_axis':bmin,'b_max_body_diagonal':bmax,'ratio_max_over_min':bmax/bmin},
      'directional_checks':checks,
      'old_identification_removed':'Using the receiver collision phase pi/63 as the gauge kinetic Courant step gives factor-dependent tangent light cones and is not retained in present formulation.',
      'new_continuous_fit_parameter_added':False,
      'source_sha256':sha(__file__)
    }
    OUT.write_text(json.dumps(cert,indent=2)+'\n')
    print('PASS_BCC_GAUGE_TANGENT_PHOTON_BRIDGE')
    print(json.dumps(cert,indent=2))
if __name__=='__main__': main()
