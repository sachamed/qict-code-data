#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    # Three-state Weyl pair.
    w=np.exp(2j*np.pi/3); X=np.roll(np.eye(3,dtype=complex),1,axis=0); Z=np.diag([1,w,w**2])
    weyl=float(np.linalg.norm(Z@X-w*X@Z,'fro'))
    # Hypercharges use right-handed convention in the construction; anomaly sums use left-handed conjugates.
    qQ,quc,qdc,qL,qec,qnc=1,-4,2,-3,6,0
    anomaly={
      'SU3_SU3_U1':2*qQ+quc+qdc,
      'SU2_SU2_U1':3*qQ+qL,
      'gravity_gravity_U1':6*qQ+3*quc+3*qdc+2*qL+qec+qnc,
      'U1_cubic':6*qQ**3+3*quc**3+3*qdc**3+2*qL**3+qec**3+qnc**3}
    # BCC quadratic anisotropy b=(11-8 sum n_i^4)/32.
    dirs={'axis':np.array([1.,0,0]),'face_diagonal':np.array([1.,1.,0])/math.sqrt(2),'body_diagonal':np.array([1.,1.,1.])/math.sqrt(3)}
    b={k:float((11-8*np.sum(v**4))/32) for k,v in dirs.items()}
    ratio=[x/(1/96) for x in [b['axis'],b['face_diagonal'],b['body_diagonal']]]
    result={'weyl_relation_fro_error':weyl,'hypercharge_anomaly_sums':anomaly,'bcc_directional_coefficients':b,'bcc_ratio_units_of_1_over_96':ratio}
    Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
