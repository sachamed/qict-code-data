#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np

def randu(rng,n):
    X=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); Q,R=np.linalg.qr(X)
    d=np.diag(R); ph=np.where(np.abs(d)>0,d/np.abs(d),1); return Q@np.diag(np.conj(ph))

def density(rng,n):
    X=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); A=X@X.conj().T; return A/np.trace(A)

def main():
    rng=np.random.default_rng(20260919); maxerr=0.0; rows=[]
    # Electron active carrier is color singlet, hence D=D_EW tensor I_3 and U_f=U_EW,f tensor U_3,
    # U_0=U_EW,0 tensor U_3 with the identical color update.
    for n in (1,2,5,8,9,10):
        A0=randu(rng,4); Af=randu(rng,4); D=randu(rng,4)
        U3=randu(rng,3); rEW=density(rng,4); r3=density(rng,3)
        fullrho=np.kron(rEW,r3); fullD=np.kron(D,np.eye(3))
        Uf=np.kron(Af,U3); U0=np.kron(A0,U3)
        full=np.trace(fullrho@fullD.conj().T@np.linalg.matrix_power(Uf,n)@fullD@np.linalg.matrix_power(U0.conj().T,n))
        ew=np.trace(rEW@D.conj().T@np.linalg.matrix_power(Af,n)@D@np.linalg.matrix_power(A0.conj().T,n))
        err=abs(full-ew); maxerr=max(maxerr,err); rows.append({'n':n,'residual_abs':float(err)})
    assert maxerr<2e-13
    out={
      'status':'PASS_EXACT_SU3_CANCELLATION_IN_RELATIVE_LEPTON_CORRELATOR',
      'max_residual_abs':float(maxerr),'tests':rows,
      'identity':'If D_f=D_EW tensor I_c and U_f=U_EW,f tensor U_c, U_0=U_EW,0 tensor U_c, then D_f^dag U_f^n D_f U_0^{-n}=(D_EW^dag U_EW,f^n D_EW U_EW,0^{-n}) tensor I_c. The color factor cancels before taking the state trace, for pure or mixed color states.',
      'scope':'Exact charged-lepton relative-moment cancellation. No assumption that the color vacuum is pure or an eigenvector is required.'
    }
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
