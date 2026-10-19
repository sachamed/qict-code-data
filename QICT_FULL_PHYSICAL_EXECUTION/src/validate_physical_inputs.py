"""Numerical screening primitives; positive screening is not an independent physics certificate."""
import numpy as np
SOURCE_ORDER=("L1","L2","L3","R1","R2","R3")

def check_density(rho, factor, tol=1e-9):
    rho=np.asarray(rho)
    factor=np.asarray(factor)
    if rho.ndim!=2 or rho.shape[0]!=rho.shape[1] or factor.ndim!=2 or factor.shape[0]!=rho.shape[0]:
        raise ValueError("Incorrect Kato density/factor dimensions")
    if rho.shape[0]>4096: raise ValueError("Requires separate large-scale certificate")
    if not np.isfinite(rho).all() or not np.isfinite(factor).all():
        raise ValueError("Nonfinite density")
    gram=factor@factor.conj().T
    tr=gram.trace().real
    if not tr>0: raise ValueError("Degenerate Gram factor")
    gram/=tr
    if np.linalg.norm(rho-gram)>tol: raise ValueError("Gram-factor identity fails")
    if np.linalg.norm(rho-rho.conj().T)>tol: raise ValueError("Not Hermitian")
    if abs(rho.trace()-1)>tol: raise ValueError("Trace != 1")
    if np.linalg.eigvalsh((rho+rho.conj().T)/2).min() < -tol:
        raise ValueError("Density not positive")
    return {"status":"DIAGNOSTIC_ONLY","dimension":int(rho.shape[0])}

def source_isometry(V,tol=1e-10):
    V=np.asarray(V)
    if V.ndim!=2 or V.shape[1]!=6: raise ValueError("Not six source columns")
    deviation=np.linalg.norm(V.conj().T@V-np.eye(6),ord=2)
    if deviation>tol: raise ValueError("Source isometry fails")
    return float(deviation)

def block_toeplitz(C):
    C=np.asarray(C)
    if C.ndim!=3 or C.shape[1:]!=(6,6) or len(C)<2:
        raise ValueError("Requires N+1 six-by-six moments")
    n=len(C)
    if n*6>2500: raise ValueError("Use independently certified sparse bound")
    T=np.zeros((6*n,6*n),dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            k=i-j
            T[6*i:6*i+6,6*j:6*j+6]=C[k] if k>=0 else C[-k].conj().T
    return T

def moment_diagnostic(C,errors,source_order=SOURCE_ORDER,tol=1e-10):
    C=np.asarray(C)
    errors=np.asarray(errors,dtype=float)
    if tuple(source_order)!=SOURCE_ORDER: raise ValueError("Incorrect source order")
    if C.shape[1:]!=(6,6) or len(C)!=len(errors): raise ValueError("Shape mismatch")
    if np.linalg.norm(C[0]-np.eye(6),ord=2)>tol: raise ValueError("C0 != I6")
    if not np.isfinite(errors).all() or (errors<0).any(): raise ValueError("Invalid error bounds")
    T=block_toeplitz(C)
    eigenmin=float(np.linalg.eigvalsh((T+T.conj().T)/2)[0])
    bound=float(errors[0]+2*sum(errors[1:]))
    return {"status":"DIAGNOSTIC_ONLY","toeplitz_float64_min":eigenmin,
            "reported_error_norm_upper":bound,"certified_psd":False}
