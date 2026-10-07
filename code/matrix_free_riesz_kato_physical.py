#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time, resource, hashlib
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def contour_nodes(center: complex, radius: float, nquad: int):
    th=2*np.pi*(np.arange(nquad)+0.5)/nquad
    return center+radius*np.exp(1j*th)


def shifted(A: LinearOperator, z: complex):
    n=A.shape[0]
    return LinearOperator((n,n), matvec=lambda x: z*x-A.matvec(x), dtype=np.complex128)


def projector_action(A: LinearOperator, B: np.ndarray, center: complex, radius: float, nquad: int,
                     rtol: float, maxiter: int, uniform_gap_lower_bound: float|None=None):
    B=np.asarray(B,np.complex128)
    if B.ndim==1: B=B[:,None]
    n,k=B.shape
    if A.shape!=(n,n): raise ValueError('operator/source mismatch')
    out=np.zeros((n,k),np.complex128)
    nodes=contour_nodes(center,radius,nquad)
    solve_rows=[]
    worst_res=0.0; worst_cert=0.0
    for qi,z in enumerate(nodes):
        M=shifted(A,z)
        X=np.empty((n,k),np.complex128)
        for j in range(k):
            x,info=gmres(M,B[:,j],rtol=rtol,atol=0.0,maxiter=maxiter,restart=min(80,n))
            r=B[:,j]-M.matvec(x)
            rn=float(np.linalg.norm(r)); bn=float(np.linalg.norm(B[:,j])); rel=rn/max(bn,1e-300)
            cert=(rn/uniform_gap_lower_bound) if uniform_gap_lower_bound and uniform_gap_lower_bound>0 else None
            solve_rows.append({'node':qi,'col':j,'info':int(info),'residual_norm':rn,'relative_residual':rel,
                               'solution_error_upper_bound_from_gap':cert})
            if info!=0: raise RuntimeError(f'GMRES failed node={qi} col={j} info={info}')
            worst_res=max(worst_res,rn)
            if cert is not None: worst_cert=max(worst_cert,cert)
            X[:,j]=x
        out += ((z-center)/nquad)*X
    return out, {'solve_count':len(solve_rows),'worst_residual_norm':worst_res,
                 'worst_solution_error_upper_bound_from_gap':worst_cert if uniform_gap_lower_bound else None,
                 'solves':solve_rows}


def low_rank_density_factor(Y: np.ndarray, tol: float=1e-13):
    # Orthonormal basis of projected seed range. Density is QQ*/rank, serialized via Gram factor Q/sqrt(rank).
    U,s,_=np.linalg.svd(np.asarray(Y,np.complex128),full_matrices=False)
    if s.size==0 or s[0]==0: raise ValueError('projected seed block has zero range')
    keep=s>tol*s[0]
    Q=U[:,keep]; r=Q.shape[1]
    if r==0: raise ValueError('zero retained range')
    F=Q/math.sqrt(r)
    return F,s,keep


def synthetic_self_test():
    # Diagonal unitary-like normal operator; contour around eigenvalue 1 selects first two basis vectors.
    diag=np.array([1,1,-1,1j,-1j,np.exp(1j*np.pi/3)],np.complex128)
    A=LinearOperator((6,6),matvec=lambda x:diag*x,dtype=np.complex128)
    B=np.eye(6,dtype=np.complex128)
    center=1+0j; radius=.1; nquad=128
    # exact uniform gap on this circle for this normal diagonal test: min distance to spectrum = radius
    gap=radius
    Y,cert=projector_action(A,B,center,radius,nquad,rtol=1e-13,maxiter=100,uniform_gap_lower_bound=gap)
    exact=np.diag([1,1,0,0,0,0]).astype(np.complex128)
    err=float(np.linalg.norm(Y-exact))
    F,s,keep=low_rank_density_factor(Y)
    rho=F@F.conj().T
    rho_exact=.5*exact
    rhoerr=float(np.linalg.norm(rho-rho_exact))
    return {'status':'PASS_MATRIX_FREE_RIESZ_KATO_SELFTEST' if err<1e-10 and rhoerr<1e-10 else 'FAIL',
            'projector_action_fro_error':err,'rho_fro_error':rhoerr,'retained_rank':int(np.count_nonzero(keep)),
            'singular_values':s.tolist(),'uniform_gap_lower_bound':gap,
            'quadrature_error_source':'analytic normal-matrix self-test only',
            'physical_promotion_allowed':False,
            'solver':{k:v for k,v in cert.items() if k!='solves'},
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope':'Infrastructure self-test only. Physical promotion requires a complete charged U_cone LinearOperator, independently certified uniform contour gap, independent quadrature error bound, and certified seed-subspace coverage.'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();
    if not a.self_test: raise SystemExit('This shipped driver is fail-closed: only --self-test is enabled until a physical charged-U_cone callback is supplied.')
    t=time.time(); out=synthetic_self_test(); out['runtime_s']=time.time()-t
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
