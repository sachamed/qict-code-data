#!/usr/bin/env python3
"""Exact Gauss-pair response for the COVER U(1) plaquette factors in QICT.

This is an external, static +/-q Gauss-charge sector, NOT dynamical fermion
matter and NOT the physical 3D C10 contraction. All couplings use the frozen
microstep phi=pi/63 and Bessel parameter a=4phi. Sector q is an integer-valued
boundary/source superselection label, not a fitted microscopic parameter.
"""
from pathlib import Path
import json, math
import numpy as np
import mpmath as mp
from scipy.special import jv

HERE=Path(__file__).resolve().parent
PHI=np.pi/63
A=4*PHI
N=12
R=7
PROBES=np.arange(6,dtype=int)
mp.mp.dps=70

# Rows vertices, cols directed cycle links (i -> i+1 mod 4)
D=np.zeros((4,4),int)
for edge in range(4):
    D[edge,edge]=1
    D[(edge+1)%4,edge]=-1
ones=np.ones(4,dtype=int)

def source_string(q, path="short"):
    assert type(q) is int
    if path == 'short': return np.array([q,0,0,0],dtype=int)
    if path == 'long': return np.array([0,-q,-q,-q],dtype=int)
    raise ValueError(path)

def electric_integer(r, string):
    E=np.asarray(r,dtype=np.int64)[:,None]+np.asarray(string,dtype=np.int64)[None,:]
    return 2*np.sum(E**2,axis=1)+np.sum(E,axis=1)

def bessel(d):
    return np.exp(-1j*A)*(1j)**int(d)*jv(int(d),A)

def tail_bound(R):
    a=4*mp.pi/63; x=a/2
    first=x**(R+1)/mp.factorial(R+1)
    return float(2*mp.exp(a*a/4)*first/(1-x/(R+2)))

def cn(q, path='short', probes=PROBES, radius=R, depth=N):
    s=source_string(q,path)
    probes=np.array(probes,dtype=int)
    lo=int(probes.min()-depth*radius)
    hi=int(probes.max()+depth*radius)
    r=np.arange(lo,hi+1,dtype=int)
    electric=np.exp(-1j*PHI*electric_integer(r,s))
    X=np.zeros((len(r),len(probes)),complex)
    X[probes-lo,np.arange(len(probes))]=1.0
    moments=[X[probes-lo,:].copy()]
    for step in range(depth):
        Z=electric[:,None]*X
        Y=np.zeros_like(Z)
        for d in range(-radius,radius+1):
            b=bessel(d)
            if d>=0: Y[d:,:]+=b*Z[:len(r)-d,:]
            else: Y[:d,:]+=b*Z[-d:,:]
        X=Y
        moments.append(X[probes-lo,:].copy())
    return np.array(moments)

def toeplitz(C):
    num,dim=C.shape[0],C.shape[1]
    mat=np.zeros((num*dim,num*dim),dtype=complex)
    for i in range(num):
        for j in range(num):
            k=j-i
            mat[i*dim:(i+1)*dim,j*dim:(j+1)*dim]=C[k] if k>=0 else C[-k].conj().T
    return (mat+mat.conj().T)/2

def compare(a,b):
    return float(max(np.linalg.norm(x-y,ord=2) for x,y in zip(a,b)))

def run():
    checks={}
    checks['frozen_phi']='pi/63'
    checks['frozen_magnetic_a']='4pi/63'
    checks['meaning']='gauge U(1) with STATIC Gauss charged pair only, NOT dynamical QICT matter'
    checks['q_sectors']=[0,1,2,4]
    checks['no_fitted_gauge_coupling']=True
    checks['flux_sources']=[int(i) for i in PROBES]
    checks['moment_depth']=N
    checks['bessel_bandwidth']=R
    eps=tail_bound(R)
    err=(N*eps*(1+eps)**(N-1))
    checks['bessel_kernel_operator_tail_bound']=eps
    checks['uniform_moment_operator_bound_n_le_12']=err
    assert np.all(D@ones==0)
    arrays={}
    checked=[]
    for q in checks['q_sectors']:
        s=source_string(q,'short')
        s2=source_string(q,'long')
        target=np.array([q,-q,0,0],int)
        assert np.array_equal(D@s,target)
        assert np.array_equal(D@s2,target)
        assert np.array_equal(s2,s-q*ones)
        xs=np.arange(-50,51,dtype=np.int64)
        # Electric polynomial exact, incl external charges.
        formula=8*xs**2+(4+4*q)*xs+(2*q*q+q)
        assert np.array_equal(electric_integer(xs,s), formula)
        assert np.array_equal(electric_integer(xs+q,s2),formula)
        # Period 63 of electric Floquet phase in all integer q-sectors
        assert np.all((1008*xs+32004+252*q)%126==0)
        A1=cn(q,'short',PROBES,R,N)
        B1=cn(q,'long',PROBES+q,R,N)
        A2=cn(q,'short',PROBES,R+2,N)
        pathres=compare(A1,B1)
        radiusres=compare(A1,A2)
        c0err=float(np.linalg.norm(A1[0]-np.eye(6),2))
        mineig=float(np.linalg.eigvalsh(toeplitz(A1)).min())
        assert pathres < 8e-14,(q,pathres)
        assert radiusres < 2*err+1e-12,(q,radiusres,err)
        assert c0err < 1e-14
        assert mineig > -2e-10,(q,mineig)
        arrays[f'C_q{q}']=A1
        checked.append({'q':q,'path_relabelling_max_norm':pathres,
                'R7_R9_max_norm':radiusres,'C0_error':c0err,
                'Toeplitz_min_eig_truncation':mineig,
                'C2_minus_C1_square_operator_norm':float(np.linalg.norm(A1[2]-A1[1]@A1[1],2)),
                'C1_electric_diagonal_trace':str(np.trace(A1[1]))})
    # Shift covariance: K_(q+4)(r)=K_q(r+1)+(12q+24) in integer units.
    for q in [0,1,2]:
        r=np.arange(-15,16)
        assert np.array_equal(electric_integer(r,source_string(q+4,'short')),
                              electric_integer(r+1,source_string(q,'short'))+(12*q+24))
        C_plus=cn(q+4,'short',PROBES,R,N)
        C_shift=cn(q,'short',PROBES+1,R,N)
        phase=np.exp(-1j*PHI*(12*q+24)*np.arange(N+1))[:,None,None]
        diff=compare(C_plus,phase*C_shift)
        assert diff < 2e-13,(q,diff)
        checks[f'q{q}_to_q{q+4}_unitary_shift_moment_max_norm']=diff
    checks['sectors']=checked
    checks['provenance']='QICT_LATTICE_ACTION_FREEZE_V3, cover U(1) plaquette K_E and K_B factors only'
    checks['physical_interacting_C10_produced']=False
    checks['physical_fermion_masses_produced']=False
    checks['regulator_limit']='Bessel flux-band at fixed Gauss plaquette only; no thermodynamic/continuum limit'
    HERE.joinpath('GAUSS_CHARGED_SECTOR_CERTIFICATE.json').write_text(json.dumps(checks,indent=2)+'\n')
    np.savez_compressed(HERE/'GAUSS_CHARGED_SECTOR_MOMENTS.npz', **arrays,
                        q_sectors=np.array(checks['q_sectors']), flux_probes=PROBES,
                        radius=np.array(R),moment_depth=np.array(N),
                        moment_error_bound=np.array(err))
    print(json.dumps({'bound':err,'sector_checks':checked,'status':'passed'},indent=2))

if __name__=='__main__':run()
