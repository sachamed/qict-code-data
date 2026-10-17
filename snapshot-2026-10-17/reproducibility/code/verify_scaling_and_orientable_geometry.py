#!/usr/bin/env python3
"""Exact comparison geometries and bandwise scaling controls."""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, math, platform
import scipy
import numpy as np
from scipy.linalg import expm, logm

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/scaling_and_geometry'

def comm(a,b):return a@b-b@a
def maxabs(a):return float(np.max(np.abs(a)))
def integer_components(a):
    assert np.all(a.real==np.rint(a.real))
    assert np.all(a.imag==np.rint(a.imag))
    assert np.max(np.abs(a.real))<2**40 and np.max(np.abs(a.imag))<2**40

def exact_zero(a):
    # Every input entry is a Gaussian integer. Matrix dimensions and entry
    # bounds keep every product and sum in the exact binary-integer range.
    integer_components(a)
    assert maxabs(a)==0

def geometry(r):
    w=2*r;c=4*r;n=w*c;dim=4*n
    eye=np.eye(dim,dtype=complex)
    gamma=np.diag(np.repeat([1,-1,-1,1],n)).astype(complex)
    swap=np.zeros((dim,dim),complex)
    for block in [0,1]:
        for i in range(w):
            for j in range(c):
                a=block*n+i*c+j;b=(block+2)*n+j*w+i
                swap[a,b]=swap[b,a]=1
    D=np.zeros((dim,dim),complex)
    for a,b in [(0,1),(2,3)]:
        D[a*n:(a+1)*n,b*n:(b+1)*n]=np.eye(n)
        D[b*n:(b+1)*n,a*n:(a+1)*n]=np.eye(n)
    def opposite(a):return swap@a.conj()@swap
    exact_zero(swap@swap-eye);exact_zero(opposite(D)-D)
    exact_zero(opposite(gamma)+gamma);exact_zero(D@gamma+gamma@D)
    exact_zero(D-D.conj().T)
    q=[np.array([[1j,0],[0,-1j]]),np.array([[0,1],[-1,0]],complex),
       np.array([[0,1j],[1j,0]])]
    def weak(a,side):
        b=np.zeros((dim,dim),complex)
        b[side*n:(side+1)*n,side*n:(side+1)*n]=np.kron(a,np.eye(c))
        return b
    def colour(a):
        b=np.zeros((dim,dim),complex)
        for side in [2,3]:
            b[side*n:(side+1)*n,side*n:(side+1)*n]=np.kron(a,np.eye(w))
        return b
    generators=[]
    for side in [0,1]:
        generators.append(weak(np.eye(w),side))
        for unit in q:generators.append(weak(np.kron(np.eye(r),unit),side))
        for i in range(r):
            e=np.zeros((r,r),complex);e[i,i]=1
            generators.append(weak(np.kron(e,np.eye(2)),side))
        for i in range(r-1):
            e=np.zeros((r,r),complex);e[i,i+1]=1
            generators.extend([weak(np.kron(e,np.eye(2)),side),weak(np.kron(e.T,np.eye(2)),side)])
    generators.extend([colour(np.eye(c)),colour(1j*np.eye(c))])
    e=np.zeros((c,c),complex);e[0,0]=1;generators.append(colour(e))
    for i in range(c-1):
        e=np.zeros((c,c),complex);e[i,i+1]=1
        generators.extend([colour(e),colour(e.T)])
    opposites=[opposite(b) for b in generators]
    checks=0
    for a in generators:
        exact_zero(comm(a,gamma))
        da=comm(D,a)
        for b in opposites:
            exact_zero(comm(a,b));exact_zero(comm(da,b));checks+=2
    eL=weak(np.eye(w),0);eR=weak(np.eye(w),1);eC=colour(np.eye(c))
    cycle=eL@opposite(eC)-eR@opposite(eC)-eC@opposite(eL)+eC@opposite(eR)
    exact_zero(cycle-gamma)
    # On the four inequivalent bimodules the joint commutant of the left and
    # opposite algebras is block-scalar. D equates the two particle scalars;
    # J equates antiparticle scalars to their conjugates. A self-adjoint
    # projection in that commutant is therefore either zero or the identity.
    basis=[np.eye(dim),np.diag(np.repeat([1j,1j,-1j,-1j],n))]
    for b in basis:
        for a in generators:exact_zero(comm(a,b))
        exact_zero(comm(D,b));exact_zero(opposite(b)-b)
    return {'r':r,'real_algebra_dimension':40*r*r,'complex_hilbert_dimension':dim,
       'generators_checked':len(generators),'order_zero_and_one_checks':checks,
       'KO6_and_orientation_residuals':'zero in Gaussian-integer arithmetic',
       'joint_real_commutant_analytic_dimension':2,'self_adjoint_commutant_analytic_dimension':1,
       'SU_colour_cubic_coefficient':-2*r+2*r,
       'symplectic_cubic_coefficient':'zero by transpose and pseudoreality',
       'mixed_simple_factor_coefficient':'zero by tracelessness',
       'global_symplectic_doublet_multiplicity':4*r}

def sine_interval(x,terms=25):
    value=sum(((-1)**k*x**(2*k+1)/math.factorial(2*k+1) for k in range(terms)),F(0))
    error=x**(2*terms+1)/math.factorial(2*terms+1)
    return value-error,value+error

def scaling():
    rows=[]
    for k in range(1,7):
        a=F(1,4**k);ca=1+F(1,2**k)
        mismatch=ca-1
        lo,hi=sine_interval(a/4)
        rows.append({'a':str(a),'speed':str(ca),
          'axis_generator_mismatch_exact':str(mismatch),
          'mismatch_divided_by_ca_a_exact':str(mismatch/(ca*a)),
          'half_step_error_interval':[str(2*lo),str(2*hi)],
          'half_step_error_over_old_bound_interval':[str(2*lo/a**2),str(2*hi/a**2)]})
        assert mismatch/(ca*a)>=1
        assert 2*lo/a**2>1
    sigma=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    max_step_ratio=max_log_ratio=max_scalar_ratio=0.0
    cases=0
    rng=np.random.default_rng(2026101401)
    directions=np.vstack([np.eye(3),np.ones(3)/np.sqrt(3),rng.normal(size=(20,3))])
    directions=directions/np.maximum(1,np.linalg.norm(directions,axis=1))[:,None]
    for a in [1/8,1/16,1/32]:
        ca=1+math.sqrt(a);tau=a/ca
        for p in directions:
            norm=np.linalg.norm(p);A=np.eye(2,dtype=complex)
            for x,S in zip(p,sigma):A=A@expm(-1j*a*x*S)
            H=sum((x*S for x,S in zip(p,sigma)),np.zeros((2,2),complex))
            step=np.linalg.norm(A-expm(-1j*a*H),2)
            logerror=np.linalg.norm(1j*logm(A)/tau-ca*H,2)
            lam=2*(1-np.prod(np.cos(a*p)))
            omega=4/tau*np.arcsin(np.sqrt(lam)/4)
            scalar=abs(omega-ca*norm)
            max_step_ratio=max(max_step_ratio,step/(a*a*norm*norm))
            max_log_ratio=max(max_log_ratio,logerror/(2*ca*a*norm*norm))
            max_scalar_ratio=max(max_scalar_ratio,scalar/((5/16)*ca*a*a*norm**3))
            assert step<=a*a*norm*norm+1e-13
            assert logerror<=2*ca*a*norm*norm+1e-12
            assert scalar<=(5/16)*ca*a*a*norm**3+1e-12
            cases+=1
    return {'exact_axis_and_clock_controls':rows,'numerical_cases':cases,
      'one_step_bound_ratio_max':max_step_ratio,'log_generator_bound_ratio_max':max_log_ratio,
      'scalar_bound_ratio_max':max_scalar_ratio,
      'numerical_role':'Floating-point cross-checks; the operator bounds are proved analytically.'}

def neutral_character():
    gamma=np.diag([1,-1]).astype(complex)
    swap=np.array([[0,1],[1,0]],complex)
    for u in [1,-1,1j,-1j]:
        exact_zero(u*np.eye(2)@swap@(u*np.eye(2)).conj()@swap-np.eye(2))
    projection=np.trace(gamma)/2*np.eye(2)
    residual=gamma-projection
    squared=np.trace(residual.conj().T@residual).real
    assert squared==2 and np.trace(residual)==0
    return {'neutral_gauge_residual':'zero in Gaussian-integer arithmetic',
      'minimum_squared_Frobenius_distance':int(squared),
      'minimum_argument':'Orthogonal projection onto scalar matrices; formula 2+2|z|^2.'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    record={'result':'SCALING_AND_ORIENTABLE_COMPARISON_CONTROLS_EXECUTED',
      'geometries':[geometry(1),geometry(2)],'scaling':scaling(),
      'neutral_character':neutral_character(),
      'comparison_models':'Algebraic witnesses; not alternative fitted QICT particle spectra.',
      'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    (OUT/'SCALING_AND_GEOMETRY_CERTIFICATE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='scaling'},indent=2))

if __name__=='__main__':main()
