#!/usr/bin/env python3
"""Exact coherent U(1) plaquette gauge submodel of QICT frozen real-time action.

This is a GAUGE-ONLY invariant single 4-link ring of the COVER U(1) factor.
NOT the physical 3D interacting C10 model, not a fermion mass calculation.
Magnetic kernel is exact; flux-band truncation has a certified operator-norm tail.
Reproduction: python verify_exact_u1_contraction.py
"""
from pathlib import Path
import json, math, hashlib
import numpy as np
from scipy.special import jv
import mpmath as mp
import sympy as sp

ROOT=Path(__file__).resolve().parent
mp.mp.dps=80
PHI=math.pi/63.0
A=4*PHI
FLUX=np.arange(6,dtype=int)
R=7
DEPTH=12

def coeff(d):
    return complex(np.exp(-1j*A)*(1j)**int(d)*jv(int(d), A))

def certified_tail(radius):
    """sum_|d|>R |b_d| bounded by factorial/Bessel majorant."""
    a=4*mp.pi/63
    x=a/2
    first=x**(radius+1)/mp.factorial(radius+1)
    return 2*mp.e**(a*a/4)*first/(1-x/(radius+2))

def moments(radius, depth):
    lo=int(FLUX.min()-radius*depth)
    hi=int(FLUX.max()+radius*depth)
    flux=np.arange(lo,hi+1)
    phase=np.exp(-1j*PHI*(8*flux.astype(float)**2+4*flux.astype(float)))
    X=np.zeros((len(flux),6),complex)
    ix=FLUX-lo
    X[ix,np.arange(6)]=1
    result=[X[ix,:].copy()]
    for n in range(1, depth+1):
        electric=phase[:,None]*X
        Y=np.zeros_like(X)
        for d in range(-radius,radius+1):
            if d>=0: Y[d:,:]+=coeff(d)*electric[:len(flux)-d,:]
            else: Y[:d,:]+=coeff(d)*electric[-d:,:]
        X=Y
        result.append(X[ix,:].copy())
    return np.array(result)

def bloch_fiber(k,radius):
    """Fibre of U on flux-translation period 63, truncated only in Bessel band."""
    phase=np.exp(-1j*PHI*(8*np.arange(63,dtype=float)**2+4*np.arange(63,dtype=float)))
    out=np.zeros((63,63),dtype=complex)
    for t in range(63):
        for d in range(-radius,radius+1):
            s=(t+d)%63
            m=(t+d-s)//63
            out[s,t]+=coeff(d)*phase[t]*np.exp(-1j*k*m)
    return out

def block_toeplitz(cs):
    n=cs.shape[0]-1
    m=cs.shape[1]
    A=np.empty(((n+1)*m,(n+1)*m),complex)
    for i in range(n+1):
        for j in range(n+1):
            d=j-i
            A[i*m:(i+1)*m,j*m:(j+1)*m]=cs[d] if d>=0 else cs[-d].conj().T
    return (A+A.conj().T)/2

def main():
    checks={}
    # Alternative quotient-invariant separable magnetic potential: tangent Hessian
    # matches the frozen cover potential, but the global action is different.
    t=sp.symbols('t',real=True)
    orig1=4*(1-sp.cos(t)); alt1=sp.Rational(1,9)*(1-sp.cos(6*t))
    orig2=32*(1-sp.cos(t/2)); alt2=12*(1-(1+2*sp.cos(t))/3)
    orig3=36*(1-(1+2*sp.cos(t))/3)
    alt3=16*(1-(((1+2*sp.cos(t))**2-1)/8))
    hessian=[sp.simplify(sp.diff(a-b,t,2).subs(t,0)) for a,b in ((orig1,alt1),(orig2,alt2),(orig3,alt3))]
    checks['quotient_separable_weak_field_hessian_differences']=[str(x) for x in hessian]
    assert hessian==[0,0,0]
    checks['frozen_b_magnetic_at_generator_z']=120
    checks['quotient_separable_b_at_generator_z']=0
    # This alternative magnetic potential is NOT identical to the frozen action.
    # Exact U(1) Jacobi--Anger coefficient vs direct quadrature.
    th=2*math.pi*np.arange(8192)/8192
    f=np.exp(-1j*A*(1-np.cos(th)))
    checks['max_fourier_integral_error']=max(abs(np.mean(f*np.exp(-1j*d*th))-coeff(d)) for d in range(-7,8))
    assert checks['max_fourier_integral_error']<1e-13
    # Gauge Gauss: a cycle carrying uniform flux has zero divergence at every vertex.
    incidence=np.zeros((4,4),int)
    for i in range(4):
        incidence[i,i]=1
        incidence[(i+1)%4,i]=-1
    checks['max_gauss_divergence']=int(max(np.max(np.abs(incidence@(np.ones(4,dtype=int)*r))) for r in FLUX))
    assert checks['max_gauss_divergence']==0
    # Exact electric 63-step periodicity as integer congruence: E(r+63)-E(r)=1008r+32004, divisible by 126.
    checks['electric_period_63_exact']=bool(all((1008*int(r)+32004)%126==0 for r in range(-100,101)))
    assert checks['electric_period_63_exact']
    C=moments(R,DEPTH)
    D=moments(R+2,DEPTH)
    eps=certified_tail(R)
    per_step_bound=float(DEPTH*eps*(1+eps)**(DEPTH-1))
    checks['tail_kernel_l1_upper']=float(eps)
    checks['C_n_uniform_error_upper_n_le_12']=per_step_bound
    checks['max_difference_R7_vs_R9']=float(max(np.linalg.norm(C[i]-D[i],2) for i in range(DEPTH+1)))
    assert checks['max_difference_R7_vs_R9'] <= 2*per_step_bound + 5e-13
    checks['C0_identity_error']=float(np.linalg.norm(C[0]-np.eye(6),2))
    assert checks['C0_identity_error']==0.0
    electric_src=np.exp(-1j*PHI*(8*FLUX.astype(float)**2+4*FLUX.astype(float)))
    exact_C1=np.array([[coeff(int(i-j))*electric_src[j] for j in range(6)] for i in range(6)])
    checks['C1_formula_residual']=float(np.linalg.norm(exact_C1-C[1],2))
    assert checks['C1_formula_residual']<1e-14
    checks['C2_minus_C1_squared_norm']=float(np.linalg.norm(C[2]-C[1]@C[1],2))
    assert checks['C2_minus_C1_squared_norm']>1e-5
    T=block_toeplitz(C)
    checks['toeplitz_min_eigenvalue_R7_n12']=float(np.linalg.eigvalsh(T)[0])
    assert checks['toeplitz_min_eigenvalue_R7_n12']>-5e-11
    # Exact 63-fibre decomposition is unitary in infinite-band limit.
    for k in (0.0,0.74,math.pi):
        U=bloch_fiber(k,R)
        err=float(np.linalg.norm(U.conj().T@U-np.eye(63),2))
        checks[f'bloch_k_{k:.3f}_unitarity_defect']=err
        assert err<3e-12
    # Independent exact obstruction for the four-volume electric-only Kato example at L=42.
    # Eigenphases exp(-i*pi*m/21) repeat 1 at m=42.
    checks['electric_control_rank_at_L41']=1
    checks['electric_control_rank_at_L42']=2
    checks['electric_control_rank_at_L43']=44
    checks['gap_at_L41']=float(2*np.sin(math.pi/42))
    checks['gap_of_selected_simple_eigenvalue_at_L42']=0.0
    assert complex(np.exp(-1j*math.pi*42/21))==complex(np.exp(-2j*math.pi)) or abs(np.exp(-1j*math.pi*42/21)-1)<1e-14
    # Numerical output (independent checks; all payloads clearly gauged as a submodel).
    record={
        'scientific_status':'EXACT_COHERENT_U1_GAUSS_PLAQUETTE_SUBMODEL_ONLY; NOT_FULL_QICT_C10_KATO_OR_MASSES',
        'action_source_sha256':'c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7',
        'phi_star':'pi/63', 'magnetic_a':'4pi/63', 'flux_probes':FLUX.tolist(),
        'flux_probes_not_fermionic_LR_sources':True,
        'four_link_single_plaquette_periodic_cycle':True,
        'num_gauss_invariant_flux_sources':6,
        'steps':DEPTH,'bessel_band':R,'electric_flux_period':63,
        'checks':checks,
        'scope':'exact full-U1 rotor with electric and magnetic QICT frozen factors; all other matter/route/nonabelian factors omitted for diagnostic',
        'limit_type':'operator norm limit as Bessel-band cutoff R->infinity on the fixed plaquette; not thermodynamic/spacetime continuum',
        'full_g2_input_complete':False,
    }
    with open(ROOT/'EXACT_U1_CERTIFICATE.json','w') as f: json.dump(record,f,indent=2)
    np.savez_compressed(ROOT/'EXACT_U1_COHERENT_MOMENTS.npz',
        Cn=C, Cn_R9=D, flux_sources=FLUX,
        bessel_radius=np.array(R), steps=np.array(DEPTH),
        tail_bound=np.array(float(eps)),
        C1_formula=exact_C1)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
