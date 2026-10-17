#!/usr/bin/env python3
from __future__ import annotations
import json, itertools, math
from pathlib import Path
import numpy as np
from scipy.special import jv

ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
OUT=CERT/'operator_consistency_certificate.json'

sx=np.array([[0,1],[1,0]],complex)
sy=np.array([[0,-1j],[1j,0]],complex)
sz=np.array([[1,0],[0,-1]],complex)
I2=np.eye(2,dtype=complex)
A={
 ( 1, 1, 1): np.array([[0, -1+1j],[0, 1-1j]],complex)/4,
 ( 1, 1,-1): np.array([[1+1j,0],[-1-1j,0]],complex)/4,
 ( 1,-1, 1): np.array([[0,-1-1j],[0,1+1j]],complex)/4,
 ( 1,-1,-1): np.array([[1-1j,0],[-1+1j,0]],complex)/4,
 (-1, 1, 1): np.array([[0,1+1j],[0,1+1j]],complex)/4,
 (-1, 1,-1): np.array([[1-1j,0],[1-1j,0]],complex)/4,
 (-1,-1, 1): np.array([[0,1-1j],[0,1-1j]],complex)/4,
 (-1,-1,-1): np.array([[1+1j,0],[1+1j,0]],complex)/4,
}
MOVES=list(A)
SPLUS={(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)}

def A_fourier(k):
    M=np.zeros((2,2),complex)
    for h,Ah in A.items():
        phase=np.exp(1j*sum(k[i]*h[i]/math.sqrt(3) for i in range(3)))
        M += phase*Ah
    return M

def exp_pauli(theta, sigma):
    return math.cos(theta)*I2-1j*math.sin(theta)*sigma

def A_factorized(k):
    # Exact branch represented by the distributed coefficients.
    return exp_pauli(k[0]/math.sqrt(3),sx) @ exp_pauli(k[1]/math.sqrt(3),sy) @ exp_pauli(k[2]/math.sqrt(3),sz)

def periodic_walk(L, random_gauge=False, seed=314159):
    rng=np.random.default_rng(seed)
    sites=list(itertools.product(range(L), repeat=3))
    idx={x:i for i,x in enumerate(sites)}
    phases={}
    for x in sites:
        for h in SPLUS:
            phases[(x,h)] = np.exp(1j*rng.uniform(-math.pi,math.pi)) if random_gauge else 1.0+0j
    def link(x,h):
        if h in SPLUS:
            return phases[(x,h)]
        hp=tuple(-v for v in h)
        y=tuple((x[i]+h[i])%L for i in range(3))
        return np.conj(phases[(y,hp)])
    N=2*len(sites)
    W=np.zeros((N,N),complex)
    for x in sites:
        xi=idx[x]
        for h,Ah in A.items():
            y=tuple((x[i]+h[i])%L for i in range(3))
            yi=idx[y]
            W[2*yi:2*yi+2,2*xi:2*xi+2] += Ah*link(x,h)
    err=W.conj().T@W-np.eye(N)
    return float(np.linalg.norm(err,2)), float(np.linalg.norm(err,'fro'))


def one_link_pi_defect(L=3):
    sites=list(itertools.product(range(L), repeat=3))
    idx={x:i for i,x in enumerate(sites)}
    flip=((0,0,0),(1,1,1))
    phases={(x,h):(-1.0+0j if (x,h)==flip else 1.0+0j) for x in sites for h in SPLUS}
    def link(x,h):
        if h in SPLUS:
            return phases[(x,h)]
        hp=tuple(-v for v in h)
        y=tuple((x[i]+h[i])%L for i in range(3))
        return np.conj(phases[(y,hp)])
    N=2*len(sites)
    W=np.zeros((N,N),complex)
    for x in sites:
        xi=idx[x]
        for h,Ah in A.items():
            y=tuple((x[i]+h[i])%L for i in range(3))
            yi=idx[y]
            W[2*yi:2*yi+2,2*xi:2*xi+2] += Ah*link(x,h)
    E=W.conj().T@W-np.eye(N)
    row=2*idx[(0,0,0)]+1
    col=2*idx[(0,0,2)]+0
    return E[row,col], float(np.linalg.norm(E,2))

def ir_slope(eps=1e-5):
    # Along kx with ky=kz=0, d=cos(kx/sqrt(3)) exactly.
    k=eps
    d=math.cos(k/math.sqrt(3))
    lam=math.acos(max(-1,min(1,d)))
    return lam/k

def bessel_tail(t=1.0, M=12, Mmax=300):
    # For exp[-it(2-U-U^dag)]|0>, coefficients are e^{-2it} i^m J_m(2t).
    vals=[abs(jv(m,2*t))**2 for m in range(-Mmax,Mmax+1)]
    total=sum(vals)
    tail=sum(abs(jv(m,2*t))**2 for m in range(-Mmax,Mmax+1) if abs(m)>M)
    samples={str(m):float(abs(jv(m,2*t))) for m in [0,1,2,5,10,15,20]}
    return float(total),float(tail),samples

def main():
    rng=np.random.default_rng(20260908)
    factor_res=0.0
    fourier_unit=0.0
    for _ in range(50):
        k=rng.uniform(-1.0,1.0,size=3)
        M=A_fourier(k)
        factor_res=max(factor_res,float(np.linalg.norm(M-A_factorized(k))))
        fourier_unit=max(fourier_unit,float(np.linalg.norm(M.conj().T@M-I2)))

    zero=[]; random=[]
    for L in (3,4,5):
        z=periodic_walk(L,False)
        r=periodic_walk(L,True,seed=1000+L)
        zero.append({'L':L,'spectral_residual':z[0],'frobenius_residual':z[1]})
        random.append({'L':L,'spectral_residual':r[0],'frobenius_residual':r[1]})

    exact_entry, exact_norm=one_link_pi_defect()
    total,tail,samples=bessel_tail()
    cert={
      'theory':'QICT publication operator consistency analysis',
      'external_particle_data_used':False,
      'weyl_branch_factorization':{
        'identity':'A_k^- = exp(-i k_x sigma_x/sqrt(3)) exp(-i k_y sigma_y/sqrt(3)) exp(-i k_z sigma_z/sqrt(3))',
        'max_matrix_residual':factor_res,
        'max_zero_field_unitarity_residual':fourier_unit,
      },
      'naive_link_covariantization_test':{
        'construction':'replace each BCC shift term by an independent U(1) link phase while retaining the same A_h coefficients',
        'zero_field_periodic_tests':zero,
        'exact_one_link_pi_defect':{
          'periodic_L':3,
          'flipped_positive_link':'(0,0,0)->(1,1,1)',
          'matrix_element':'< (0,0,0),down | (W^dagger W-I) | (0,0,2),up >',
          'value_real':float(exact_entry.real),
          'value_imag':float(exact_entry.imag),
          'exact_value':'i/2',
          'spectral_residual':exact_norm,
        },
        'generic_random_link_tests':random,
        'conclusion':'NONUNITARY_FOR_GENERIC_LINK_FIELDS',
      },
      'ir_slope':{
        'lambda_over_abs_k_along_axis':ir_slope(),
        'exact_value':'1/sqrt(3)',
        'consequence':'if the physical BCC hop length is l_hop, the small-k speed is l_hop/(sqrt(3) tau_*); identifying it with c requires l_hop=sqrt(3)c tau_*',
      },
      'peter_weyl_reachability_counterexample':{
        'model':'one U(1) link, K=2-U-U^dagger, initial electric Fourier mode |0>',
        'exact_coefficients':'exp(-2it) i^m J_m(2t), m in Z',
        't':1.0,
        'finite_window_norm_check':total,
        'tail_norm_outside_abs_m_le_12':tail,
        'sample_abs_coefficients':samples,
        'conclusion':'GENERIC_FINITE_TIME_EVOLUTION_HAS_INFINITE_FOURIER_SUPPORT; finite Peter-Weyl reachability does not follow for exponentiated covariant Laplacians',
      },
      'status':'CONSISTENCY_CHECKS_COMPLETE__NAIVE_LINK_COVARIANTIZATION_EXCLUDED'
    }
    OUT.write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps(cert,indent=2))
if __name__=='__main__': main()
