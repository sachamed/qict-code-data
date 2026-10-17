#!/usr/bin/env python3
"""Exact U(1) L=2 plaquette contraction by the integer dual-cycle representation.

For the periodic 2x2x2 complex, closed integer plaquette fluxes ker_Z(boundary_2)
have rank 10.  A primitive integral basis is obtained from seven dual scalar
potentials (one scalar is gauge-fixed) plus three seam cocycles.  The resulting
sum is evaluated exactly at a finite character cutoff without oscillatory Monte
Carlo.  No particle data enter.
"""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
from scipy.special import jv
from numba import njit

ROOT=Path(__file__).resolve().parents[2]
OUTDIR=ROOT/'results'/'microscopic_closure'; OUTDIR.mkdir(parents=True,exist_ok=True)
PHI=math.pi/63.0; X=4.0*PHI; NPLAQ=24


def wtab(K:int):
    # indices -K..K
    return np.asarray([(1j**k)*jv(k,X) for k in range(-K,K+1)],dtype=np.complex128)


def directional_factor(K:int, eps:int):
    """Integrate the primitive seam harmonic exactly for four parallel dual edges.

    delta_i = s(x_rho=0)-s(x_rho=1) on the four edge pairs.  Each pair contains
    two primal plaquettes.  The seam representative gives characters
       eps*(h+delta_i), eps*(-delta_i).
    h is summed over all integers; finite support of the cutoff reduces it to
    [-2K,2K].
    """
    D=2*K+1; wt=wtab(K); F=np.zeros(D**4,dtype=np.complex128)
    def W(k):
        return wt[k+K] if -K<=k<=K else 0j
    idx=0
    for d0 in range(-K,K+1):
      for d1 in range(-K,K+1):
       for d2 in range(-K,K+1):
        for d3 in range(-K,K+1):
          ds=(d0,d1,d2,d3)
          base=1+0j
          for d in ds: base*=W(-eps*d)
          sh=0j
          if base!=0:
            for h in range(-2*K,2*K+1):
              p=1+0j
              for d in ds:
                p*=W(eps*(h+d))
                if p==0: break
              sh+=p
          F[idx]=base*sh; idx+=1
    return F


@njit(cache=True)
def contract_cube(K:int,Fx,Fy,Fz):
    D=2*K+1
    total=0.0+0.0j
    # Tree increments: b=a+ib, c=a+ic, d=a+id, e=b+ie,
    # f=b+iff, g=c+ig, h=e+ih; a=0.  Each tree edge is already
    # constrained to [-K,K].  Five non-tree cube edges are checked explicitly.
    for ib in range(-K,K+1):
      b=ib
      for ic in range(-K,K+1):
       c=ic
       for idd in range(-K,K+1):
        d=idd
        for ie in range(-K,K+1):
         e=b+ie
         if abs(e-c)>K: continue  # x: c-e
         for iff in range(-K,K+1):
          f=b+iff
          if abs(f-d)>K: continue # x: d-f
          for ig in range(-K,K+1):
           g=c+ig
           if abs(g-d)>K: continue # y: d-g
           for ih in range(-K,K+1):
            h=e+ih
            if abs(h-g)>K: continue # x: g-h
            if abs(h-f)>K: continue # y: f-h
            # delta = scalar on rho=0 endpoint minus scalar on rho=1 endpoint.
            dx0=-b; dx1=c-e; dx2=d-f; dx3=g-h
            dy0=-c; dy1=b-e; dy2=d-g; dy3=f-h
            dz0=-d; dz1=b-f; dz2=c-g; dz3=e-h
            ix=(((dx0+K)*D+(dx1+K))*D+(dx2+K))*D+(dx3+K)
            iy=(((dy0+K)*D+(dy1+K))*D+(dy2+K))*D+(dy3+K)
            iz=(((dz0+K)*D+(dz1+K))*D+(dz2+K))*D+(dz3+K)
            total += Fx[ix]*Fy[iy]*Fz[iz]
    return total


def boundary_matrix_and_dual_basis():
    L=2
    sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
    links=[(mu,x,y,z) for mu in range(3) for x,y,z in sites]; li={t:i for i,t in enumerate(links)}
    pmeta=[]; plaquettes=[]
    for mu in range(3):
      for nu in range(mu+1,3):
       for x,y,z in sites:
        s=[x,y,z];sm=s.copy();sm[mu]=(sm[mu]+1)%L;sn=s.copy();sn[nu]=(sn[nu]+1)%L
        plaquettes.append([(li[(mu,*s)],1),(li[(nu,*sm)],1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)])
        pmeta.append((mu,nu,x,y,z))
    B=np.zeros((24,24),dtype=np.int64)
    for p,po in enumerate(plaquettes):
      for l,sgn in po:B[l,p]+=sgn
    def prev(s,r):
      a=list(s);a[r]=(a[r]-1)%L;return tuple(a)
    vars_=[s for s in sites if s!=(0,0,0)]
    D=np.zeros((24,10),dtype=np.int64)
    for c,sj in enumerate(vars_):
      ss={s:0 for s in sites};ss[sj]=1
      for p,(mu,nu,x,y,z) in enumerate(pmeta):
        s=(x,y,z);r=3-mu-nu
        eps=1 if (mu,nu,r) in ((0,1,2),(1,2,0),(2,0,1)) else -1
        D[p,c]=eps*(ss[s]-ss[prev(s,r)])
    for r in range(3):
      for p,(mu,nu,x,y,z) in enumerate(pmeta):
        s=(x,y,z);rr=3-mu-nu
        eps=1 if (mu,nu,rr) in ((0,1,2),(1,2,0),(2,0,1)) else -1
        D[p,7+r]=eps*(1 if rr==r and s[r]==0 else 0)
    # Fixed primitive 10x10 minor found once and rechecked on every run.
    rows=np.array([0,4,7,12,13,14,18,21,22,23],dtype=int)
    det=round(float(np.linalg.det(D[rows,:])))
    return B,D,rows,det


def one(K:int):
    B,D,rows,det=boundary_matrix_and_dual_basis()
    assert np.max(np.abs(B@D))==0
    assert np.linalg.matrix_rank(B)==14 and np.linalg.matrix_rank(D)==10 and abs(det)==1
    Fp=directional_factor(K,+1); Fm=directional_factor(K,-1)
    # x and z have epsilon +1; y has epsilon -1 for the ordered plaquettes
    # (yz),(xz),(xy), respectively.
    raw=contract_cube(K,Fp,Fm,Fp)
    z=np.exp(-1j*X*NPLAQ)*raw
    # conservative local character L1 remainder and telescoping product bound
    tail=2.0*sum(abs(jv(k,X)) for k in range(K+1,80))
    local_l1=sum(abs(jv(k,X)) for k in range(-K,K+1))
    # Full exponential character L1 norm <= exp(|X|). Product difference bound.
    product_tail=NPLAQ*tail*math.exp(abs(X)*(NPLAQ-1))
    return {
      'K':K,'Z_real':float(z.real),'Z_imag':float(z.imag),'abs_Z':float(abs(z)),
      'minus_log_abs':float(-math.log(abs(z))) if abs(z)>0 else None,
      'character_tail_l1_single_gate':float(tail),
      'conservative_24_gate_absolute_remainder_bound':float(product_tail),
      'dual_kernel_rank':10,'boundary_rank':14,'unimodular_minor_det':int(det),
      'unimodular_minor_rows':rows.tolist(),
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kmax',type=int,default=6);a=ap.parse_args()
    seq=[]
    for K in range(0,a.kmax+1):
      r=one(K);seq.append(r);print(json.dumps(r),flush=True)
    for i in range(1,len(seq)):
      z0=complex(seq[i-1]['Z_real'],seq[i-1]['Z_imag']);z1=complex(seq[i]['Z_real'],seq[i]['Z_imag'])
      seq[i]['delta_from_previous_abs']=float(abs(z1-z0))
    final=seq[-1]
    report={
      'status':'PASS_EXACT_U1_L2_CHARACTER_CONTRACTION',
      'external_particle_data_used':False,
      'geometry':{'L':2,'links':24,'plaquettes':24},
      'action':{'phi_star':'pi/63','A_U1':4,'x':'4*pi/63'},
      'method':'exact Haar projection -> integer closed-flux sum -> primitive dual 7-scalar + 3-seam-cocycle contraction',
      'integer_kernel_certificate':{'rank_boundary':14,'rank_kernel':10,'primitive_minor_det':final['unimodular_minor_det'],'rows':final['unimodular_minor_rows']},
      'sequence':seq,
      'frozen_cutoff':a.kmax,
      'frozen_Z':{'real':final['Z_real'],'imag':final['Z_imag'],'abs':final['abs_Z']},
      'frozen_minus_log_abs':final['minus_log_abs'],
      'rigorous_absolute_truncation_bound':final['conservative_24_gate_absolute_remainder_bound'],
    }
    txt=json.dumps(report,indent=2)+'\n'; out=OUTDIR/'L2_U1_EXACT_CHARACTER_CONTRACTION.json';out.write_text(txt)
    report['sha256_without_embedded_digest']=hashlib.sha256(txt.encode()).hexdigest()
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
