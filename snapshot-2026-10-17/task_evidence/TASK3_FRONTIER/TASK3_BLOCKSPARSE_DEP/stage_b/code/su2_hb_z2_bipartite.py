#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,math
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
import su2_global_spin_network as gs

def gf2_solve(A,b):
    A=np.array(A,dtype=np.uint8)%2;b=np.array(b,dtype=np.uint8)%2
    M=np.c_[A,b];m,n=A.shape;r=0;piv=[]
    for c in range(n):
        rows=np.where(M[r:,c])[0]
        if len(rows)==0:continue
        q=r+rows[0];M[[r,q]]=M[[q,r]]
        for i in range(m):
            if i!=r and M[i,c]:M[i]^=M[r]
        piv.append(c);r+=1
        if r==m:break
    for i in range(r,m):
        if not M[i,:n].any() and M[i,n]:raise RuntimeError('inconsistent')
    x=np.zeros(n,dtype=np.uint8)
    for i,c in enumerate(piv):x[c]=M[i,n]
    return x,piv,M

def gamma(st,z):
    return -1 if (sum(int(z[e])*st.links[e] for e in range(24))%2) else 1

def self_test():
    A=np.zeros((24,24),dtype=np.uint8)
    for p,pl in enumerate(gs.PLAQ):
        for e in pl['edges']:A[p,e]^=1
    z,piv,_=gf2_solve(A,np.ones(24,dtype=np.uint8))
    synd=(A@z)%2;assert np.all(synd==1)
    # Every fundamental plaquette transition flips Gamma because all four j2 labels change parity.
    import su2_k0_orbit_krylov as kk
    vac=kk.canonical(gs.vacuum_state());v={vac:1+0j}; tested=0;bad=0
    for depth in range(3):
        for st in list(v)[:min(len(v),120)]:
            g0=gamma(st,z)
            for pid in range(24):
                for t,a in kk.wp_action(st,pid):
                    if abs(a)>1e-13:
                        tested+=1
                        if gamma(t,z)!=-g0:bad+=1
        v=kk.H_on_orbit_vector(v)
    assert bad==0
    # Exact even moments already executed by orbit Krylov.
    d3=json.load(open(os.path.join(os.path.dirname(__file__),'..','results','SU2_K0_ORBIT_KRYLOV_D3_CACHED.json')))
    d4=json.load(open(os.path.join(os.path.dirname(__file__),'..','results','SU2_K0_ORBIT_KRYLOV_D4_PARALLEL.json')))
    m0=1.0;m2=d3['vacuum_moments_HB_power'][2][0];m4=d4['vacuum_moment_HB4'][0]
    m6=d3['vector_norm_sq_by_depth'][3];m8=d4['norm2_d4']
    b1sq=m2
    b2sq=m4/m2-b1sq
    # symmetric Jacobi recurrence from moments; formulae obtained by Gram-Schmidt.
    # q2=(x^2-b1^2)/(b1*b2); norm of x q2 minus b2 q1 gives b3^2.
    b3sq=(m6-2*b1sq*m4+b1sq*b1sq*m2)/(b1sq*b2sq)-b2sq
    # Build orthonormal polynomial q3 and use m8 to get beta4.
    # q3 proportional x^3-(b1^2+b2^2)x.
    A2=b1sq+b2sq
    normp3=m6-2*A2*m4+A2*A2*m2
    # q3 norm denominator should b1^2*b2^2*b3^2.
    den=b1sq*b2sq*b3sq
    norm_res=abs(normp3-den)
    # beta4^2 = ||x q3 - beta3 q2||^2
    # numerator for ||x q3||^2 = m8-2 A2 m6+A2^2 m4 over den.
    xq3=(m8-2*A2*m6+A2*A2*m4)/den
    b4sq=xq3-b3sq
    return {'status':'PASS_SU2_HB_EXACT_Z2_BIPARTITE_SYMMETRY','gf2_rank':len(piv),
      'center_flip_link_ids':[int(i) for i,x in enumerate(z) if x],
      'center_flip_link_labels':[list(gs.LINKS[i]) for i,x in enumerate(z) if x],
      'all_plaquette_center_coboundary_minus_one':bool(np.all(synd==1)),
      'nonzero_transitions_checked':tested,'bipartite_transition_failures':bad,
      'theorem':'Let Gamma translate U_e -> -U_e on links with z_e=1. Since the Z2 coboundary of z is one on every plaquette, Gamma chi_1/2(U_p) Gamma=-chi_1/2(U_p), hence Gamma H_B Gamma=-H_B. The Haar vacuum is Gamma-even, so every odd vacuum moment vanishes exactly and the Lanczos Jacobi diagonal is identically zero.',
      'executed_even_moments':{'m0':m0,'m2':m2,'m4':m4,'m6':m6,'m8':m8},
      'lanczos_beta_squared':[b1sq,b2sq,b3sq,b4sq],
      'lanczos_beta':[math.sqrt(max(0,x)) for x in (b1sq,b2sq,b3sq,b4sq)],
      'q3_norm_identity_residual':norm_res,
      'computational_consequence':'Magnetic Krylov vectors alternate exactly between Gamma parity sectors. Odd diagonal Lanczos coefficients need not be evaluated and even/odd polynomial components can be propagated separately.'}
if __name__=='__main__':print(json.dumps(self_test(),indent=2))
