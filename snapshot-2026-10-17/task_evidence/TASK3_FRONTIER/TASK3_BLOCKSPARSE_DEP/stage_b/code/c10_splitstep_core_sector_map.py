#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'C10_SPLITSTEP_CORE_SECTOR_MAP.json'
DIRS=list(itertools.product((-1,1),repeat=3))
L=2
SITES=[(x,y,z) for x in range(2) for y in range(2) for z in range(2)]
SID={s:i for i,s in enumerate(SITES)}
LINKS=[(mu,x,y,z) for mu in range(3) for x,y,z in SITES]
LID={e:i for i,e in enumerate(LINKS)}

def bcc_ball(R):
    seen={(0,0,0):0}; frontier=[(0,0,0)]
    for d in range(R):
        nxt=[]
        for x in frontier:
            for v in DIRS:
                y=tuple(x[i]+v[i] for i in range(3))
                if y not in seen:
                    seen[y]=d+1; nxt.append(y)
        frontier=nxt
    return seen

def potential(R):
    seen=bcc_ball(R); nodes=list(seen); idx={x:i for i,x in enumerate(nodes)}
    A=lil_matrix((len(nodes),len(nodes)),dtype=float)
    for x,i in idx.items():
        A[i,i]=8.0
        for v in DIRS:
            y=tuple(x[k]+v[k] for k in range(3)); j=idx.get(y)
            if j is not None: A[i,j]-=1.0
    b=np.zeros(len(nodes)); b[idx[(0,0,0)]]=1.0
    phi,info=cg(A.tocsr(),b,rtol=1e-12,atol=0.0,maxiter=50000)
    if info!=0: raise RuntimeError((R,info))
    return seen,idx,phi

def add_segment(E,c,mu,sgn,f):
    if sgn==1:
        E[LID[(mu,c[0]%2,c[1]%2,c[2]%2)]] += f
        a=list(c); a[mu]+=1; return tuple(a)
    a=list(c); a[mu]-=1; dest=tuple(a)
    E[LID[(mu,dest[0]%2,dest[1]%2,dest[2]%2)]] -= f
    return dest

def embed_edge(E,x,y,f):
    sig=tuple(y[i]-x[i] for i in range(3))
    if not all(abs(s)==1 for s in sig): raise ValueError((x,y))
    c=x
    # frozen position-space order: Sz, Sy, Sx
    for mu in (2,1,0): c=add_segment(E,c,mu,sig[mu],f)
    if c!=y: raise AssertionError((c,y))

def pushforward(R):
    seen,idx,phi=potential(R); nodes=set(seen); E=np.zeros(24); done=set(); bout=0.0
    for x in nodes:
        for v in DIRS:
            y=tuple(x[i]+v[i] for i in range(3))
            if y in nodes:
                key=tuple(sorted((x,y)))
                if key in done: continue
                done.add(key)
                f=float(phi[idx[x]]-phi[idx[y]])
                embed_edge(E,x,y,f)
            else:
                f=float(phi[idx[x]])
                bout += f
                embed_edge(E,x,y,f)
    return E,bout,float(phi[idx[(0,0,0)]]),len(nodes)

def incidence():
    B=np.zeros((8,24))
    for j,(mu,x,y,z) in enumerate(LINKS):
        a=(x,y,z); b=list(a); b[mu]^=1; b=tuple(b)
        B[SID[a],j]+=1; B[SID[b],j]-=1
    return B

def plaquette_matrix():
    ps=[]
    for mu in range(3):
      for nu in range(mu+1,3):
       for s in SITES:
        a=list(s); am=a.copy(); am[mu]^=1; an=a.copy(); an[nu]^=1
        v=np.zeros(24)
        v[LID[(mu,*a)]]+=1; v[LID[(nu,*am)]]+=1
        v[LID[(mu,*an)]]-=1; v[LID[(nu,*a)]]-=1
        ps.append(v)
    return np.array(ps).T

def main():
    B=incidence(); P=plaquette_matrix()
    rb=int(np.linalg.matrix_rank(B,tol=1e-11)); rp=int(np.linalg.matrix_rank(P,tol=1e-11))
    U,s,_=np.linalg.svd(P,full_matrices=False); Qp=U[:,:rp]
    _,_,Vh=np.linalg.svd(B,full_matrices=True); Qc=Vh[rb:].T
    M=(np.eye(24)-Qp@Qp.T)@Qc
    Uh,sh,_=np.linalg.svd(M,full_matrices=False); rh=int(np.sum(sh>1e-10)); Qh=Uh[:,:rh]
    target=np.zeros(8); target[SID[(0,0,0)]]=1; target[SID[(1,1,1)]]=-1
    rows=[]; max_div=0.; max_harm=0.; max_boundary=0.; max_loop_res=0.; prev=None
    radii=[2,4,6,8,10,12,16,20,24,28,32,40]
    for R in radii:
        E,bout,reff,n=pushforward(R); div=B@E
        diverr=float(np.max(np.abs(div-target)))
        hnorm=float(np.linalg.norm(Qh.T@E))
        # subtract a fixed least-norm affine representative; the remainder must be a plaquette boundary
        Eaff=np.linalg.pinv(B,rcond=1e-12)@target
        cyc=E-Eaff; loopres=float(np.linalg.norm(cyc-Qp@(Qp.T@cyc)))
        row={'R':R,'bcc_vertices':n,'unit_resistance':reff,'boundary_flux':bout,
             'core_divergence_residual':diverr,'core_harmonic_norm':hnorm,
             'core_cycle_outside_plaquette_span_norm':loopres,'core_flux_l2_norm':float(np.linalg.norm(E))}
        if prev is not None:
            d=E-prev; row['successive_even_R_difference_divergence_norm']=float(np.linalg.norm(B@d))
            row['successive_even_R_difference_harmonic_norm']=float(np.linalg.norm(Qh.T@d))
            row['successive_even_R_difference_outside_plaquette_span_norm']=float(np.linalg.norm(d-Qp@(Qp.T@d)))
        prev=E; rows.append(row)
        max_div=max(max_div,diverr); max_harm=max(max_harm,hnorm); max_boundary=max(max_boundary,abs(bout-1)); max_loop_res=max(max_loop_res,loopres)
    checks={
      'incidence_rank_is_7':rb==7,
      'plaquette_boundary_rank_is_14':rp==14,
      'harmonic_dimension_is_3':rh==3,
      'all_even_R_fold_to_000_minus_111_pair':max_div<3e-10,
      'all_even_R_have_trivial_torus_homology':max_harm<1e-10,
      'all_even_R_cycles_lie_in_plaquette_span':max_loop_res<3e-10,
      'dirichlet_boundary_flux_is_unit':max_boundary<3e-9,
    }
    out={
      'status':'PASS_C10_SPLITSTEP_CORE_AFFINE_SECTOR_MAP' if all(checks.values()) else 'FAIL',
      'construction':'Solve the unit C10 Dirichlet harmonic current on each BCC ball, compile each BCC edge into the frozen Sz->Sy->Sx split-step path, then quotient every segment modulo L=2. The resulting 24-link real current is analyzed only as a Gauss/homology selector, not as a compact-link quantum amplitude.',
      'core_geometry':{'sites':8,'links':24,'incidence_rank':rb,'plaquette_span_rank':rp,'cycle_dimension':24-rb,'harmonic_dimension':rh},
      'endpoint_sector':'For every even dressing radius tested, div E_core = delta_(000)-delta_(111) and the harmonic torus component vanishes. Multiplication by the integer carrier charge gives the q-sector.',
      'theorem_consequence':'At the level C10 is allowed to fix (Gauss sector and infrared homology), all even-R dressings map to one L=2 affine source-sink sector. Radius dependence is entirely a plaquette/closed-loop coordinate. A compact quantum state must still be selected by the Kato/Riesz rule inside this sector; the real tangent current itself is not an electric-flux basis state.',
      'radii':radii,'rows':rows,'maxima':{'divergence_residual':max_div,'harmonic_norm':max_harm,'cycle_outside_plaquette_span_norm':max_loop_res,'boundary_flux_error':max_boundary},'checks':checks,
      'anti_overclaim':'This certificate constructs the missing boundary-to-core SUPERSELECTION/TOPOLOGY map, not by itself the reduced compact Kato density tensor and not h_8,h_9,h_10.'
    }
    OUT.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
