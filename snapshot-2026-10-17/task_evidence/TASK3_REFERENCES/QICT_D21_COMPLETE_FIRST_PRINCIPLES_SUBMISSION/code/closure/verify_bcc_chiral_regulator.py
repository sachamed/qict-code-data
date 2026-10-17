#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; DATA=ROOT/'data'
P=math.pi
SIGMAS=list(itertools.product((-1,1), repeat=3))
# BCC direct primitive vectors in the integer-coordinate convention used by C7.
A=np.array([[1,1,-1],[1,-1,1],[-1,1,1]],dtype=float).T
B=2*P*np.linalg.inv(A).T  # reciprocal primitive vectors


def c7(k):
    x,y,z=map(float,k)
    cx,cy,cz=np.cos([x,y,z]); sx,sy,sz=np.sin([x,y,z])
    d=float(cx*cy*cz-sx*sy*sz)
    n=np.array([sx*cy*cz+cx*sy*sz,
                cx*sy*cz-sx*cy*sz,
                cx*cy*sz+sx*sy*cz],float)
    return d,n

def jac(k,h=1e-6):
    k=np.asarray(k,float); J=np.empty((3,3))
    for j in range(3):
        e=np.zeros(3);e[j]=h
        J[:,j]=(c7(k+e)[1]-c7(k-e)[1])/(2*h)
    return J

def Wbcc(k):
    k=np.asarray(k,float)
    return float(1.0-np.prod(np.cos(k)))

def Wold(k):
    k=np.asarray(k,float)
    return float(np.sum(1.0-np.cos(k)))

def equiv(a,b):
    # exact search in the small reciprocal lattice needed for the certified node list
    d=np.asarray(a)-np.asarray(b)
    for m in itertools.product(range(-3,4),repeat=3):
        if np.linalg.norm(d-B@np.asarray(m,float))<1e-10:
            return True
    return False

def node_list():
    out=[]
    for x in (-P,0.0):
      for y in (-P,0.0):
       for z in (-P,0.0): out.append(np.array((x,y,z),float))
    for x in (-P/2,P/2):
      for y in (-P/2,P/2):
       for z in (-P/2,P/2): out.append(np.array((x,y,z),float))
    return out

def classes(nodes):
    cs=[]
    for i,k in enumerate(nodes):
        for c in cs:
            if equiv(k,nodes[c[0]]): c.append(i);break
        else: cs.append([i])
    return cs

def site_index(x,L): return (x[0]*L+x[1])*L+x[2]

def build_links(L,rng):
    sites=list(itertools.product(range(L),repeat=3))
    pos=[s for s in SIGMAS if s[0]==1]
    U={}
    for x in sites:
      for s in pos:
        y=tuple((x[i]+s[i])%L for i in range(3))
        # assign each oriented positive edge once; reverse is exact adjoint
        if (x,s) not in U:
            z=np.exp(1j*rng.uniform(-P,P))
            U[(x,s)]=z
            sm=tuple(-v for v in s)
            U[(y,sm)]=np.conj(z)
    # Odd L can cause a pair to be reached from another representative; enforce exact reciprocity.
    for x in sites:
      for s in SIGMAS:
        y=tuple((x[i]+s[i])%L for i in range(3)); sm=tuple(-v for v in s)
        if (x,s) not in U:
            # deterministic fallback respecting the reverse already present
            if (y,sm) in U: U[(x,s)]=np.conj(U[(y,sm)])
            else:
                z=np.exp(1j*rng.uniform(-P,P)); U[(x,s)]=z;U[(y,sm)]=np.conj(z)
    return U

def build_WU(L,U):
    Ns=L**3; S=np.zeros((Ns,Ns),complex)
    for x in itertools.product(range(L),repeat=3):
      src=site_index(x,L)
      for s in SIGMAS:
        y=tuple((x[i]+s[i])%L for i in range(3));dst=site_index(y,L)
        S[dst,src]+=U[(x,s)]/8.0
    return np.eye(Ns,dtype=complex)-S

def gauge_transform_links(L,U,lam):
    Up={}
    for (x,s),u in U.items():
        y=tuple((x[i]+s[i])%L for i in range(3))
        Up[(x,s)]=lam[y]*u*np.conj(lam[x])
    return Up

def main():
    # reciprocal basis cross-check against the exact known basis
    Btarget=np.array([[P,P,0],[P,0,P],[0,P,P]],float).T
    reciprocal_res=float(np.linalg.norm(B-Btarget))

    # reciprocal periodicity: new regulator passes, old regulator fails.
    rng=np.random.default_rng(20260914)
    new_res=0.0; old_res=0.0
    for _ in range(128):
        k=rng.uniform(-P,P,3); m=rng.integers(-3,4,3); G=B@m
        new_res=max(new_res,abs(Wbcc(k+G)-Wbcc(k)))
        old_res=max(old_res,abs(Wold(k+G)-Wold(k)))

    nodes=node_list(); cs=classes(nodes)
    rows=[]; class_rows=[]
    for ci,c in enumerate(cs):
        vals=[]
        for idx in c:
            k=nodes[idx]; d,n=c7(k); w=Wbcc(k); wo=Wold(k)
            det=float(np.linalg.det(jac(k)))
            vals.append(w)
            rows.append({'index':idx,'class':ci,'kx_over_pi':float(k[0]/P),'ky_over_pi':float(k[1]/P),'kz_over_pi':float(k[2]/P),
                         'quasienergy':'0' if d>0 else 'pi','chirality':int(np.sign(det)),'W_bcc':w,'W_old_axis':wo})
        rep=rows[-len(c)]
        class_rows.append({'class':ci,'multiplicity_in_cubic_cover':len(c),'quasienergy':rep['quasienergy'],'chirality':rep['chirality'],
                           'W_bcc':float(np.mean(vals)),'W_bcc_spread':float(max(vals)-min(vals)),
                           'old_W_spread':float(max(rows[j]['W_old_axis'] for j in range(len(rows)-len(c),len(rows)))-min(rows[j]['W_old_axis'] for j in range(len(rows)-len(c),len(rows))))})
    # sort physical class first then by quasienergy/W
    # Gamma class is the one containing node index 7 in the historic ordering.
    gamma_class=next(i for i,c in enumerate(cs) if 7 in c)
    nongamma=[r['W_bcc'] for r in class_rows if r['class']!=gamma_class]
    minW=float(min(nongamma))
    ratio=minW/2.0
    selected=[]
    for r in class_rows:
        w=r['W_bcc']; mL=-ratio+w; mR=ratio+w
        r['m_left_over_rW']=float(mL);r['m_right_over_rW']=float(mR);r['wall_sign_change']=bool(mL*mR<0)
        if r['wall_sign_change']: selected.append(r['class'])

    # Gauge-covariant range-one BCC Wilson operator audit.
    L=3; U=build_links(L,rng); WU=build_WU(L,U)
    herm=float(np.linalg.norm(WU-WU.conj().T,2))
    evals=np.linalg.eigvalsh((WU+WU.conj().T)/2)
    pos_min=float(evals.min()); pos_max=float(evals.max())
    lam={x:np.exp(1j*rng.uniform(-P,P)) for x in itertools.product(range(L),repeat=3)}
    Up=gauge_transform_links(L,U,lam); WUp=build_WU(L,Up)
    G=np.diag([lam[x] for x in itertools.product(range(L),repeat=3)])
    cov=float(np.linalg.norm(WUp-G@WU@G.conj().T,2))

    # Node-level interaction stability: the minimax margin is rW/2. By Weyl's inequality,
    # any Hermitian self-energy perturbation with norm < margin cannot close a nonphysical node gap.
    margin=ratio
    # Numerical stress test only; theorem is analytic and stated in certificate.
    stress_min=1e9
    for _ in range(200):
        X=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); H=(X+X.conj().T)/2
        H/=np.linalg.norm(H,2); V=0.49*margin*H
        H0=np.diag([-margin,-margin,margin,margin])
        stress_min=min(stress_min,float(np.min(np.abs(np.linalg.eigvalsh(H0+V)))))

    status=(reciprocal_res<1e-12 and new_res<1e-12 and old_res>1e-3 and len(cs)==4
            and max(r['W_bcc_spread'] for r in class_rows)<1e-12 and abs(minW-1.0)<1e-12
            and abs(ratio-0.5)<1e-12 and selected==[gamma_class]
            and herm<1e-12 and cov<1e-12 and pos_min>-1e-12 and pos_max<2+1e-12 and stress_min>0)

    import csv
    with (DATA/'bcc_c7_node_classes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (DATA/'bcc_c7_reciprocal_classes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)

    cert={
      'status':'PASS' if status else 'FAIL',
      'logical_status':'DERIVED',
      'evidence_type':'analytic identity + numerical verification',
      'direct_bcc_primitive_vectors':A.tolist(),
      'reciprocal_bcc_primitive_vectors':B.tolist(),
      'reciprocal_basis_residual':reciprocal_res,
      'canonical_range_one_scalar':'W_BCC(k)=1-cos(kx)cos(ky)cos(kz)=1-(1/8) sum_{sigma in {+-1}^3} exp(i sigma.k)',
      'uniqueness_scope':'unique up to positive overall scale among onsite + one-hop BCC scalar Hermitian operators invariant under the full cubic orbit and normalized to vanish at Gamma',
      'new_regulator_reciprocal_periodicity_residual':new_res,
      'axis_regulator_reciprocal_periodicity_residual':old_res,
      'cubic_cover_zero_count':len(nodes),
      'physical_bcc_reciprocal_class_count':len(cs),
      'class_rows':class_rows,
      'gamma_class':gamma_class,
      'min_W_on_nonGamma_classes':minW,
      'topological_window':'0 < M0/rW < 1',
      'unique_minimax_M0_over_rW':ratio,
      'selected_wall_classes':selected,
      'gauge_covariant_operator':'W_BCC[U]=I-(1/8)sum_sigma T_sigma[U]',
      'gauge_operator_hermiticity_defect':herm,
      'gauge_operator_spectrum_min':pos_min,
      'gauge_operator_spectrum_max':pos_max,
      'gauge_covariance_defect':cov,
      'node_gap_margin_over_rW':margin,
      'interaction_stability_theorem':'For every nonphysical C7 node, the free Hermitian regulator gap is at least rW/2. If a Hermitian proper self-energy perturbation restricted to that node has operator norm < rW/2, Weyl eigenvalue inequalities forbid a zero crossing. At Gamma the two asymptotic wall masses retain opposite signs under mass-channel perturbations of norm < rW/2.',
      'random_stress_test_min_abs_eigenvalue':stress_min,
      'external_particle_data_used':False,
      'axis_regulator_status':'EXCLUDED_BY_BCC_RECIPROCAL_PERIODICITY'
    }
    (CERT/'bcc_chiral_regulator_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps(cert,indent=2))
    raise SystemExit(0 if status else 1)

if __name__=='__main__': main()
