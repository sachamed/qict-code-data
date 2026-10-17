import sys, json, csv, time, math
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,'/mnt/data')
import qict_fullblind_blocks as base

def invsqrt_pos(H, tol=1e-14):
    w,U=eigh((H+H.conj().T)/2)
    if np.min(w) <= tol:
        raise RuntimeError(f'block residual rank loss: min eig {np.min(w)}')
    B=(U*np.sqrt(w))@U.conj().T
    Binv=(U*(1/np.sqrt(w)))@U.conj().T
    return B,Binv

def canonical_block_lanczos_diag(xs,V,nblocks=7):
    G=(V.conj().T@V + (V.conj().T@V).conj().T)/2
    _,Ginv=invsqrt_pos(G)
    Q0=V@Ginv
    Qs=[Q0]; As=[]; Bs=[]
    Qprev=None; Bprev=None
    orth_err=[]
    for n in range(nblocks):
        Q=Qs[-1]
        KQ=xs[:,None]*Q
        A=Q.conj().T@KQ; A=(A+A.conj().T)/2
        As.append(A)
        R=KQ-Q@A
        if n>0:
            R-=Qprev@Bprev
        # corrective reorthogonalization against all previous blocks
        for Qo in Qs:
            R-=Qo@(Qo.conj().T@R)
        H=R.conj().T@R
        B,Binv=invsqrt_pos(H)
        Bs.append(B)
        Qnext=R@Binv
        orth_err.append(float(np.linalg.norm(Qnext.conj().T@Qnext-np.eye(3),2)))
        Qprev=Q; Bprev=B; Qs.append(Qnext)
    # block recurrence residuals for first 7 A blocks using B0..B6
    rec=[]
    for n in range(nblocks):
        Q=Qs[n]; lhs=xs[:,None]*Q; rhs=Q@As[n]
        if n>0: rhs += Qs[n-1]@Bs[n-1]
        rhs += Qs[n+1]@Bs[n]
        rec.append(float(np.linalg.norm(lhs-rhs,2)))
    return As,Bs,Qs,orth_err,rec,float(np.linalg.norm(Q0.conj().T@Q0-np.eye(3),2))

def cmplx(M):
    return [[[float(x.real),float(x.imag)] for x in row] for row in M]

def run(N):
    t=time.time(); xs,V,gram,kinfo,udef=base.build_measure(N)
    As,Bs,Qs,oe,rr,q0e=canonical_block_lanczos_diag(xs,V,7)
    return {
      'N':N,'states':len(xs),'seconds':time.time()-t,
      'source_gram_defect_before_normalization':float(np.linalg.norm(gram-np.eye(3),2)),
      'Q0_orthonormality_defect':q0e,
      'next_block_orthonormality_defects':oe,
      'block_recurrence_residuals':rr,
      'kato':kinfo,'max_gauge_block_unitarity_defect':udef,
      'A':[cmplx(A) for A in As], 'B':[cmplx(B) for B in Bs],
      'A_eigenvalues':[np.linalg.eigvalsh(A).tolist() for A in As],
      'B_eigenvalues':[np.linalg.eigvalsh(B).tolist() for B in Bs],
      'B_singular_values':[np.linalg.svd(B,compute_uv=False).tolist() for B in Bs],
    }

def toM(obj):
    return np.array([[complex(*x) for x in row] for row in obj])

def compare(a,b):
    out={'A':[],'B':[]}
    for key in ('A','B'):
      for n,(ma,mb) in enumerate(zip(a[key],b[key])):
        A=toM(ma); B=toM(mb)
        absd=float(np.max(np.abs(B-A)))
        reld=absd/max(1e-300,float(np.max(np.abs(A))),float(np.max(np.abs(B))))
        out[key].append({'shell':n,'max_abs_element_change':absd,'max_relative_element_change':reld,
                         'frobenius_change':float(np.linalg.norm(B-A,'fro'))})
    return out

if __name__=='__main__':
    cases=[]
    for N in (15,17,19):
      print('RUN',N,flush=True); r=run(N); cases.append(r); print('DONE',N,'sec',r['seconds'],'maxrec',max(r['block_recurrence_residuals']),flush=True)
    obj={'blind':True,'external_particle_mass_input':False,'method':'Cayley spectral measure + canonical positive-B block Lanczos','cases':cases,
         'comparisons':{'15_to_17':compare(cases[0],cases[1]),'17_to_19':compare(cases[1],cases[2])}}
    Path('/mnt/data/BLIND_CANONICAL_BLOCKS_15_17_19.json').write_text(json.dumps(obj,indent=2)+'\n')
    rows=[]
    for c in cases:
      for key in ('A','B'):
        for n,Mobj in enumerate(c[key]):
          M=toM(Mobj)
          for i in range(3):
            for j in range(3):
              rows.append({'N':c['N'],'kind':key,'shell':n,'i':i,'j':j,'real':M[i,j].real,'imag':M[i,j].imag})
    with open('/mnt/data/BLIND_CANONICAL_MATRICES_15_17_19.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    srows=[]
    for label,cmp in obj['comparisons'].items():
      for key in ('A','B'):
        for r in cmp[key]: srows.append({'comparison':label,'kind':key,**r})
    with open('/mnt/data/BLIND_CANONICAL_CONVERGENCE_15_17_19.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=srows[0].keys()); w.writeheader(); w.writerows(srows)
    print(json.dumps(obj['comparisons']['17_to_19'],indent=2))
