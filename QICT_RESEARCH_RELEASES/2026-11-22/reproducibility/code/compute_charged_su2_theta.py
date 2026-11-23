#!/usr/bin/env python3
"""Coherent charged weak theta graph with two independent vertex intertwiners."""
from pathlib import Path
from functools import lru_cache
from fractions import Fraction as F
import hashlib,json,math,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from compute_coupled_su2_spin_network import triangle,racah_data,racah2,electric

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/charged_su2_theta'
PRIOR=ROOT/'reproducibility/results/coupled_su2_spin_network'
PHI=math.pi/63


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def intertwiners(a,b,c):
    return [J for J in [c-1,c+1] if triangle(a,b,J) and triangle(c,1,J)]


def labels(radius):
    out=[]
    for a in range(radius+1):
        for b in range(radius+1):
            cs=sorted(set(c for J in range(abs(a-b),a+b+1,2)
                          for c in [J-1,J+1] if c>=0))
            for c in cs:
                jj=intertwiners(a,b,c)
                out.extend((a,b,c,J,K) for J in jj for K in jj)
    return out


@lru_cache(100000)
def element(x,y,which):
    a,b,c,J,K=x;aa,bb,cc,JJ,KK=y
    if which==0:
        if bb!=b:return 0.,F(0)
        factors=[(a,b,J,JJ,1,aa),(c,1,J,JJ,1,cc),
                 (a,b,K,KK,1,aa),(c,1,K,KK,1,cc)]
        sign=1
    else:
        if aa!=a:return 0.,F(0)
        factors=[(b,a,J,JJ,1,bb),(c,1,J,JJ,1,cc),
                 (b,a,K,KK,1,bb),(c,1,K,KK,1,cc)]
        exponent=b+bb-(J+K+JJ+KK)//2
        sign=(-1)**exponent
    changed,changed2=(a,aa) if which==0 else (b,bb)
    squared=F((changed+1)*(changed2+1)*(c+1)*(cc+1)*
              (J+1)*(JJ+1)*(K+1)*(KK+1))
    for t in factors:
        delta,total=racah_data(*t)
        squared*=delta*total*total
        if total<0:sign=-sign
    if not squared:return 0.,squared
    return sign*math.sqrt(float(squared)),squared


def matrix(lab):
    index={x:i for i,x in enumerate(lab)};rows=[];cols=[];data=[]
    row_errors=np.zeros(len(lab));col_errors=np.zeros(len(lab))
    for j,x in enumerate(lab):
        a,b,c,J,K=x
        for which in [0,1]:
            for shifted in [(a-1,b),(a+1,b)] if which==0 else [(a,b-1),(a,b+1)]:
                aa,bb=shifted
                if min(aa,bb)<0:continue
                for cc in [c-1,c+1]:
                    if cc<0:continue
                    jj=intertwiners(aa,bb,cc)
                    for JJ in jj:
                        for KK in jj:
                            y=(aa,bb,cc,JJ,KK)
                            if y not in index:continue
                            value,squared=element(x,y,which)
                            if not value:continue
                            i=index[y];rows.append(i);cols.append(j);data.append(value)
                            # Exact verification of each serialized sqrt
                            # central value, independent of libm accuracy.
                            f=F.from_float(abs(value))
                            err=abs(squared-f*f)/f
                            row_errors[i]+=float(err);col_errors[j]+=float(err)
    M=sparse.csr_matrix((data,(rows,cols)),shape=(len(lab),len(lab)))
    herm=float(sparse.linalg.norm(M-M.T))
    assert herm<3e-12,herm
    E=np.array([electric(x[:3]) for x in lab],np.int64)
    return M,E,herm,max(row_errors.max(),col_errors.max())


def source(lab,neutral_labels,neutral_vector):
    index={x:i for i,x in enumerate(lab)};v=np.zeros(len(lab),complex)
    for (a,b,J),value in zip(neutral_labels,neutral_vector):
        for c in [J-1,J+1]:
            if c<0:continue
            x=(int(a),int(b),int(c),int(J),int(J))
            v[index[x]]+=value*math.sqrt((int(c)+1)/(2*(int(J)+1)))
    assert abs(np.vdot(v,v)-1)<3e-13
    return v


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cpath=PRIOR/'COUPLED_SU2_SPIN_NETWORK_CERTIFICATE.json'
    dpath=PRIOR/'COUPLED_SU2_THETA_STATE_AND_MOMENTS.npz'
    cert=json.loads(cpath.read_text())
    with np.load(dpath,allow_pickle=False) as z:
        neutral_labels=z['R16_labels']
    witness=cert['cutoffs'][-1]
    ground=np.array([float(F(w))*math.sqrt(math.prod(int(t)+1 for t in lab))
        for lab,w in zip(neutral_labels,witness['weighted_rational_witness'])])
    ground/=math.sqrt(float(F(witness['witness_normalization_squared_rational'])))
    records=[];arrays={};previous=None
    for radius in [16,20,24,28,60]:
        start=time.monotonic();lab=labels(radius);M,E,herm,error=matrix(lab)
        v=source(lab,neutral_labels,ground);initial=v.copy();a=[];norm=[]
        pe=np.exp(-1j*PHI*(E%126));generator=1j*PHI*16*M
        for n in range(13):
            a.append(np.vdot(initial,v));norm.append(float(abs(np.vdot(v,v)-1)))
            v=np.exp(-1j*PHI*64)*expm_multiply(generator,pe*v,traceA=0)
        a=np.array(a)
        arrays[f'R{radius}_labels']=np.array(lab,np.int16)
        arrays[f'R{radius}_source']=initial
        arrays[f'R{radius}_charged_returns']=a
        degree=2*(radius+1-16)
        # Chebyshev circuit return: terms of total degree <degree cannot
        # leave this spin box and return. For x<=16/5 and n<=12,
        # the coefficient tail is bounded exactly with e<11/4.
        # The generating-function estimate requires r=5*d/(8*n)>1.
        # Otherwise use the unconditional difference-of-unitaries bound.
        alias=(F(2) if 5*degree<=8*12 else
               min(F(2),F(2)**13*F(22*12,5*degree)**degree))
        state_error=F(witness['ground_projector_trace_norm_error_upper_rational'])
        records.append({'radius':radius,'charged_dimension':len(lab),'nonzero_magnetic_entries':M.nnz,
            'hermiticity_diagnostic':herm,'matrix_entry_rounding_norm_diagnostic':error,
            'source_norm_error':float(abs(np.vdot(initial,initial)-1)),
            'propagated_norm_error':max(norm),
            'successive_return_change':None if previous is None else float(np.max(abs(a-previous))),
            'closed_return_minimum_exit_degree':degree,
            'Chebyshev_propagation_return_bound_rational':str(alias),
            'Chebyshev_propagation_return_bound_upper':min(2.,float(alias)),
            'primitive_state_and_return_bound_rational':str(state_error+alias),
            'seconds':time.monotonic()-start})
        print(json.dumps(records[-1]),flush=True)
        previous=a
        if radius==60:
            arrays['R60_charged_magnetic_csr_data']=M.data
            arrays['R60_charged_magnetic_csr_indices']=M.indices
            arrays['R60_charged_magnetic_csr_indptr']=M.indptr
        # Squared Racah entries can be regenerated; release caches between
        # regulators to bound memory use.
        element.cache_clear()
    # The neutral and charged sectors use the same outer-path spin box.
    from compute_coupled_su2_spin_network import labels as neutral_basis, matrix as neutral_matrix
    neutral_lab=neutral_basis(60)
    _,M0,E0=neutral_matrix(neutral_lab)
    index0={x:i for i,x in enumerate(neutral_lab)}
    seed=np.zeros(len(neutral_lab),complex)
    for lab,value in zip(neutral_labels,ground):seed[index0[tuple(map(int,lab))]]=value
    vv=seed.copy();neutral_returns=[]
    for n in range(13):
        neutral_returns.append(np.vdot(seed,vv))
        vv=np.exp(-1j*PHI*64)*expm_multiply(1j*PHI*16*M0,
              np.exp(-1j*PHI*(E0%126))*vv,traceA=0)
    arrays['R60_neutral_labels']=np.array(neutral_lab,np.int16)
    arrays['R60_neutral_source']=seed
    arrays['R60_neutral_returns']=np.array(neutral_returns)
    arrays['R60_neutral_magnetic_csr_data']=M0.data
    arrays['R60_neutral_magnetic_csr_indices']=M0.indices
    arrays['R60_neutral_magnetic_csr_indptr']=M0.indptr
    weak=arrays['R60_neutral_returns'];charged=arrays['R60_charged_returns']
    arrays['R60_weak_Cn6']=np.array([np.diag([weak[n].conj()*charged[n]]*3+
                    [abs(weak[n])**2]*3) for n in range(13)])
    prob={}
    final=vv
    # Source and intertwiner data remain available without allocating the
    # square density matrix.
    npz=OUT/'CHARGED_SU2_THETA_SOURCES_AND_RETURNS.npz'
    np.savez_compressed(npz,**arrays)
    result={'status':'EXECUTED_COHERENT_CHARGED_SU2_THETA_INTERTWINER_CARRIER',
        'charged_graph':'Three gauge paths and fundamental endpoint matter/dual pair',
        'labels':'a,b,c,J_left,J_right; J_left and J_right are coherent multiplicities',
        'source_embedding':'c=J+-1 with amplitude sqrt((c+1)/(2*(J+1))); J_left=J_right=neutral shared spin J',
        'electric_cost':'3*a*(a+2)+3*b*(b+2)+c*(c+2)',
        'all_charged_vertex_multiplicities_retained':True,
        'cutoffs':records,'ground_state_certificate_sha256':sha(cpath),
        'ground_data_sha256':sha(dpath),'data_sha256':sha(npz),
        'propagation_truncation_certified':True,
        'propagation_certificate':'Closed Chebyshev-path coefficient tail; n<=12 and exact physical multiplication norm <=4.',
        'relative_weak_moment_state_and_alias_bound_rational':str(2*(state_error+alias)),
        'independent_CG_vertex_check_completed':True,
        'independent_CG_vertex_certificate_sha256':sha(ROOT/'reproducibility/results/coupled_su2_verification/INDEPENDENT_COUPLED_SU2_VERTEX_CERTIFICATE.json'),
        'floating_point_interval_certified':False,
        'full_C10_contract_completed':False,
        'source_CAR_scope':'Three fundamental CAR doublets and three singlet modes give nine component creation operators. Six receiver channels retain the dual endpoint index; they are not six generic scalar weak fields.',
        'scope':'Actual two-plaquette static neutral/charged SU2 component with independent vertex and analytic propagation-tail checks; not global C10.'}
    (OUT/'CHARGED_SU2_THETA_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'last_change':records[-1]['successive_return_change']}))


if __name__=='__main__':main()
