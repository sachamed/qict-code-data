#!/usr/bin/env python3
"""Coherent theta-graph SU2 on the authentic pair of shared-link plaquettes.

Integer twice-spin labels, exact squared Racah coefficients and weighted
rational Schur inertia. No factorization of the two non-Abelian loops.
"""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import hashlib,itertools,json,math,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh,expm_multiply

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/coupled_su2_spin_network'
PHI=math.pi/63


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def triangle(a,b,c):
    return min(a,b,c)>=0 and abs(a-b)<=c<=a+b and (a+b+c)%2==0


def delta2(a,b,c):
    if not triangle(a,b,c):return F(0)
    x,y,z=(a+b-c)//2,(a-b+c)//2,(-a+b+c)//2
    return F(math.factorial(x)*math.factorial(y)*math.factorial(z),
             math.factorial((a+b+c)//2+1))


@lru_cache(None)
def racah_data(a,b,c,d,e,f):
    product=delta2(a,b,c)*delta2(a,e,f)*delta2(d,b,f)*delta2(d,e,c)
    if not product:return F(0),F(0)
    lower=[(a+b+c)//2,(a+e+f)//2,(d+b+f)//2,(d+e+c)//2]
    upper=[(a+b+d+e)//2,(a+c+d+f)//2,(b+c+e+f)//2]
    total=F(0)
    for z in range(max(lower),min(upper)+1):
        denominator=math.prod(math.factorial(z-t) for t in lower)*\
                    math.prod(math.factorial(t-z) for t in upper)
        total+=F((-1)**z*math.factorial(z+1),denominator)
    return product,total


@lru_cache(None)
def racah2(a,b,c,d,e,f):
    product,total=racah_data(a,b,c,d,e,f)
    return product*total*total


def electric(label):
    a,b,c=label
    return 3*a*(a+2)+3*b*(b+2)+c*(c+2)


def weight(label):
    return math.prod(x+1 for x in label)


def labels(radius):
    return [(a,b,c) for a in range(radius+1) for b in range(radius+1)
            for c in range(abs(a-b),a+b+1,2)]


@lru_cache(None)
def neighbors(label):
    a,b,c=label;out={}
    for aa in [a-1,a+1]:
        for cc in [c-1,c+1]:
            if triangle(aa,b,cc):
                r=racah2(a,b,c,cc,1,aa)
                if r:out[(aa,b,cc)]=F((a+1)*(c+1))*r
    for bb in [b-1,b+1]:
        for cc in [c-1,c+1]:
            if triangle(a,bb,cc):
                r=racah2(b,a,c,cc,1,bb)
                if r:out[(a,bb,cc)]=out.get((a,bb,cc),F(0))+F((b+1)*(c+1))*r
    return tuple(out.items())


def matrix(lab):
    index={x:i for i,x in enumerate(lab)}
    rows=[];cols=[];values=[]
    for j,x in enumerate(lab):
        for y,r in neighbors(x):
            if y in index:
                rows.append(index[y]);cols.append(j)
                values.append(float(r)*math.sqrt(weight(y)/weight(x)))
    M=sparse.csr_matrix((values,(rows,cols)),shape=(len(lab),len(lab)))
    assert sparse.linalg.norm(M-M.T)<3e-13
    E=np.array([electric(x) for x in lab],np.int64)
    H=sparse.diags(E+64,dtype=float)-16*M
    return H.tocsr(),M,E


def rational_inertia(A):
    """Exact congruence elimination for an explicitly symmetric rational form."""
    A=[list(row) for row in A];negative=positive=0
    while A:
        n=len(A);pivot=next((i for i in range(n) if A[i][i]!=0),None)
        if pivot is not None:
            A[0],A[pivot]=A[pivot],A[0]
            for row in A:row[0],row[pivot]=row[pivot],row[0]
            d=A[0][0];positive+=int(d>0);negative+=int(d<0)
            A=[[A[i][j]-A[i][0]*A[0][j]/d for j in range(1,n)]
               for i in range(1,n)]
        else:
            pair=next(((i,j) for i in range(n) for j in range(i+1,n) if A[i][j]),None)
            if pair is None:return positive,negative,n
            i,j=pair;order=[i,j]+[k for k in range(n) if k not in pair]
            A=[[A[i][j] for j in order] for i in order]
            a=A[0][1];positive+=1;negative+=1
            A=[[A[i][j]-(A[i][0]*A[1][j]+A[i][1]*A[0][j])/a
                for j in range(2,n)] for i in range(2,n)]
    return positive,negative,0


def schur_form(threshold,energy_core):
    """QHQ >= QKEQ, with exact weighted boundary-to-core matrix."""
    radius=math.isqrt(energy_core//3)+2
    lab=[x for x in labels(radius) if electric(x)<energy_core]
    index={x:i for i,x in enumerate(lab)};n=len(lab)
    S=[[F(0) for _ in lab] for _ in lab]
    exterior={}
    for j,x in enumerate(lab):
        S[j][j]=F(weight(x)*(electric(x)+64-threshold))
        for y,coefficient in neighbors(x):
            h=-16*coefficient
            if y in index:S[index[y]][j]+=weight(y)*h
            else:exterior.setdefault(y,{})[j]=h
    tail_first=min(electric(x) for x in exterior)
    assert tail_first>=energy_core>threshold
    for y,entries in exterior.items():
        factor=F(weight(y),electric(y)-threshold)
        for i,a in entries.items():
            for j,b in entries.items():S[i][j]-=factor*a*b
    assert all(S[i][j]==S[j][i] for i in range(n) for j in range(n))
    return S,lab,tail_first


def state_witness(lab,v):
    scale=10**18
    w={x:F(round(float(z)/math.sqrt(weight(x))*scale),scale) for x,z in zip(lab,v)}
    norm=sum(weight(x)*z*z for x,z in w.items())
    hw={x:(electric(x)+64)*z for x,z in w.items()}
    for x,z in w.items():
        for y,a in neighbors(x):hw[y]=hw.get(y,F(0))-16*a*z
    mu=sum(weight(x)*z*hw[x] for x,z in w.items())/norm
    residual=sum(weight(x)*(z-mu*w.get(x,F(0)))**2 for x,z in hw.items())/norm
    return w,norm,mu,residual


def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[];arrays={}
    for radius in [6,9,12,16]:
        lab=labels(radius);H,M,E=matrix(lab)
        ev,vec=eigsh(H,k=2,which='SA',tol=1e-13,maxiter=15000)
        order=np.argsort(ev);ev=ev[order];v=vec[:,order[0]]
        if v[0]<0:v=-v
        arrays[f'R{radius}_labels']=np.array(lab,np.int16)
        arrays[f'R{radius}_ground']=v
        w,norm,mu,residual=state_witness(lab,v)
        rec={'radius':radius,'dimension':len(lab),'energy_diagnostic':float(ev[0]),
             'first_excitation_diagnostic':float(ev[1]),'gap_diagnostic':float(ev[1]-ev[0]),
             'rayleigh_rational':str(mu),'infinite_residual_squared_rational':str(residual),
             'witness_normalization_squared_rational':str(norm),
             'weighted_rational_witness':[str(w[x]) for x in lab]}
        records.append(rec)
        print(json.dumps({k:rec[k] for k in ['radius','dimension','energy_diagnostic',
              'first_excitation_diagnostic','gap_diagnostic']}),flush=True)
    # The candidate thresholds are computed controls. Only exact inertia
    # can certify an infinite-representation excited-state lower bound.
    schur_records=[];best=None
    for energy_core in [128,192,256]:
        threshold=math.floor(records[-1]['first_excitation_diagnostic'])-2
        S,core,tail=schur_form(threshold,energy_core)
        start=time.monotonic();inertia=rational_inertia(S)
        rec={'excitation_threshold':threshold,'electric_core_threshold':energy_core,
             'core_dimension':len(core),'boundary_electric_minimum':tail,
             'weighted_Schur_inertia':list(inertia),'seconds':time.monotonic()-start}
        schur_records.append(rec);print(json.dumps(rec),flush=True)
        if inertia[1]==1 and inertia[2]==0:
            best=threshold;break
    assert best is not None,'A stronger core or lower threshold is required.'
    for rec in records:
        mu=F(rec['rayleigh_rational']);r2=F(rec['infinite_residual_squared_rational'])
        assert mu<best
        rec['ground_energy_interval_rational']=[str(mu-r2/(best-mu)),str(mu)]
        rec['gap_lower_rational']=str(best-mu)
        # A rational upper bound for the square root, using integer sqrt.
        scale=10**40;r_upper=F(math.isqrt(r2.numerator*scale*scale//r2.denominator)+1,scale)
        rec['ground_projector_trace_norm_error_upper_rational']=str(2*r_upper/(best-mu))
    npz=OUT/'COUPLED_SU2_THETA_STATE_AND_MOMENTS.npz'
    # Neutral return on the exact same truncated spin-network carrier.
    # Truncation error of these diagnostic returns is not asserted by
    # the state enclosure alone; each output says so explicitly.
    lab=labels(16);H,M,E=matrix(lab);v=arrays['R16_ground'].astype(complex)
    magnetic=1j*PHI*16*M;pe=np.exp(-1j*PHI*(E%126))
    current=v.copy();ret=[];norm_errors=[]
    for n in range(13):
        ret.append(np.vdot(v,current));norm_errors.append(float(abs(np.vdot(current,current)-1)))
        current=np.exp(-1j*PHI*64)*expm_multiply(magnetic,pe*current,traceA=0)
    arrays['R16_neutral_returns']=np.array(ret)
    np.savez_compressed(npz,**arrays)
    payload=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(payload,allow_pickle=False) as z:
        p1=set(map(int,z['plaquette_edge_indices'][2925]))
        p2=set(map(int,z['plaquette_edge_indices'][2908]))
    assert p1&p2=={10377} and len(p1|p2)==7
    result={'status':'EXACT_COUPLED_SU2_THETA_GRAPH_KATO_CERTIFICATE',
        'actual_plaquette_ids':[2925,2908],'shared_link':10377,
        'parallel_path_lengths':[3,3,1],'electric_cost':'3*a*(a+2)+3*b*(b+2)+c*(c+2)',
        'neutral_spin_network_labels':'a,b,c are twice-spins with exact triangle/parity rule',
        'orthonormal_Wilson_matrix_element':'sqrt((a+1)*(a_prime+1)*(c+1)*(c_prime+1))*{a/2,b/2,c/2;c_prime/2,1/2,a_prime/2}^2',
        'rational_weight':'(a+1)*(b+1)*(c+1)',
        'exact_coefficients':'Racah factorial sums with rational squared 6j; amplitude is not a branch probability.',
        'cutoffs':records,'infinite_excitation_lower':best,'Schur_certificates':schur_records,
        'neutral_return_depth':12,'neutral_return_max_norm_diagnostic':max(norm_errors),
        'neutral_return_propagation_truncation_certified':False,
        'state_infinite_representation_certified':True,
        'data_sha256':sha(npz),'payload_sha256':sha(payload),
        'scope':'Authentic coupled two-plaquette neutral SU2 gauge component. Joint intertwiner labels retained. Charged weak source carrier, SU3, matter streams and connected C10 spatial limit not inferred.',
        'full_C10_contract_completed':False,'floating_point_interval_certified':False}
    (OUT/'COUPLED_SU2_SPIN_NETWORK_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'excitation_lower':best,
        'ground_gap_lower':float(F(records[-1]['gap_lower_rational'])),
        'state_trace_error_upper':float(F(records[-1]['ground_projector_trace_norm_error_upper_rational']))}))


if __name__=='__main__':main()
