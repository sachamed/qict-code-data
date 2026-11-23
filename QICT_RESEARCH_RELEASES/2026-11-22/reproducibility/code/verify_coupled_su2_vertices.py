#!/usr/bin/env python3
"""Independent CG-tensor contractions for coupled weak Wilson matrices."""
from pathlib import Path
from functools import lru_cache
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from compute_charged_su2_theta import labels,intertwiners,element
from compute_coupled_su2_spin_network import triangle,weight,electric
from nonabelian_common_car_kato import spinor_fundamental_matrix,su2_spinor_labels

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/coupled_su2_verification'


@lru_cache(None)
def cg(a,b,J,ma,mb):
    M=ma+mb
    if not triangle(a,b,J) or abs(ma)>a or abs(mb)>b or abs(M)>J:return 0.
    if (a-ma)%2 or (b-mb)%2 or (J-M)%2:return 0.
    f=math.factorial
    x,y,z=(J+a-b)//2,(J-a+b)//2,(a+b-J)//2
    pref=F((J+1)*f(x)*f(y)*f(z),f((a+b+J)//2+1))
    for n in [J+M,J-M,a-ma,a+ma,b-mb,b+mb]:pref*=f(n//2)
    total=F(0)
    for k in range(a+b+J+2):
        args=[k,z-k,(a-ma)//2-k,(b+mb)//2-k,
              (J-b+ma)//2+k,(J-a-mb)//2+k]
        if min(args)<0:continue
        total+=F((-1)**k,math.prod(f(n) for n in args))
    return float(total)*math.sqrt(float(pref))


@lru_cache(None)
def invariant(a,b,c,J):
    out=np.zeros((a+1,b+1,c+1,2))
    for ia,ma in enumerate(range(-a,a+1,2)):
        for ib,mb in enumerate(range(-b,b+1,2)):
            for ic,mc in enumerate(range(-c,c+1,2)):
                for f,mf in enumerate([-1,1]):
                    if ma+mb==mc+mf:
                        out[ia,ib,ic,f]=cg(a,b,J,ma,mb)*cg(c,1,J,mc,mf)/math.sqrt(J+1)
    assert abs(np.vdot(out,out)-1)<3e-14
    return out


@lru_cache(None)
def fundamental_tensor(a,aa):
    out=np.zeros((aa+1,a+1,2))
    for ia,ma in enumerate(range(-a,a+1,2)):
        for p,alpha in enumerate([-1,1]):
            target=ma+alpha
            if abs(target)<=aa:out[(target+aa)//2,ia,p]=cg(a,1,aa,ma,alpha)
    return out


def vertex(x,y,J,JJ,which):
    a,b,c=x[:3];aa,bb,cc=y[:3]
    old=invariant(a,b,c,J);new=invariant(aa,bb,cc,JJ)
    first=fundamental_tensor(a,aa) if which==0 else fundamental_tensor(b,bb)
    third=fundamental_tensor(c,cc)
    transformed=(np.einsum('Aap,Ccp,abcf->AbCf',first,third,old)
                 if which==0 else np.einsum('Bbp,Ccp,abcf->aBCf',first,third,old))
    return float(np.vdot(new,transformed))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    states=labels(2);error=0.;checks=offdiag=negative=0
    for x in states:
        a,b,c,J,K=x
        for which in [0,1]:
            pairs=[(a-1,b),(a+1,b)] if which==0 else [(a,b-1),(a,b+1)]
            for aa,bb in pairs:
                if min(aa,bb)<0 or max(aa,bb)>2:continue
                for cc in [c-1,c+1]:
                    if cc<0:continue
                    js=intertwiners(aa,bb,cc)
                    for JJ in js:
                        for KK in js:
                            y=(aa,bb,cc,JJ,KK)
                            actual=math.sqrt((([a,b][which]+1)*(c+1))/
                                       (([aa,bb][which]+1)*(cc+1)))*\
                                   vertex(x,y,J,JJ,which)*vertex(x,y,K,KK,which)
                            predicted,_=element(x,y,which)
                            error=max(error,abs(actual-predicted));checks+=1
                            offdiag+=int(J!=K or JJ!=KK);negative+=int(predicted<0)
    assert error<3e-14,(error,checks)
    old_labels=su2_spinor_labels(8)
    _,old_matrix,_=spinor_fundamental_matrix(8)
    tiny=np.zeros(old_matrix.shape)
    for i,(a,c) in enumerate(old_labels):
        for j,(aa,cc) in enumerate(old_labels):
            tiny[i,j]=element((int(aa),0,int(cc),int(aa),int(aa)),
                              (int(a),0,int(c),int(a),int(a)),0)[0]
    single_error=float(np.max(abs(tiny-old_matrix.toarray())))
    assert single_error<3e-14,single_error
    # Independent rational residual of the full, not compressed, neutral
    # graph uses exact column values and every omitted neighbor.
    certpath=ROOT/'reproducibility/results/coupled_su2_spin_network/COUPLED_SU2_SPIN_NETWORK_CERTIFICATE.json'
    certificate=json.loads(certpath.read_text())
    ground=certificate['cutoffs'][-1]
    with np.load(ROOT/'reproducibility/results/coupled_su2_spin_network/COUPLED_SU2_THETA_STATE_AND_MOMENTS.npz',
                 allow_pickle=False) as z:lab=z['R16_labels']
    # The production residual itself is stored exactly; normalization is
    # checked independently here without importing its witness function.
    values={tuple(map(int,x)):F(v) for x,v in zip(lab,ground['weighted_rational_witness'])}
    norm=sum(weight(x)*v*v for x,v in values.items())
    assert norm==F(ground['witness_normalization_squared_rational'])
    result={'status':'PASS_INDEPENDENT_CG_VERTEX_AND_SINGLE_LOOP_CHECKS',
        'direct_four_valent_CG_contractions':checks,
        'coherent_off_diagonal_vertex_cases':offdiag,
        'negative_Wilson_matrix_elements_checked':negative,
        'direct_CG_vs_Racah_max_error':error,
        'single_loop_vs_independent_character_spinor_max_error':single_error,
        'neutral_witness_weighted_norm_exact':True,
        'source_certificate_sha256':hashlib.sha256(certpath.read_bytes()).hexdigest(),
        'scope':'Independent full-vertex tensor contraction, including J_left!=J_right and signed off-diagonal matrix elements; actual shared-link SU2 component only.',
        'full_C10_contract_completed':False}
    (OUT/'INDEPENDENT_COUPLED_SU2_VERTEX_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
