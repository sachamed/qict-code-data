#!/usr/bin/env python3
"""Independent Weyl-torus FFT evaluation of the authentic gauge/cell module.

Neutral SU2/SU3 class functions are embedded in Weyl-antisymmetric weight
lattices. The charged weak kernel is independently rebuilt from archived
exact integer CG entries. This is not a new complete charged C10 regulator.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from nonabelian_common_car_kato import polynomial_tail,sqrt_upper,su2_source_embedding
from certify_exact_resonance_reduction import normalized_witness

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/weyl_bloch'
PRIOR=ROOT/'reproducibility/results/nonabelian_common_car'
PHI=math.pi/63;DEPTH=12


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def orbit(u,v):
    """Signed A2 Weyl orbit; s1,s2 are exact integral reflections."""
    result={(u,v):1};queue=[(u,v)]
    for a,b in queue:
        for target in [(-a,a+b),(a+b,-b)]:
            sign=-result[(a,b)]
            if target in result:assert result[target]==sign
            else:result[target]=sign;queue.append(target)
    assert len(result)==6
    return result


def class_seed(group,record,length):
    values=normalized_witness(record)
    if group=='U1':
        out=np.zeros(length,complex)
        for (r,),v in zip(record['support_labels'],values):out[r%length]+=v
        support=max(abs(x[0]) for x in record['support_labels'])
    elif group=='SU2':
        out=np.zeros(length,complex)
        for (k,),v in zip(record['support_labels'],values):
            out[(k+1)%length]+=v/math.sqrt(2)
            out[(-k-1)%length]-=v/math.sqrt(2)
        support=max(x[0]+1 for x in record['support_labels'])
    else:
        out=np.zeros((length,length),complex)
        support=0
        for (p,q),v in zip(record['support_labels'],values):
            for (a,b),sign in orbit(p+1,q+1).items():
                out[a%length,b%length]+=sign*v/math.sqrt(6)
                support=max(support,abs(a),abs(b))
    assert abs(np.vdot(out,out)-1)<2e-14
    return out,support


def phases(group,length,charge=0):
    labels=np.arange(length,dtype=np.int64)
    angle=2*np.pi*np.arange(length)/length
    if group=='U1':
        electric=8*labels*labels+4*charge*labels+2*charge*charge+charge
        magnetic=4-4*np.cos(angle)
    elif group=='SU2':
        electric=4*(labels*labels-1)
        magnetic=32-32*np.cos(angle)
    else:
        u=labels[:,None];v=labels[None,:]
        electric=4*(u*u+v*v+u*v-3)
        a=angle[:,None];b=angle[None,:]
        magnetic=36-12*(np.cos(a)+np.cos(b)+np.cos(b-a))
    return np.exp(-1j*PHI*(electric%126)),np.exp(-1j*PHI*magnetic)


def flat_returns(seed,phase_e,phase_b):
    v=seed.copy();returns=[];norm_error=0
    for n in range(DEPTH+1):
        returns.append(np.vdot(seed,v))
        norm_error=max(norm_error,float(abs(np.vdot(v,v)-1)))
        v=np.fft.ifftn(phase_b*np.fft.fftn(phase_e*v))
    return np.asarray(returns),norm_error,v


def anti_defect(group,v):
    length=v.shape[0]
    if group=='SU2':
        return float(np.linalg.norm(v+v[(-np.arange(length))%length]))
    u,vv=np.indices(v.shape)
    a=v[(-u)%length,(u+vv)%length]
    b=v[(u+vv)%length,(-vv)%length]
    return float(max(np.linalg.norm(v+a),np.linalg.norm(v+b)))


def charged_weak(record,data):
    labels=data['L3_SU2_spinor_labels'];edges=data['L3_SU2_magnetic_symbolic_edges']
    matrix=sparse.csr_matrix((edges[:,2]/(2*np.sqrt(edges[:,3].astype(float))),
         (edges[:,0],edges[:,1])),shape=(len(labels),len(labels)))
    assert sparse.linalg.norm(matrix-matrix.T)<1e-13
    assert np.max(abs(labels[edges[:,0],0]-labels[edges[:,1],0]))==1
    maximum=int(labels[:,0].max())
    values=normalized_witness(record)
    scalar=np.zeros(maximum+1);scalar[:len(values)]=values
    seed=su2_source_embedding(scalar,maximum);v=seed.astype(complex)
    cost=3*labels[:,0]*(labels[:,0]+2)+labels[:,1]*(labels[:,1]+2)
    electric=np.exp(-1j*PHI*(cost%126))
    A=1j*PHI*16*matrix;constant=np.exp(-1j*PHI*32)
    result=[];error=0
    for n in range(DEPTH+1):
        result.append(np.vdot(seed,v));error=max(error,float(abs(np.vdot(v,v)-1)))
        v=constant*expm_multiply(A,electric*v,traceA=0)
    return np.asarray(result),error


def fusion_checks():
    """Exact signed-weight compression of fundamental multiplication."""
    tests=[]
    for p in range(4):
        for q in range(4-p):
            basis=orbit(p+1,q+1);actual={}
            for (u,v),sign in basis.items():
                for du,dv in [(1,0),(-1,1),(0,-1)]:
                    target=(u+du,v+dv)
                    actual[target]=actual.get(target,0)+sign
            actual={k:v for k,v in actual.items() if v}
            expected={}
            for a,b in [(p+1,q),(p-1,q+1),(p,q-1)]:
                if min(a,b)<0:continue
                for uv,sign in orbit(a+1,b+1).items():
                    expected[uv]=expected.get(uv,0)+sign
            expected={k:v for k,v in expected.items() if v}
            assert actual==expected,(p,q)
            tests.append((p,q))
    return tests


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cert_path=PRIOR/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json'
    data_path=PRIOR/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz'
    prior=json.loads(cert_path.read_text())
    grounds=prior['cutoff_results'][-1]['ground_certificates']
    with np.load(data_path,allow_pickle=False) as z:data={k:z[k] for k in z.files if k.startswith('L3_')}
    weak_return,weak_norm=charged_weak(grounds['SU2'],data)
    assert np.max(abs(weak_return-data['L3_charged_group_returns'][2]))<2e-13
    state_bound=sum(sqrt_upper(F(2))*F(g['state_trace_distance_upper_rational'])
                    for g in grounds.values())
    weak_tail=polynomial_tail(F(8,5),23)
    arrays={};records=[];previous=None
    for samples in [1,2,4,8]:
        length=63*samples;returns={};errors={};bounds={}
        for group,x in [('U1',F(1,5)),('SU2',F(8,5)),('SU3',F(9,5))]:
            seed,support=class_seed(group,grounds[group],length)
            electric,magnetic=phases(group,length)
            a,norm,v=flat_returns(seed,electric,magnetic)
            returns[group]=a
            errors[group]={'norm_error':norm,'dimension':int(seed.size)}
            if group!='U1':errors[group]['Weyl_antisymmetry_defect']=anti_defect(group,v)
            order=(length-2*support-1)//DEPTH
            tail=polynomial_tail(x,order)
            bounds[group]={'polynomial_order':order,'state_coordinate_support':support,
               'one_return_alias_bound_rational':str(2*DEPTH*tail*(1+tail)**(DEPTH-1))}
        yseed,_=class_seed('U1',grounds['U1'],length)
        charged={}
        for q in [-3,-6]:
            pe,pb=phases('U1',length,q);a,norm,_=flat_returns(yseed,pe,pb)
            charged[q]=a;errors['Y'+str(q)]={'norm_error':norm}
        left=returns['U1'].conj()*charged[-3]*returns['SU2'].conj()*weak_return*abs(returns['SU3'])**2
        right=returns['U1'].conj()*charged[-6]*abs(returns['SU2'])**2*abs(returns['SU3'])**2
        c=np.array([np.diag([left[n]]*3+[right[n]]*3) for n in range(DEPTH+1)])
        cell=c.copy();cell[1::2]=0
        old_difference=float(np.max(abs(c-data['L3_gauge_Cn6'])))
        assert old_difference<3e-12
        error=4*state_bound+2*F(bounds['U1']['one_return_alias_bound_rational'])+\
              2*F(bounds['SU2']['one_return_alias_bound_rational'])+\
              2*F(bounds['SU3']['one_return_alias_bound_rational'])+\
              2*DEPTH*weak_tail*(1+weak_tail)**(DEPTH-1)
        neutral_difference=float(np.max(abs(np.array(list(returns.values()))-data['L3_neutral_group_returns'])))
        toeplitz=np.block([[c[j-i] if j>=i else c[i-j].conj().T
                           for j in range(DEPTH+1)] for i in range(DEPTH+1)])
        minimum=float(np.linalg.eigvalsh(toeplitz).min())
        assert minimum>-1e-11 and np.linalg.norm(c[0]-np.eye(6))<1e-12
        records.append({'Bloch_quadrature_samples_per_torus_axis':samples,
          'equivalent_cyclic_weight_period':length,
          'analytic_state_alias_and_weak_padding_bound_rational':str(error),
          'analytic_state_alias_and_weak_padding_bound_upper':min(2.,float(error)),
          'uncapped_analytic_bound_upper':float(error),
          'independent_previous_moment_difference':old_difference,
          'independent_previous_neutral_return_difference':neutral_difference,
          'moment_successive_change':None if previous is None else float(np.max(abs(c-previous))),
          'Toeplitz_minimum_diagnostic':minimum,'C0_Frobenius_error':float(np.linalg.norm(c[0]-np.eye(6))),
          'alias_bounds_by_group':bounds,'floating_point_diagnostics':errors})
        arrays[f'M{samples}_Cn6']=c;arrays[f'M{samples}_gauge_cell_Cn6']=cell
        arrays[f'M{samples}_neutral_returns']=np.array(list(returns.values()))
        arrays[f'M{samples}_charged_Y_returns']=np.array(list(charged.values()))
        previous=c
        print(json.dumps({'M':samples,'bound':min(2.,float(error)),
                          'independent_moment_difference':old_difference}),flush=True)
    arrays['charged_weak_returns']=weak_return
    npz=OUT/'WEYL_BLOCH_THREE_GROUP_CAR_MOMENTS.npz'
    np.savez_compressed(npz,**arrays)
    result={'status':'EXECUTED_EXACT_WEYL_TORUS_REPRESENTATION_AND_CHARGED_CAR_CROSSCHECK',
      'neutral_maps':{'SU2':'|k> -> (|k+1>-|-k-1>)/sqrt(2)',
                     'SU3':'|p,q> -> sum_w det(w)|w(p+1,q+1)>/sqrt(6)'},
      'neutral_electric_torus_polynomials':{'SU2':'4*(m^2-1)','SU3':'4*(u^2+v^2+u*v-3)'},
      'neutral_magnetic_torus_multipliers':{'SU2':'32-32*cos(theta)',
          'SU3':'36-12*(cos(theta1)+cos(theta2)+cos(theta2-theta1))'},
      'exact_A2_fundamental_fusion_tests':fusion_checks(),
      'exact_one_cell_electric_period':63,
      'charged_weak_method':'Exact archived integer CG edges independently reconstructed as sparse matrix; both fusion branches retained. No charged weak periodicity is presumed.',
      'charged_weak_norm_error':weak_norm,'cutoffs':records,
      'original_primitive_certificate_sha256':sha(cert_path),
      'original_common_CAR_data_sha256':sha(data_path),
      'scope':'Independent infinite-representation computation of the authentic one-plaquette gauge/cell module, combining exact neutral Weyl transforms and certified charged weak buffers. Not a single new finite physical C10 regulator or a connected spatial limit.',
      'floating_point_interval_certified':False,'full_C10_contract_completed':False,
      'data_sha256':sha(npz)}
    (OUT/'WEYL_BLOCH_COMMON_CAR_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'final_moment_difference':records[-1]['independent_previous_moment_difference'],
      'final_analytic_bound':records[-1]['analytic_state_alias_and_weak_padding_bound_upper'],
      'full_C10':False},indent=2))


if __name__=='__main__':main()
