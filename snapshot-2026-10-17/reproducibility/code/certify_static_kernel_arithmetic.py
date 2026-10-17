#!/usr/bin/env python3
"""Integer fixed-point enclosures for all 64 static physical source kernels.

No floating-point operation enters the enclosed propagation. Floating-point
arrays are read only to bound their differences from the enclosed kernels.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/static_wilson_car'
SCALE=10**40
ERROR_SCALE=10**35
ORDER=40


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def upper(x,denominator=ERROR_SCALE):
    n=x*denominator
    return F((n.numerator+n.denominator-1)//n.denominator,denominator)


def lower(x,denominator=10**16):
    n=x*denominator
    return F(n.numerator//n.denominator,denominator)


def nearest(n):
    return (n+SCALE//2)//SCALE if n>=0 else -((-n+SCALE//2)//SCALE)


def quantize_interval(lo,hi):
    q=round((lo+hi)*SCALE/2)
    return q,max(abs(F(q,SCALE)-lo),abs(hi-F(q,SCALE)))


def sqrt_interval(x):
    n=math.isqrt(x.numerator*SCALE**2//x.denominator)
    lo=F(n,SCALE)
    hi=lo if lo*lo==x else F(n+1,SCALE)
    return lo,hi


def pi_bounds():
    def atan(x):
        a=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(30)),F(0))
        b=a+x**61/F(61)
        return a,b
    a,b=atan(F(1,5));c,d=atan(F(1,239))
    return lower(16*a-4*d,10**45),upper(16*b-4*c,10**45)


def phases(pi):
    result={};maximum=F(0)
    remainder=F(22,7)**63/math.factorial(63)
    for residue in range(-63,63):
        x=F(residue,126)*(pi[0]+pi[1])
        argument_error=F(abs(residue),126)*(pi[1]-pi[0])
        cos=sum(((-1)**k*x**(2*k)/math.factorial(2*k) for k in range(32)),F(0))
        sin=sum(((-1)**k*x**(2*k+1)/math.factorial(2*k+1) for k in range(32)),F(0))
        error=argument_error+remainder
        real,er=quantize_interval(cos-error,cos+error)
        imag,ei=quantize_interval(sin-error,sin+error)
        result[residue]=(real,imag);maximum=max(maximum,er+ei)
    return result,upper(maximum)


def normalized_seed(values,dimension,start=0):
    values=[F(v) for v in values];normalization=sum(v*v for v in values)
    a,b=sqrt_interval(1/normalization)
    result=[(0,0) for _ in range(dimension)];error=F(0)
    for i,v in enumerate(values):
        lo,hi=sorted((v*a,v*b));q,e=quantize_interval(lo,hi)
        result[start+i]=(q,0);error+=e
    return result,upper(error)


def phase_multiply(vector,costs,table,inverse):
    result=[]
    for (a,b),cost in zip(vector,costs):
        residue=(cost+63)%126-63
        # The forward electric multiplier is exp(-i*pi*cost/63).
        c,d=table[residue]
        if not inverse:d=-d
        result.append((nearest(a*c-b*d),nearest(a*d+b*c)))
    return result


class FixedUpdate:
    def __init__(self,rows,costs,norm_bound,coefficient_error,phase_table,phase_error):
        self.rows=rows;self.costs=costs;self.dimension=len(rows);self.table=phase_table
        self.coefficient_error=upper(coefficient_error)
        x=norm_bound+self.coefficient_error
        assert x<F(2)
        tail=x**(ORDER+1)/math.factorial(ORDER+1)/(1-x/F(ORDER+2))
        # Taylor term rounding: <= 2*d/S per term, propagated with exp(x)<9.
        error=2*self.coefficient_error+2*tail+F(720*self.dimension,SCALE)
        error+=2*phase_error+F(self.dimension,SCALE)
        self.local_error=upper(error)

    def magnetic(self,vector,sign):
        term=vector;total=list(vector)
        for k in range(1,ORDER+1):
            new=[]
            for row in self.rows:
                real=imag=0
                for j,a in row:
                    r,i=term[j];real-=sign*a*i;imag+=sign*a*r
                real=nearest(real);imag=nearest(imag)
                real=(real+k//2)//k if real>=0 else -((-real+k//2)//k)
                imag=(imag+k//2)//k if imag>=0 else -((-imag+k//2)//k)
                new.append((real,imag))
            term=new
            total=[(a+c,b+d) for (a,b),(c,d) in zip(total,term)]
        return total

    def apply(self,state,inverse=False):
        vector,error=state
        assert error+self.local_error<1
        if inverse:
            result=phase_multiply(self.magnetic(vector,-1),self.costs,self.table,True)
        else:
            result=self.magnetic(phase_multiply(vector,self.costs,self.table,False),1)
        return result,error+self.local_error


def weak_update(z,h,pi,phase_table,phase_error):
    labels=[tuple(map(int,x)) for x in z[f'weak_J{h}_labels']]
    rows=[[] for _ in labels];errors=F(0);entries={}
    for i,j,coefficient,radicand in z[f'weak_J{h}_symbolic_edges']:
        i,j=int(i),int(j);coefficient=F(str(coefficient));a,b=sqrt_interval(F(str(radicand)))
        endpoints=[16*p*coefficient*x/63 for p in pi for x in (a,b)]
        q,e=quantize_interval(min(endpoints),max(endpoints))
        rows[i].append((j,q));entries[i,j]=q;errors+=e
    assert all(entries.get((j,i))==v for (i,j),v in entries.items())
    costs=[3*s*(s+2)+t*(t+2)+32 for s,t in labels]
    update=FixedUpdate(rows,costs,F(8,5),errors,phase_table,phase_error)
    source=[];source_error=F(0)
    for s,t in labels:
        a,b=sqrt_interval(F(t+1,(h+1)*(s+1)))
        q,e=quantize_interval(a,b);source.append((s,q));source_error+=e
    return update,source,upper(source_error)


def y_update(Q,pi,phase_table,phase_error):
    a,e=quantize_interval(2*pi[0]/63,2*pi[1]/63)
    rows=[[((i-1)%504,a),((i+1)%504,a)] for i in range(504)]
    costs=[8*r*r+4*Q*r+2*Q*Q+Q+4 for r in range(-252,252)]
    return FixedUpdate(rows,costs,F(1,5),1008*e,phase_table,phase_error)


def source_apply(state,source,error):
    vector,radius=state;result=[]
    assert radius+2*error+F(len(source),SCALE)<1
    for i,a in source:
        r,b=vector[i];result.append((nearest(a*r),nearest(a*b)))
    return result,radius+2*error+F(len(source),SCALE)


def field_orbits(neutral,update,source=None,source_error=F(0)):
    states=[]
    for t,neutral_state in enumerate(neutral):
        state=neutral_state if source is None else source_apply(neutral_state,source,source_error)
        for _ in range(t):state=update.apply(state,True)
        states.append(state)
    return states


def gram(states):
    result=[]
    for x,ex in states:
        row=[]
        for y,ey in states:
            real=sum(a*c+b*d for (a,b),(c,d) in zip(x,y))
            imag=sum(a*d-b*c for (a,b),(c,d) in zip(x,y))
            row.append((F(real,SCALE**2),F(imag,SCALE**2),upper(ex+ey+ex*ey)))
        result.append(row)
    return result


def main():
    producer_path=OUT/'STATIC_WILSON_CAR_CERTIFICATE.json'
    producer=json.loads(producer_path.read_text())
    data=OUT/'STATIC_WILSON_CAR_MODULES_AND_KERNELS.npz'
    assert sha(data)==producer['data_sha256']
    primitive_path=ROOT/'reproducibility/results/nonabelian_common_car/NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json'
    primitive=json.loads(primitive_path.read_text());ground=primitive['cutoff_results'][-1]['ground_certificates']
    pi=pi_bounds();phase_table,phase_error=phases(pi)
    yvalues=ground['U1']['rational_witness_components'];ry=(len(yvalues)-1)//2
    Y0=y_update(0,pi,phase_table,phase_error)
    yneutral=[normalized_seed(yvalues,504,252-ry)]
    for _ in range(6):yneutral.append(Y0.apply(yneutral[-1]))
    Y={Q:gram(field_orbits(yneutral,y_update(Q,pi,phase_table,phase_error))) for Q in range(-36,1,3)}
    W={};operator_bounds={}
    with np.load(data,allow_pickle=False) as z:
        W0,_,_=weak_update(z,0,pi,phase_table,phase_error)
        core=producer['diagnostics']['weak_seed_core_radius']
        weak_values=ground['SU2']['rational_witness_components'][:core+1]
        wneutral=[normalized_seed(weak_values,61)]
        for _ in range(6):wneutral.append(W0.apply(wneutral[-1]))
        for h in range(4):
            U,source,error=weak_update(z,h,pi,phase_table,phase_error)
            W[h]=gram(field_orbits(wneutral,U,source,error))
            operator_bounds[str(h)]={'dimension':U.dimension,'generator_coefficient_norm_error_upper_rational':str(U.coefficient_error),
                'one_update_arithmetic_error_upper_rational':str(U.local_error),'source_coefficient_norm_error_upper_rational':str(error)}
        weights={0:{0:F(1)},1:{1:F(1)},2:{0:F(1,4),2:F(3,4)},3:{1:F(1,2),3:F(1,2)}}
        exact_centres=[];records=[];maximum_radius=maximum_reported=F(0)
        truncation=F(producer['four_representation_boxes'][-1]['physical_kernel_error_bound_rational'])
        denominator=4*SCALE**4
        for nl in range(4):
            for nr in range(4):
                Q=-3*nl-6*nr;block=[];centres=[]
                for t in range(7):
                    row=[];centre_row=[]
                    for s in range(7):
                        a,b,e=Y[Q][t][s]
                        c=sum(p*W[h][t][s][0] for h,p in weights[nl].items())
                        d=sum(p*W[h][t][s][1] for h,p in weights[nl].items())
                        f=sum(p*W[h][t][s][2] for h,p in weights[nl].items())
                        real,imag=a*c-b*d,a*d+b*c
                        radius=upper(e+f+e*f);maximum_radius=max(maximum_radius,radius)
                        re,im=real*denominator,imag*denominator
                        assert re.denominator==im.denominator==1
                        centre_row.append([str(re.numerator),str(im.numerator)])
                        total=radius+truncation
                        row.append([str(lower(real-total)),str(upper(real+total,10**16)),
                                    str(lower(imag-total)),str(upper(imag+total,10**16))])
                        for mask in range(64):
                            if (mask&7).bit_count()!=nl or (mask>>3).bit_count()!=nr:continue
                            value=z['R60_physical_kernels'][mask,t,s]
                            diff=abs(F(float(value.real))-real)+abs(F(float(value.imag))-imag)+2*total
                            maximum_reported=max(maximum_reported,diff)
                    block.append(row);centres.append(centre_row)
                records.append({'left_occupations':nl,'right_occupations':nr,'physical_kernel_real_imag_intervals_rational':block})
                exact_centres.append(centres)
                print('enclosed occupation type',nl,nr,flush=True)
    output_data=OUT/'STATIC_PHYSICAL_KERNEL_INTERVAL_CENTRES.npz'
    np.savez_compressed(output_data,gaussian_integer_centres=np.array(exact_centres,dtype=str),
        common_denominator=np.array(str(denominator)),type_order=np.array([(a,b) for a in range(4) for b in range(4)]))
    certificate={'status':'PASS_INTEGER_ARITHMETIC_STATIC_KERNEL_ENCLOSURES',
       'source_program_sha256':sha(Path(__file__)),'producer_certificate_sha256':sha(producer_path),
       'producer_data_sha256':sha(data),'interval_centre_data_sha256':sha(output_data),
       'fixed_point_scale':str(SCALE),'Taylor_order':ORDER,'pi_interval_rational':[str(x) for x in pi],
       'phase_coefficient_error_upper_rational':str(phase_error),'weak_operator_bounds':operator_bounds,
       'physical_time_range':[0,6],'occupation_sectors':64,'distinct_occupation_types':16,
       'maximum_finite_kernel_arithmetic_radius_rational':str(upper(maximum_radius)),
       'analytic_state_and_representation_bound_rational':str(truncation),
       'maximum_reported_entry_error_upper_rational':str(upper(maximum_reported,10**16)),
       'maximum_reported_7_by_7_block_operator_error_upper_rational':str(upper(7*maximum_reported,10**16)),
       'reported_roundoff_and_analytic_truncation_combined':True,
       'kernel_intervals_include_exact_Haar_static_primitive_state':True,
       'interval_scope':'All 64 static one-plaquette physical two-time kernels; no comparison-resolvent or moving C10 interval inference.',
       'floating_point_used_in_enclosed_propagation':False,'full_C10_closure_inferred':False,
       'records':records}
    (OUT/'STATIC_PHYSICAL_KERNEL_ARITHMETIC_CERTIFICATE.json').write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps({k:v for k,v in certificate.items() if k in ('status','maximum_finite_kernel_arithmetic_radius_rational',
        'maximum_reported_entry_error_upper_rational','maximum_reported_7_by_7_block_operator_error_upper_rational')},indent=2))


if __name__=='__main__':main()
