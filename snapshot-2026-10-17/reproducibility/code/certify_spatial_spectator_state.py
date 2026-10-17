#!/usr/bin/env python3
"""Exact nonstationarity bound and spatial spectator test for the A7/GNS state.

The disjoint-plaquette restriction is explicit. It is a necessary state
selection control, not the full connected C10 thermodynamic limit.
"""
from fractions import Fraction as F
from pathlib import Path
from functools import reduce
import hashlib,itertools,json,math,sys
import numpy as np
from scipy.linalg import expm,schur
from nonabelian_common_car_kato import sqrt_upper

sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/spatial_state'
PAYLOAD=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
BASE=ROOT/'reproducibility/results/nonabelian_common_car'


def pi_interval():
    def atan_bounds(x,N=50):
        value=sum((F((-1)**k,(2*k+1)*x**(2*k+1)) for k in range(N)),F(0))
        other=value+F((-1)**N,(2*N+1)*x**(2*N+1))
        return min(value,other),max(value,other)
    a,b=atan_bounds(5),atan_bounds(239)
    lo,hi=16*a[0]-4*b[1],16*a[1]-4*b[0]
    # tan(4*atan(1/5)-atan(1/239))=1 on the (0,pi/2) branch.
    assert (F(120,119)-F(1,239))/(1+F(120,119)*F(1,239))==1
    scale=10**55
    low=F((lo*scale).numerator//(lo*scale).denominator,scale)
    x=hi*scale;high=F(-(-x.numerator//x.denominator),scale)
    assert low<=lo<hi<=high
    return low,high


def exp_numerator(p,q,integer,order=64):
    """Taylor exponential exp(-i integer*p/q), with common integer denominator."""
    real=imag=0;fac=math.factorial(order)
    for k in range(order+1):
        term=(integer*p)**k*q**(order-k)*(fac//math.factorial(k))
        a=k%4
        if a==0:real+=term
        elif a==1:imag-=term
        elif a==2:real-=term
        else:imag+=term
    return real,imag


def exp_upper(x,stop=64):
    assert x<stop+2
    return sum((x**j/math.factorial(j) for j in range(stop+1)),F(0)) + (x**(stop+1)/math.factorial(stop+1))/(1-x/F(stop+2))


def rounded(value,digits=40,up=False):
    scale=10**digits;x=value*scale
    n=-(-x.numerator//x.denominator) if up else x.numerator//x.denominator
    return F(n,scale)


def certified_return():
    data=json.loads((BASE/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json').read_text())
    ground=data['cutoff_results'][-1]['ground_certificates']['U1']
    labels=[x[0] for x in ground['support_labels']]
    weights=[F(x) for x in ground['rational_witness_components']]
    common=math.lcm(*(w.denominator for w in weights))
    integers={r:int(w*common) for r,w in zip(labels,weights)}
    norm=sum(x*x for x in integers.values())
    plow,phigh=pi_interval();phi=(plow+phigh)/126
    p,q=phi.numerator,phi.denominator
    eorder,morder=64,14
    cache={e:exp_numerator(p,q,e,eorder) for e in {8*r*r%126 for r in labels}}
    vector={r:(integers[r]*cache[8*r*r%126][0],integers[r]*cache[8*r*r%126][1]) for r in labels}
    totalreal=totalimag=0;fac=math.factorial(morder)
    for k in range(morder+1):
        a=sum(integers[r]*vector.get(r,(0,0))[0] for r in labels)
        b=sum(integers[r]*vector.get(r,(0,0))[1] for r in labels)
        coefficient=(2*p)**k*q**(morder-k)*(fac//math.factorial(k))
        j=k%4
        if j==1:a,b=-b,a
        elif j==2:a,b=-a,-b
        elif j==3:a,b=b,-a
        totalreal+=coefficient*a;totalimag+=coefficient*b
        following={}
        for r,(a,b) in vector.items():
            for s in [r-1,r+1]:
                u,v=following.get(s,(0,0));following[s]=(u+a,v+b)
        vector=following
    denominator=q**(eorder+morder)*math.factorial(eorder)*math.factorial(morder)*norm
    real,imag=F(totalreal,denominator),F(totalimag,denominator)
    # The constant magnetic phase is irrelevant for the absolute return.
    electric_tail=exp_upper(F(7))*F(7)**65/math.factorial(65)
    magnetic_tail=exp_upper(F(1,5))*F(1,5)**15/math.factorial(15)
    state_error=2*F(ground['state_trace_distance_upper_rational'])
    phase_error=F(129)*(phigh-plow)/126
    error=state_error+magnetic_tail+electric_tail*(1+magnetic_tail)+phase_error
    upper=(sqrt_upper(real*real+imag*imag)+error)**2
    upper=rounded(upper,40,True)
    assert upper<1
    return {'return_real_interval_rational':[str(rounded(real-error)),str(rounded(real+error,up=True))],
      'return_imag_interval_rational':[str(rounded(imag-error)),str(rounded(imag+error,up=True))],
      'return_convention':'Constant exp(-4i*phi) removed; magnitude is unchanged.',
      'return_modulus_squared_upper_rational':str(upper),
      'return_modulus_squared_upper':float(upper),
      'pure_density_stationarity_distance_squared_lower_rational':str(1-upper),
      'pure_density_stationarity_distance_lower':math.sqrt(float(1-upper)),
      'Taylor_and_state_error_upper_rational':str(rounded(error,40,True)),
      'pi_interval_rational':[str(plow),str(phigh)],
      'electric_Taylor_order':eorder,'magnetic_Taylor_order':morder,
      'all_arithmetic_rational_or_integer_with_verified_square_root_upper':True}


def kron_all(values):
    return reduce(np.kron,values)


def finite_C9_check():
    r=np.arange(-2,3);d=len(r);phi=math.pi/63
    fusion=np.diag(np.ones(d-1),1)+np.diag(np.ones(d-1),-1)
    h=np.diag(8*r*r+4)-2*fusion
    _,v=np.linalg.eigh(h);seed=v[:,0]
    mag=expm(-1j*phi*(4*np.eye(d)-2*fusion))
    u0=mag@np.diag(np.exp(-1j*phi*(8*r*r%126)))
    neutral,Q=schur(u0,output='complex');lambdas=np.diag(neutral)
    assert np.linalg.norm(neutral-np.diag(lambdas))<1e-12
    coeff=Q.conj().T@seed
    results=[];arrays={};reference=None;rho_one=None
    for N in [1,2,3]:
        labels=list(itertools.product(range(d),repeat=N))
        spectrum=np.array([np.prod(lambdas[list(a)]) for a in labels])
        amplitudes=np.array([np.prod(coeff[list(a)]) for a in labels])
        groups=[]
        for i,value in enumerate(spectrum):
            group=next((g for g in groups if abs(spectrum[g[0]]-value)<1e-10),None)
            if group is None:groups.append([i])
            else:group.append(i)
        representatives=[spectrum[g[0]] for g in groups]
        gap=min((abs(a-b) for j,a in enumerate(representatives) for b in representatives[j+1:]),default=2)
        assert gap>1e-7
        root=np.zeros((d**N,d**N),complex);rho=root.copy()
        for g in groups:
            a=np.zeros(d**N,complex);a[g]=amplitudes[g]
            weight=np.vdot(a,a).real
            rho+=np.outer(a,a.conj())
            if weight>1e-26:root+=np.outer(a,a.conj())/math.sqrt(weight)
        bigQ=kron_all([Q]*N);root=bigQ@root@bigQ.conj().T;rho=bigQ@rho@bigQ.conj().T
        bigu0=kron_all([u0]*N)
        stat=float(np.linalg.norm(bigu0@rho@bigu0.conj().T-rho))
        moments=np.zeros((5,6,6),complex)
        for charge,start in [(-3,0),(-6,3)]:
            uq=mag@np.diag(np.exp(-1j*phi*((8*r*r+4*charge*r+2*charge*charge+charge)%126)))
            biguf=kron_all([uq]+[u0]*(N-1));x=root.copy()
            for n in range(5):
                value=np.vdot(root,x)
                for a in range(start,start+3):moments[n,a,a]=value
                x=biguf@x@bigu0.conj().T
        if reference is None:reference=moments.copy();rho_one=rho.copy()
        error=float(np.max(abs(moments-reference)))
        assert error<2e-11 and stat<2e-12
        independent_density=kron_all([rho_one]*N)
        independent_dephasing_difference=float(np.linalg.norm(rho-independent_density))
        results.append({'spectator_blocks':N-1,'dimension':d**N,
          'full_dephasing_cluster_count':len(groups),'minimum_between_cluster_gap_diagnostic':gap,
          'stationarity_Frobenius_error':stat,'C0_Frobenius_error':float(np.linalg.norm(moments[0]-np.eye(6))),
          'moment_difference_from_one_block':error,
          'Frobenius_difference_from_independently_dephased_product':independent_dephasing_difference})
        arrays[f'N{N}_C9_Cn6']=moments
    return results,arrays


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(PAYLOAD,allow_pickle=False) as z:
        faces=z['plaquette_edge_indices'].astype(int)
    used=set();packing=[]
    for p,edges in enumerate(faces):
        if not used.intersection(map(int,edges)):
            packing.append(p);used.update(map(int,edges))
    assert len(used)==4*len(packing)
    certificate=certified_return()
    with np.load(BASE/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz',allow_pickle=False) as z:
        a=np.prod(z['L3_neutral_group_returns'],axis=0)
        local=z['L3_gauge_Cn6'].copy()
    sizes=sorted(set([n for n in [1,2,4,8,16,32,64,128,256,512,1024,len(packing)] if n<=len(packing)]))
    moments=np.array([local*(abs(a)**(2*(N-1)))[:,None,None] for N in sizes])
    finite,arrays=finite_C9_check()
    arrays.update({'link_disjoint_actual_plaquette_ids':np.array(packing,np.int32),
      'plaquette_counts':np.array(sizes),'neutral_single_block_return':a,
      'primitive_A7_spectator_Cn6':moments})
    npz=OUT/'SPATIAL_SPECTATOR_AND_C9_DATA.npz';np.savez_compressed(npz,**arrays)
    result={'status':'EXACT_NONSTATIONARITY_BOUND_AND_SPATIAL_STATE_SELECTION_TEST',
      'payload_sha256':hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
      'actual_link_disjoint_plaquettes':len(packing),
      'exact_U1_primitive_return_certificate':certificate,
      'pure_primitive_relative_moment_with_N_blocks':'C_N(n)=C_1(n)*|<Omega|U0^n|Omega>|^(2*(N-1))',
      'rigorous_first_moment_norm_bound':'||C_N(1)|| <= q_upper^(N-1), q_upper is the certified U1 return modulus squared upper.',
      'limit_of_first_primitive_relative_moment_in_disjoint_sequence':'zero',
      'computed_three_group_single_block_return_modulus_squared_n1':float(abs(a[1])**2),
      'C9_exact_volume_identity':'Partial trace of the full mean-ergodic product dephasing gives the one-block C9 state; local relative observables and their stationary GNS moments are independent of the spectator count.',
      'independent_C9_finite_regulator_checks':finite,
      'global_C9_dephasing_is_not_independent_factor_dephasing':True,
      'scope':'Gauge/cell factors on isolated link-disjoint plaquettes selected from the actual payload. The finite C9 control uses the U1 component. No limit for the complete connected BCC/route-first C10 is inferred.',
      'full_connected_spatial_limit_certified':False,
      'data_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
    (OUT/'SPATIAL_STATE_SELECTION_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'actual_disjoint_plaquettes':len(packing),
      'strict_U1_return_modulus_squared_upper':certificate['return_modulus_squared_upper'],
      'primitive_stationarity_distance_lower':certificate['pure_density_stationarity_distance_lower'],
      'C9_moment_volume_error':max(x['moment_difference_from_one_block'] for x in finite)},indent=2))


if __name__=='__main__':main()
