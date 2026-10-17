#!/usr/bin/env python3
"""Independent position-space, rational-residual and perturbation checks."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,itertools,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/global_boundary_verification'


def exact_coefficients(diagonal,V,zero):
    c=[F(int(V[i,zero])) for i in range(len(diagonal))]
    second=-sum((c[i]*c[i]/int(diagonal[i]) for i in range(len(c)) if i!=zero),F(0))
    third=sum((c[i]*int(V[i,j])*c[j]/(int(diagonal[i])*int(diagonal[j]))
               for i in range(len(c)) for j in range(len(c)) if i!=zero and j!=zero),F(0))
    norm=sum((c[i]*c[i]/int(diagonal[i])**2 for i in range(len(c)) if i!=zero),F(0))
    return second,third,norm


def position_space_gap():
    # Independently construct real-space BCC translations on a primitive box.
    L=3;positions=list(itertools.product(range(L),repeat=3));index={x:i for i,x in enumerate(positions)}
    d=len(positions);w=np.eye(d,dtype=complex);ni=[np.zeros((d,d),complex) for _ in range(3)]
    qY=-3
    for sx,sy,sz in itertools.product((-1,1),repeat=3):
        delta=((sx+sy)//2,(sx+sz)//2,(sy+sz)//2)
        alpha=-sx*sy*sz*math.pi/9
        t=np.zeros((d,d),complex)
        for x in positions:
            y=tuple((a+b)%L for a,b in zip(x,delta));t[index[x],index[y]]=np.exp(1j*qY*alpha)
        assert np.linalg.norm(t.conj().T@t-np.eye(d))<1e-13
        w-=t/8
        ni[0]+=(sx/(8j)-sy*sz/8)*t
        ni[1]+=(sy/(8j)+sx*sz/8)*t
        ni[2]+=(sz/(8j)-sx*sy/8)*t
    pauli=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    gamma=[np.kron(pauli[1],s) for s in pauli];g5=np.kron(pauli[2],np.eye(2))
    D=sum(1j*np.kron(a,g) for a,g in zip(ni,gamma))+np.kron(w-.5*np.eye(d),np.eye(4))
    H=np.kron(np.eye(d),g5)@D
    constant=np.kron(np.ones((d,1))/math.sqrt(d),np.eye(4))
    herm=float(np.linalg.norm(H-H.conj().T));residual=float(np.linalg.norm(H@constant))
    minimum=float(np.linalg.svd(H,compute_uv=False).min())
    assert herm<2e-13 and residual<2e-13 and minimum<2e-13
    return {'primitive_volume':L,'matrix_dimension':4*d,'hermiticity_error':herm,
            'constant_spinor_null_residual':residual,'minimum_singular_value':minimum}


def coupled_residual():
    base=ROOT/'reproducibility/results/coupled_boundary'
    cert=json.loads((base/'COUPLED_BOUNDARY_GNS_CERTIFICATE.json').read_text())
    with np.load(base/'COUPLED_PLAQUETTE_BOUNDARY_MOMENTS.npz',allow_pickle=False) as z:
        B=z['integer_cycle_matrix'].astype(int);labels=z['L3_ground_labels']
    record=cert['cutoffs'][-1]['ground'];w={tuple(x):F(y) for x,y in zip(labels,record['rational_witness_components'])}
    radius=record['radius'];norm=sum(x*x for x in w.values());mu=F(record['rayleigh_rational'])
    residual2=F(0);hv0=F(0)
    for r,s in itertools.product(range(-radius-1,radius+2),repeat=2):
        current=B@np.array([r,s]);electric=2*sum(int(x)**2 for x in current)+sum(map(int,current))
        value=w.get((r,s),F(0))
        hv=(electric+8)*value-2*sum((w.get(a,F(0)) for a in [(r-1,s),(r+1,s),(r,s-1),(r,s+1)]),F(0))
        residual2+=(hv-mu*value)**2;hv0+=value*hv
    assert hv0/norm==mu and residual2/norm==F(record['residual_squared_rational'])
    return {'state_radius':radius,'rational_Rayleigh_value_matches':True,
            'rational_infinite_residual_matches':True,'integer_Gauss_matrix_used':True}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    y=exact_coefficients([8,0,8],np.array([[0,-2,0],[-2,0,-2],[0,-2,0]]),1)
    weak=exact_coefficients([0,12],np.array([[0,-16],[-16,0]]),0)
    color=exact_coefficients([0,16,16],np.array([[0,-6,-6],[-6,0,-6],[-6,-6,0]]),0)
    totals=tuple(sum(x[j] for x in [y,weak,color]) for j in range(3))
    assert totals==(-F(161,6),-F(27,16),F(629,288))
    p=ROOT/'reproducibility/results/global_primitive_seed/GLOBAL_A7_KATO_SEED_CERTIFICATE.json'
    globalcert=json.loads(p.read_text());N=globalcert['plaquettes_used']
    assert F(globalcert['energy_Taylor_coefficients']['quadratic_rational'])==N*totals[0]
    assert F(globalcert['energy_Taylor_coefficients']['cubic_rational'])==N*totals[1]
    assert F(globalcert['ground_vector_derivative_norm_squared_rational'])==N*totals[2]
    paths=['reproducibility/results/chiral_cone_domain/CHIRAL_CONE_DOMAIN_CERTIFICATE.json',
           'reproducibility/results/global_primitive_seed/GLOBAL_A7_KATO_SEED_CERTIFICATE.json',
           'reproducibility/results/coupled_boundary/COUPLED_BOUNDARY_GNS_CERTIFICATE.json',
           'reproducibility/results/spatial_state/SPATIAL_STATE_SELECTION_CERTIFICATE.json']
    for path in paths:
        c=json.loads((ROOT/path).read_text());directory=(ROOT/path).parent
        npz=next(directory.glob('*.npz'))
        assert hashlib.sha256(npz.read_bytes()).hexdigest()==c['data_sha256']
    spatial=json.loads((ROOT/paths[-1]).read_text())
    pi_interval=spatial['exact_U1_primitive_return_certificate']['pi_interval_rational']
    xlo,xhi=[4*F(value)/63 for value in pi_interval]
    assert F(199,1000)<xlo<xhi<F(1,5)
    j0_lower=1-xhi*xhi/4
    j1_lower=xlo/2-xhi**3/16
    bessel_distance_lower=2*j0_lower*j1_lower
    assert bessel_distance_lower>F(9801,50000)
    result={'status':'PASS_INDEPENDENT_GLOBAL_SEED_GAUGE_KERNEL_AND_COUPLED_BOUNDARY_CHECKS',
      'exact_energy_coefficients_per_plaquette':[str(totals[0]),str(totals[1])],
      'exact_vector_derivative_norm_per_plaquette':str(totals[2]),
      'independent_real_space_chiral_kernel':position_space_gap(),
      'independent_coupled_infinite_residual':coupled_residual(),
      'independent_infinite_boundary_state_distance_lower_rational':str(bessel_distance_lower),
      'independent_infinite_boundary_state_distance_lower':float(bessel_distance_lower),
      'input_certificate_sha256':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
      'scope':'Independent checks of the exact new statements and supplied finite computations; not a certificate of the complete charged C10 or its connected spatial limit.'}
    (OUT/'INDEPENDENT_GLOBAL_BOUNDARY_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
