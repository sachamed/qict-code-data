#!/usr/bin/env python3
"""Independent GF(2), Clebsch--Gordan quadrature and dressed-source checks."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from fractions import Fraction

from nonabelian_common_car_kato import spinor_fundamental_matrix, su2_spinor_labels

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/nonabelian_common_car'


def cg_square_general(j1,j2,J,m1,m2):
    """Factorial-sum CG square; all labels are doubled integers."""
    M=m1+m2
    if abs(M)>J or abs(m1)>j1 or abs(m2)>j2 or J<abs(j1-j2) or J>j1+j2:
        return Fraction(0)
    half=lambda x:x//2
    a,b,c=half(J+j1-j2),half(J-j1+j2),half(j1+j2-J)
    f=math.factorial
    pref=Fraction((J+1)*f(a)*f(b)*f(c),f(half(j1+j2+J)+1))
    for x in [J+M,J-M,j1-m1,j1+m1,j2-m2,j2+m2]:pref*=f(half(x))
    total=Fraction(0)
    for k in range(j1+j2+J+2):
        arguments=[k,c-k,half(j1-m1)-k,half(j2+m2)-k,
                   half(J-j2+m1)+k,half(J-j1-m2)+k]
        if min(arguments)<0:continue
        total+=Fraction((-1)**k,math.prod(f(x) for x in arguments))
    return pref*total*total


def main():
    primitive=ROOT/'reproducibility/results/c10_primitive_gauss'
    c=json.loads((primitive/'C10_PRIMITIVE_GAUSS_KATO_SOURCE_CERTIFICATE.json').read_text())
    with np.load(primitive/'C10_INTEGER_GAUSS_BASIS_AND_SOURCE_PATHS.npz',allow_pickle=False) as z:
        chord_index={int(edge):i for i,edge in enumerate(z['chord_links'])}
    payload=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
    with np.load(payload,allow_pickle=False) as z:
        faces=z['plaquette_edge_indices']
    # Entirely different elimination arithmetic: binary columns and XOR.
    binary_pivots={}
    for face in faces:
        bits=0
        for edge in face:
            if int(edge) in chord_index:bits ^= 1<<chord_index[int(edge)]
        while bits:
            pivot=bits.bit_length()-1
            if pivot in binary_pivots:bits ^= binary_pivots[pivot]
            else:binary_pivots[pivot]=bits;break
    assert len(binary_pivots)==c['integer_boundary2_rank']==13356
    # Independent SU2 class quadrature, using the general factorial-sum CG
    # formula instead of the specialized Laurent integers used for propagation.
    labels=su2_spinor_labels(8);nquad=64
    theta=np.pi*np.arange(1,nquad+1)/(nquad+1)
    weight=2*np.sin(theta)**2/(nquad+1)
    functions=np.zeros((len(labels),nquad,2),complex)
    for i,(s,t) in enumerate(labels):
        for a,alpha in enumerate([1,-1]):
            for twice_M in range(-t,t+1,2):
                squared=cg_square_general(t,1,s,twice_M,alpha)
                functions[i,:,a]+=float(squared)*np.exp(1j*twice_M*theta)
        functions[i]*=math.sqrt((t+1)/(s+1))
    gram=np.einsum('ita,jta,t->ij',functions.conj(),functions,weight)
    magnetic=np.einsum('ita,jta,t->ij',functions.conj(),functions,weight*2*np.cos(theta))
    _,direct,_=spinor_fundamental_matrix(8)
    gram_error=float(np.linalg.norm(gram-np.eye(len(labels)),2))
    fusion_error=float(np.linalg.norm(magnetic-direct.toarray(),2))
    assert gram_error<4e-14 and fusion_error<4e-14
    source_error=0.
    for s in range(7):
        character=np.sin((s+1)*theta)/np.sin(theta)
        source=np.stack([np.exp(-1j*theta),np.exp(1j*theta)],axis=1)*character[:,None]/math.sqrt(2)
        overlaps=np.einsum('ita,ta,t->i',functions.conj(),source,weight)
        expected=np.array([math.sqrt((t+1)/(2*(s+1))) if r==s else 0 for r,t in labels])
        source_error=max(source_error,float(np.linalg.norm(overlaps-expected)))
    assert source_error<4e-14
    # Explicit six columns on a common local representation carrier: twelve
    # left (three families x weak x compensator), three right components.
    rng=np.random.default_rng(9);dressed_error=0.
    for _ in range(32):
        x=rng.normal(size=4);x/=np.linalg.norm(x)
        u=np.array([[x[0]+1j*x[3],x[2]+1j*x[1]],[-x[2]+1j*x[1],x[0]-1j*x[3]]])
        angle=float(rng.uniform(-np.pi,np.pi))
        v=np.zeros((15,6),complex)
        for a in range(3):
            v[4*a:4*a+4,a]=np.exp(-3j*angle)*u.reshape(-1)/math.sqrt(2)
            v[12+a,3+a]=np.exp(-6j*angle)
        dressed_error=max(dressed_error,float(np.linalg.norm(v.conj().T@v-np.eye(6))))
    assert dressed_error<3e-15
    # The specified cell swap exchanges identical representation copies;
    # the N=0 and N=1 restrictions give the parity-dependent return exactly.
    cell=np.zeros((31,31),complex);cell[0,0]=1
    cell[1:16,16:31]=np.eye(15);cell[16:31,1:16]=np.eye(15)
    sources=np.zeros((31,6),complex);sources[1:16]=v
    cell_error=0.
    for n in range(13):
        observed=sources.conj().T@np.linalg.matrix_power(cell,n)@sources
        expected=np.eye(6) if n%2==0 else np.zeros((6,6))
        cell_error=max(cell_error,float(np.linalg.norm(observed-expected)))
    assert cell_error<4e-15
    main=json.loads((OUT/'NONABELIAN_COMMON_CAR_KATO_CERTIFICATE.json').read_text())
    data=OUT/'NONABELIAN_COMMON_CAR_KATO_MOMENTS.npz'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==main['generated_npz_sha256']
    with np.load(data,allow_pickle=False) as z:
        for level in range(4):
            a=z[f'L{level}_gauge_Cn6'];b=z[f'L{level}_gauge_cell_Cn6']
            assert np.allclose(a[0],np.eye(6),rtol=0,atol=2e-14)
            assert np.array_equal(a[::2],b[::2]) and not np.any(b[1::2])
    result={'status':'PASS_INDEPENDENT_INTEGER_GRAPH_AND_SU2_CAR_SOURCE_CHECKS',
            'independent_GF2_boundary2_rank':len(binary_pivots),
            'CG_quadrature_points':nquad,'CG_basis_orthogonality_error':gram_error,
            'CG_fusion_matrix_error':fusion_error,'CG_source_embedding_error':source_error,
            'explicit_dressed_six_source_Gram_error':dressed_error,
            'explicit_cell_six_source_moment_error':cell_error,
            'scope':'Integer graph rank plus compact gauge/cell sublayer, not complete charged C10 or a physical mass calculation.'}
    (OUT/'INDEPENDENT_NONABELIAN_CAR_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
