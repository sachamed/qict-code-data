"""Exact rational positivity and polynomial bounds for the recorded C9 moments.

The theorem is conditional on the explicitly Hermitian, diagonally normalized
stored moments. It certifies their finite moment problem, not continuum physics.
Only 13-by-13 scalar blocks are needed: all off-diagonal input entries are zero.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parent
ZERO = (F(0), F(0))
ONE = (F(1), F(0))
def add(a,b): return (a[0]+b[0], a[1]+b[1])
def sub(a,b): return (a[0]-b[0], a[1]-b[1])
def mul(a,b): return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
def conj(a): return (a[0], -a[1])
def scale(a,s): return (a[0]*s, a[1]*s)
def exact(z): return (F(float(np.real(z))), F(float(np.imag(z))))
def qsum(values):
    out=ZERO
    for z in values: out=add(out,z)
    return out
def upward(q, digits=12):
    d=10**digits; n=-((-q.numerator*d)//q.denominator)
    return f'{n//d}.{n%d:0{digits}d}'


def certify_block(raw):
    normalization=F(float(raw[0].real))
    moments=[ONE]+[scale(exact(z),1/normalization) for z in raw[1:]]
    N=len(raw)
    T=[[moments[j-k] if j>=k else conj(moments[k-j]) for k in range(N)] for j in range(N)]
    L=[[ZERO for _ in range(N)] for _ in range(N)]; pivots=[]
    for j in range(N):
        pivot=sub(T[j][j],qsum(scale(mul(L[j][k],conj(L[j][k])),pivots[k]) for k in range(j)))
        assert pivot[1] == 0 and pivot[0] > 0
        pivots.append(pivot[0]);L[j][j]=ONE
        for i in range(j+1,N):
            value=sub(T[i][j],qsum(scale(mul(L[i][k],conj(L[j][k])),pivots[k]) for k in range(j)))
            L[i][j]=scale(value,1/pivots[j])
    # Floating point proposes a polynomial; exact rational arithmetic verifies it.
    c=np.array([complex(float(z[0]),float(z[1])) for z in moments])
    tf=np.array([[c[j-k] if j>=k else c[k-j].conjugate() for k in range(N)] for j in range(N)])
    w=np.linalg.solve(tf,np.ones(N));w=w/np.sum(w)
    weights=[exact(z) for z in w[:-1]]
    weights.append(sub(ONE,qsum(weights)))
    assert qsum(weights) == ONE
    bound=qsum(mul(conj(weights[j]),mul(T[j][k],weights[k])) for j in range(N) for k in range(N))
    assert bound[1] == 0 and bound[0] > 0
    return {'moment_count':N,'positive_exact_LDL_pivots':len(pivots),
            'weights_real_imag_rational':[[str(z[0]),str(z[1])] for z in weights],
            'polynomial_energy_rational':str(bound[0]),
            'atom_weight_upper_outward_12dp':upward(bound[0]),
            'atom_weight_upper_nominal':float(bound[0])}


def main():
    path=ROOT/'FINITE_C9_COMMON_STATE_AND_MOMENTS.npz'
    rows=[]
    with np.load(path,allow_pickle=False) as data:
        for r in range(1,5):
            C=data[f'R{r}_moments']
            assert np.count_nonzero(C-np.array([np.diag(np.diag(m)) for m in C]))==0
            for group,indices in [('left',[0,1,2]),('right',[3,4,5])]:
                assert all(np.array_equal(C[:,i,i],C[:,indices[0],indices[0]]) for i in indices)
                row=certify_block(C[:,indices[0],indices[0]])
                row.update(regulator=f'R{r}',source_group=group,channels=indices)
                rows.append(row)
    out={'scope':'Exact rational certificate for the Hermitian C0-normalized recorded finite moment problem.',
         'normalization':'Remove Im C0 and divide each diagonal moment by Re C0; retain all input arrays unchanged.',
         'physical_moment_errors':'Independent input uncertainties and regulator/continuum errors are separate.',
         'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'blocks':rows}
    dest=ROOT.parent/'data/EXACT_MOMENT_POLYNOMIAL_CERTIFICATE.json'
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['regulator','source_group','atom_weight_upper_outward_12dp']} for r in rows],indent=2))


if __name__=='__main__': main()
