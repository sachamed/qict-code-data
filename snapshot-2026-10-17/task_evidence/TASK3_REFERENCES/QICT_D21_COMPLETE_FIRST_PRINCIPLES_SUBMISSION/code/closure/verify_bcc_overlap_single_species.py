#!/usr/bin/env python3
from __future__ import annotations
import json, math, itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
P=math.pi

# Pauli / Euclidean gamma matrices in chiral representation
s1=np.array([[0,1],[1,0]],complex)
s2=np.array([[0,-1j],[1j,0]],complex)
s3=np.array([[1,0],[0,-1]],complex)
I2=np.eye(2,dtype=complex); Z2=np.zeros((2,2),complex)
# gamma_i = [[0,-i sigma_i],[i sigma_i,0]], gamma5=diag(I,-I)
gam=[]
for s in (s1,s2,s3):
    gam.append(np.block([[Z2,-1j*s],[1j*s,Z2]]))
g5=np.block([[I2,Z2],[Z2,-I2]])
I4=np.eye(4,dtype=complex)

def c7_n(k):
    x,y,z=map(float,k)
    cx,cy,cz=np.cos([x,y,z]); sx,sy,sz=np.sin([x,y,z])
    return np.array([sx*cy*cz+cx*sy*sz,
                     cx*sy*cz-sx*cy*sz,
                     cx*cy*sz+sx*sy*cz],float)

def W(k):
    x,y,z=map(float,k)
    return float(1-np.cos(x)*np.cos(y)*np.cos(z))

def kernel(k,M=0.5):
    n=c7_n(k); m=W(k)-M
    DW=m*I4 + 1j*sum(float(n[j])*gam[j] for j in range(3))
    H=g5@DW
    return H,n,m

def sign_herm(H):
    w,V=np.linalg.eigh((H+H.conj().T)/2)
    if np.min(np.abs(w))<1e-12: raise RuntimeError('kernel gap closed')
    return (V*np.sign(w))@V.conj().T

def Dov(k,M=0.5):
    H,n,m=kernel(k,M)
    return I4+g5@sign_herm(H)

def exact_zero_classes():
    # Complete solutions of n(k)=0 in the cubic cover [-pi,pi)^3.
    # Away from cos=0, tan equations force tan x=tan y=tan z=0.
    # If any cos vanishes, the equations force all three cos to vanish.
    nodes=[]
    for x in (-P,0.0):
      for y in (-P,0.0):
       for z in (-P,0.0): nodes.append((x,y,z))
    for x in (-P/2,P/2):
      for y in (-P/2,P/2):
       for z in (-P/2,P/2): nodes.append((x,y,z))
    return nodes

def main():
    # Clifford checks
    cliff=0.0
    for i in range(3):
      cliff=max(cliff,float(np.linalg.norm(gam[i]@gam[i]-I4)))
      cliff=max(cliff,float(np.linalg.norm(gam[i]@g5+g5@gam[i])))
      for j in range(i): cliff=max(cliff,float(np.linalg.norm(gam[i]@gam[j]+gam[j]@gam[i])))
    
    nodes=exact_zero_classes()
    node_rows=[]; zero_count=0
    for k in nodes:
        n=c7_n(k); w=W(k); m=w-0.5
        D=Dov(k); sv=np.linalg.svd(D,compute_uv=False)
        iszero=float(sv.min())<1e-10
        zero_count += int(iszero)
        node_rows.append({'k_over_pi':[float(v/P) for v in k], 'n_norm':float(np.linalg.norm(n)), 'W':w, 'mW':m, 'Dov_min_sv':float(sv.min()), 'overlap_zero':bool(iszero)})
    # In cubic cover Gamma has four reciprocal representatives with W=0; count physical class by W/m sign.
    # Direct theorem: D_ov eigenvalues are 1+(m +/- i|n|)/sqrt(m^2+|n|^2); zero iff n=0 and m<0.
    physical_zero_representatives=sum(1 for r in node_rows if r['overlap_zero'])
    # Reciprocal BCC makes those four representatives one class.
    physical_zero_classes=1 if physical_zero_representatives==4 else -1

    rng=np.random.default_rng(2026091402)
    gw=0.0; formula=0.0; min_kernel=1e9; min_dov_nonphys=1e9
    for _ in range(5000):
        k=rng.uniform(-P,P,3)
        H,n,m=kernel(k); ev=np.linalg.eigvalsh(H); min_kernel=min(min_kernel,float(np.min(np.abs(ev))))
        D=Dov(k)
        gw=max(gw,float(np.linalg.norm(g5@D+D@g5-D@g5@D,2)))
        R=math.sqrt(float(np.dot(n,n))+m*m)
        # eigenvalue modulus squared of overlap: 2(1+m/R)
        sv=np.linalg.svd(D,compute_uv=False)
        formula=max(formula,abs(float(np.min(sv)**2)-2*(1+m/R)))
        if W(k)>0.75: min_dov_nonphys=min(min_dov_nonphys,float(np.min(sv)))

    # Dense global scan: supplementary numerical audit, theorem does not depend on it.
    dense_min_kernel=1e9; dense_arg=None
    N=61
    grid=np.linspace(-P,P,N,endpoint=False)
    for x in grid:
      for y in grid:
       for z in grid:
        H,n,m=kernel((x,y,z)); g=math.sqrt(float(np.dot(n,n))+m*m)
        if g<dense_min_kernel: dense_min_kernel=g; dense_arg=(x,y,z)

    # The M=1/2 choice is the unique maximizer of min(M,1-M) on (0,1).
    xs=np.linspace(1e-5,1-1e-5,100001); margins=np.minimum(xs,1-xs); i=int(np.argmax(margins))
    minimax=float(xs[i]); maxmargin=float(margins[i])

    status=(cliff<1e-12 and physical_zero_classes==1 and gw<1e-11 and formula<1e-10 and
            abs(minimax-0.5)<2e-5 and abs(maxmargin-0.5)<2e-5 and dense_min_kernel>0.45)
    cert={
      'status':'PASS' if status else 'FAIL',
      'taxonomy':'DEMONSTRATED + CALCULATED',
      'kernel':'H_W(k)=gamma5[i sum_i gamma_i n_i(k) + (W_BCC(k)-1/2)]',
      'overlap_operator':'D_ov(k)=I+gamma5 sign(H_W(k))',
      'exact_overlap_zero_criterion':'D_ov(k) has a zero iff n(k)=0 and W_BCC(k)-1/2<0',
      'exact_c7_zero_solution_proof':'If all cosines are nonzero, division gives tx+ty tz=0, ty-tx tz=0, tz+tx ty=0, hence tx=ty=tz=0. If any cosine vanishes, the remaining equations force all three cosines to vanish. These give the 16 cubic-cover nodes and four reciprocal BCC classes.',
      'cubic_cover_zero_representatives':physical_zero_representatives,
      'physical_reciprocal_zero_classes':physical_zero_classes,
      'node_rows':node_rows,
      'unique_minimax_M0_over_rW':0.5,
      'minimax_margin_over_rW':0.5,
      'ginsparg_wilson_max_residual':gw,
      'overlap_singular_formula_max_residual':formula,
      'random_kernel_min_abs_eigenvalue':min_kernel,
      'dense_grid_N_per_axis':N,
      'dense_kernel_min_abs_eigenvalue':dense_min_kernel,
      'dense_min_arg_over_pi':[float(v/P) for v in dense_arg],
      'clifford_residual':cliff,
      'external_particle_data_used':False,
      'interpretation':'The corrected BCC-periodic Wilson kernel gives exactly one physical overlap zero class. The strict 3D two-band no-go is escaped because chirality is realized by the overlap/domain-wall regulator class, not by deleting a compensating Weyl node inside C7.'
    }
    (CERT/'bcc_overlap_single_species_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps(cert,indent=2))
    raise SystemExit(0 if status else 1)
if __name__=='__main__': main()
