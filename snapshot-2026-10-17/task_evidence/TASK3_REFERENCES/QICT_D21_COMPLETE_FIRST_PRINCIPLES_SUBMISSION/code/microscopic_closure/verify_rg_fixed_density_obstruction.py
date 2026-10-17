#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'microscopic_closure'/'RG_FIXED_DENSITY_OBSTRUCTION.json'
OUT.parent.mkdir(parents=True,exist_ok=True)

sx=np.array([[0,1],[1,0]],complex)
sy=np.array([[0,-1j],[1j,0]],complex)
sz=np.array([[1,0],[0,-1]],complex)
I=np.eye(2,dtype=complex)

def U(k):
    out=I.copy()
    for x,s in zip(k,(sx,sy,sz)):
        out=out@(math.cos(x)*I-1j*math.sin(x)*s)
    return out

def kappa(k):
    k=np.asarray(k,float)
    c=np.cos(k); s=np.sin(k)
    ct=float(np.prod(c)-np.prod(s))
    th=math.acos(max(-1.0,min(1.0,ct)))
    return math.log(3.0)-(1.0/3.0)*math.log(max(abs(1+2*math.cos(3*th)),1e-300))

def main():
    # Nontrivial derivative at k=0: finite difference norm.
    h=1e-7
    dUx=(U((h,0,0))-U((-h,0,0)))/(2*h)
    deriv_norm=float(np.linalg.norm(dUx,ord='fro'))

    # Exact 2x2x2 blocking folds eight Brillouin sectors. Its fibre has dimension 8*2=16.
    K=np.array([0.31,-0.44,0.27])
    aliases=[]
    for nx in (0,1):
      for ny in (0,1):
       for nz in (0,1):
        k=(K+math.pi*np.array([nx,ny,nz]))/2
        aliases.extend(np.angle(np.linalg.eigvals(U(k))).tolist())
    aliases=np.sort(np.mod(np.array(aliases)+math.pi,2*math.pi)-math.pi)

    # The original two-component fibre has only two quasienergies.
    coarse_two=np.sort(np.mod(np.angle(np.linalg.eigvals(U(K)))+math.pi,2*math.pi)-math.pi)

    # Direct scale-dependence diagnostic of the canonical compression density along a generic ray.
    k0=np.array([0.73,-0.51,0.39])
    scales=np.array([1.0,0.8,0.6,0.4,0.2,0.1])
    kappas=np.array([kappa(t*k0) for t in scales])
    # logarithmic finite-difference slopes d kappa/d ell, ell=-ln t
    ell=-np.log(scales)
    slopes=np.diff(kappas)/np.diff(ell)

    report={
      'status':'PASS_RG_EXACT_FIXED_DENSITY_OBSTRUCTION',
      'external_data_used':False,
      'theorem':'For a nonconstant analytic finite-range Bloch QCA U(k), exact continuous dilation covariance U(e^{-ell}k)=W(ell)U(k)W(ell)^dagger with k-independent W on an open ell interval is impossible. Differentiating at k=0 gives e^{-ell} d_iU(0)=W d_iU(0) W^dagger; unitary invariance of the Frobenius norm forces d_iU(0)=0 for ell!=0. Repeating for the first nonzero Taylor derivative forces all nonconstant derivatives to vanish, contradicting a nontrivial stream.',
      'C8_dUdkx_at_origin_frobenius_norm':deriv_norm,
      'exact_2x2x2_blocking':{
        'fine_alias_count':8,
        'fine_internal_dimension_after_exact_folding':16,
        'original_internal_dimension':2,
        'sample_folded_quasienergies':aliases.tolist(),
        'sample_original_two_component_quasienergies':coarse_two.tolist(),
        'consequence':'Exact decimation does not close on the same two-component C8 symbol; projecting back is an additional lossy RG map.'
      },
      'canonical_stream_scale_diagnostic':{
        'k0':k0.tolist(),'scale_factors':scales.tolist(),'ell':ell.tolist(),
        'kappa':kappas.tolist(),'finite_difference_dkappa_dell':slopes.tolist(),
        'slope_range':[float(slopes.min()),float(slopes.max())]
      },
      'verdict':'The current microscopic QCA does not support a proof that gamma_0(ell)=alpha_exact on the entire Planck-to-IR window. A constant density can only be an emergent fixed-point approximation after a specified coarse-graining/projection and a demonstrated scaling regime. The exact exponential absolute-mass law is therefore not a theorem of the present finite-range C8 operator.'
    }
    assert deriv_norm>1.0
    assert len(aliases)==16
    assert np.ptp(slopes)>1e-3
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
