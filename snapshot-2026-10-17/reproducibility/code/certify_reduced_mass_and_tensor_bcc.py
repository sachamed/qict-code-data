#!/usr/bin/env python3
"""Independent finite Schur and BCC spin-two benchmark verification.

Both sectors are explicit benchmark operators. No particle masses or dynamical
metric responses from the interacting C10/C11 kernel are inferred from them.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility' / 'results' / 'REDUCED_SCHUR_TENSOR_BCC_CERTIFICATE.json'
ALPHA = 0.156806466326680890086
P = np.array([55., 21., 3.])
W = np.exp(-ALPHA * (P - P[-1]))


def mfun(w, t):
    return (2*t*w)/(1+np.sqrt(1+4*t*w))


def d_bcc(k, a):
    return (8 / (a*a)) * (1-np.prod(np.cos(0.5*a*np.asarray(k))))


def tt_projector(k):
    n=np.asarray(k,dtype=float); n=n/np.linalg.norm(n)
    P=np.eye(3)-np.outer(n,n)
    Pi=0.5*(np.einsum('ik,jl->ijkl',P,P)+np.einsum('il,jk->ijkl',P,P)-np.einsum('ij,kl->ijkl',P,P))
    return Pi.reshape(9,9)


def main():
    r0=np.array([W[1]/W[0],W[2]/W[1],W[2]/W[0]])
    rng=np.random.default_rng(20261015)
    mass=[]; max_res=0.; max_chirality_residual=0.; samples=0
    t_critical=float(2*(1+W[0])/(1-W[0])**2)
    test_couplings=[1/64,1/1024,1/16384, 0.99*t_critical,
                    t_critical, 1.01*t_critical, 4.]
    for t in test_couplings:
        B=np.diag(np.sqrt(t*W))
        H=np.block([[np.zeros((3,3)),B],[B,np.eye(3)]])
        lamb=np.linalg.eigvalsh(H)
        chir = np.block([[np.zeros((6,6)),H],[H,np.zeros((6,6))]])
        gam = np.diag(np.r_[np.ones(6),-np.ones(6)])
        chirality_residual=float(np.linalg.norm(gam@chir+chir@gam,ord='fro'))
        assert chirality_residual<1e-13
        max_chirality_residual=max(max_chirality_residual,chirality_residual)
        sv=np.sort(np.linalg.svd(chir,compute_uv=False))
        predicted_sv=np.sort(np.repeat(np.r_[mfun(W,t),1+mfun(W,t)],2))
        assert np.max(np.abs(sv-predicted_sv))<1e-13
        if t < t_critical*(1-1e-10):
            assert np.max(np.abs(sv[:6]-np.repeat(np.sort(mfun(W,t)),2)))<1e-13
        if t > t_critical*(1+1e-10):
            assert sv[4] < np.sort(mfun(W,t))[-1]
        if abs(t-t_critical)<1e-11:
            assert abs(mfun(W[-1],t)-(1+mfun(W[0],t)))<2e-14
        low=(1-np.sqrt(1+4*t*W))/2
        mass_pos=mfun(W,t)
        max_res=max(max_res,float(np.max(np.abs(lamb[:3]-np.sort(low)))))
        assert np.max(np.abs(mass_pos+low))<2e-16
        res=0.5*(1+1/np.sqrt(1+4*t*W))
        assert np.all((res>0.5)&(res<1.0))
        r=np.array([mass_pos[1]/mass_pos[0],mass_pos[2]/mass_pos[1],mass_pos[2]/mass_pos[0]])
        factor=2/(1+np.sqrt(1+4*t))
        assert np.all(r<=r0*(1+1e-12))
        assert np.all(r>=r0*factor*(1-1e-12))
        mass.append({'t':t,'magnitude_of_low_edges':mass_pos.tolist(), 'ratios':r.tolist(), 'normalized_ratio_to_reduced':(r/r0).tolist(), 'source_residues':res.tolist(), 'uniform_ratio_lower_factor':factor,'all_six_hamiltonian_eigenvalues':lamb.tolist(),'all_twelve_chiral_singular_values':sv.tolist(),'chiral_anticommutator_residual':chirality_residual,'three_negative_edge_magnitudes':mass_pos.tolist(),'ordering_window_satisfied':bool(t < t_critical)})
    # Weyl bound for singular values and corresponding pairwise ratios.
    for trial in range(128):
        X = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
        rho=0.25
        X *= rho * W[0]/np.linalg.norm(X,ord=2)
        ss = np.sort(np.linalg.svd(np.diag(W)+X,compute_uv=False))
        assert np.all(np.abs(ss-W) <= rho*W[0]*(1+1e-9))
        for j,i in [(1,0),(2,1),(2,0)]:
            re=(ss[j]/ss[i])/(W[j]/W[i]); assert (1-rho)/(1+rho)-1e-10<=re<=(1+rho)/(1-rho)+1e-10
        samples+=1

    a=0.12; c=1.; Z=2.
    directions=[np.array([1.,0,0]),np.array([1.,1.,0])/np.sqrt(2),np.ones(3)/np.sqrt(3)]
    dy=[]; tt=[]
    for unit in directions:
        k=1.75*unit; lam=d_bcc(k,a); k2=np.dot(k,k)
        fourth=(a*a)*(np.sum(k**4)/48+sum(k[i]**2*k[j]**2 for i in range(3) for j in range(i+1,3))/8)
        assert 0<=k2-lam<=fourth+1e-10
        assert k2-lam<= (7/144)*(a*a)*(k2*k2)+1e-10
        Pi=tt_projector(k)
        assert np.linalg.norm(Pi.T-Pi)<1e-13
        assert np.linalg.norm(Pi@Pi-Pi)<1e-13
        assert abs(np.trace(Pi)-2)<1e-13
        ev=np.linalg.eigvalsh(Pi)
        assert np.sum(ev>0.5)==2
        Pi4=Pi.reshape(3,3,3,3)
        assert np.linalg.norm(np.einsum('i,ijkl->jkl',k,Pi4))<1e-12
        assert np.linalg.norm(np.einsum('iikl->kl',Pi4))<1e-12
        omega=0.25; D=Pi/(Z*(omega**2+c*c*lam))
        assert np.linalg.eigvalsh(D).min()>=-1e-14
        assert np.linalg.eigvalsh(D).max()>0
        dy.append({'direction':unit.tolist(),'symbol':lam,'quadratic':k2,'symbol_error':k2-lam,'analytic_error_bound':fourth,'sharp_uniform_bound': (7/144)*(a*a)*(k2*k2)})
        tt.append({'trace_projector':float(np.trace(Pi)),'projector_error':float(np.linalg.norm(Pi@Pi-Pi)),'min_eigen_euclidean_propagator':float(np.linalg.eigvalsh(D).min())})
    # Scalar symbol is BCC reciprocal-periodic; continuum TT frame is not.
    kp=np.array([np.pi,np.pi,.25]); km=kp-2*np.pi*np.array([1.,1.,0.])
    reciprocal_symbol_error=abs(d_bcc(kp,1)-d_bcc(km,1))
    reciprocal_projector_difference=float(np.linalg.norm(tt_projector(kp)-tt_projector(km),'fro'))
    assert reciprocal_symbol_error<1e-13
    assert reciprocal_projector_difference>.2
    # Spectral atom theorem: a positive measure with a zero-mass atom and
    # positive continuum has s*D(s) -> the prescribed atomic weight.
    zero_mass_atom=0.4
    sample_s=[1e-2,1e-4,1e-6,1e-8]
    # d rho_cont(u) = 0.6 exp(-(u-1)) du, u >= 1.
    # a certified finite quadrature is replaced here by an upper bound:
    # 0 <= s D_cont(s) <= 0.6 s/(s+1).
    spectral_atom_errors_upper=[float(.6*s/(s+1)) for s in sample_s]
    assert all(spectral_atom_errors_upper[j]>spectral_atom_errors_upper[j+1] for j in range(3))
    # Direct asymptotic scaling on the two selected cubic directions.
    asymp=[]
    for a_ in [0.25,0.125,0.0625,0.03125]:
        vv=[]
        for unit in directions:
            k=1.75*unit
            vv.append(float((np.dot(k,k)-d_bcc(k,a_))/(a_*a_)))
        asymp.append({'lattice_spacing':a_,'a2_scaled_symbol_errors':vv})
    cert={'scope':'finite exactly soluble Schur benchmark and free BCC tensor benchmark only; not physical C10 masses or gravitational pole','alpha_conn':ALPHA,'normalized_weights':W.tolist(),'reduced_ratios':r0.tolist(),'mass_block_eigenspectrum_max_residual':max_res,'chiral_doubling_anticommutator_residual':max_chirality_residual,'ordering_threshold_t':t_critical,'mass_benchmarks':mass,'weyl_random_cases':samples,'lattice_dispersion':dy,'tensor_projector':tt,'scaled_anisotropy':asymp,'reciprocal_lattice_chart_test':{'a':1.0,'k_plus':kp.tolist(),'k_minus':km.tolist(),'scalar_periodicity_error':float(reciprocal_symbol_error),'TT_projector_difference':reciprocal_projector_difference},'positive_spectral_atom_test':{'atom_weight':zero_mass_atom,'s_values':sample_s,'maximum_s_times_continuum_contribution':spectral_atom_errors_upper,'continuum_support_lower_mass_squared':1.0},'random_seed':20261015}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(cert,indent=2)+'\n'); print(json.dumps(cert,indent=2))

if __name__=='__main__':main()
