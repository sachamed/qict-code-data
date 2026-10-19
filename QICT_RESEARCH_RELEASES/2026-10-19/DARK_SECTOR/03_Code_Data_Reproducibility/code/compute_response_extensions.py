"""Finite Kato derivatives and finite-reheating decay integrals from stated inputs."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.special import k1

ROOT=Path(__file__).resolve().parent


def main():
    with np.load(ROOT/'SPECTRAL_GRAVITY_SELFTEST_INPUT.npz') as z:
        h=z['K0'];E,V=eigh(h);i=int(z['selected_index']);u=V[:,i]
        assert i==0 and np.max(abs(z['K1_q'].conj().T-z['K1_mq']))<1e-14
        a=V.conj().T@z['K1_q']@u;b=V.conj().T@z['K1_mq']@u
        ids=np.arange(len(E))!=i;gaps=E[ids]-E[i]
        weights=abs(a[ids])**2+abs(b[ids])**2
        response=float(np.sum(weights/gaps))
        contact=float(np.vdot(u,z['K2_qmq']@u).real)
        curvature=contact-response
        volume=float(np.vdot(u,z['K1_vol']@u).real)
        cells=float(z['N_cells'])
    fi=[]
    for x in [.001,.1,1.,10.]:
        value,error=quad(lambda t:t**3*k1(t),x,np.inf,epsabs=1e-12,epsrel=1e-12)
        fi.append({'x_reheat':x,'yield_fraction_of_high_T_limit':value/(1.5*np.pi),
                   'quadrature_error_estimate':error/(1.5*np.pi),
                   'missing_fraction_upper_small_x_bound':min(1.,2*x**3/(9*np.pi))})
    out={'gravity_finite_source':{
        'ground_gap':float(gaps.min()),'connected_static_D':response,
        'contact':contact,'energy_mixed_curvature':curvature,
        'connected_D_per_cell':response/cells,'contact_per_cell':contact/cells,
        'energy_mixed_curvature_per_cell':curvature/cells,
        'volume_energy_derivative':volume,'volume_derivative_per_cell':volume/cells,
        'epsilon_if_dlogvolume_dJ_equals4':volume/(4*cells),
        'momentum_qell':.125,'N_cells':cells,
        'physical_continuum_coefficients':'These finite source derivatives do not identify continuum epsilon_0 or gamma_R.'},
        'finite_reheating_freezein':fi}
    (ROOT.parent/'data/RESPONSE_AND_REHEATING_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
