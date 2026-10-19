"""Independent regression checks for analytic domains and corrected observables."""
from pathlib import Path
import json, math
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import roots_genlaguerre, roots_legendre
from compute_operator_closure import fejer_atom_upper, inverse_toeplitz_atom_upper
from numerical_methods import constant_rate_freezeout
import compute_portal_thermal as portal

P=Path(__file__).resolve().parents[1]
checks={}

# A unitary block need not have self-adjoint off-diagonal blocks.
angle=.61; U=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
z=1.7+.2j; A,B,C,D=U[0,0],U[0,1],U[1,0],U[1,1]
direct=np.linalg.inv(z*np.eye(2)-U)[0,0]
assert abs(direct-1/(z-A-B*C/(z-D)))<1e-14
assert abs(direct-1/(z-A-B*B.conjugate()/(z-D)))>.01
checks['unitary_feshbach_independent_blocks']='PASS'

# Noncommuting matrix moments at a nonzero phase test both Toeplitz conventions.
rng=np.random.default_rng(8421);Q,_=np.linalg.qr(rng.normal(size=(18,18))+1j*rng.normal(size=(18,18)))
phases=np.linspace(-2.8,2.6,18);W=Q@np.diag(np.exp(1j*phases))@Q.conj().T
V=np.eye(18,dtype=complex)[:,:2];mom=np.array([V.conj().T@np.linalg.matrix_power(W,n)@V for n in range(7)])
theta=phases[8];u=V.conj().T@Q[:,8];atom=np.outer(u,u.conj())
last=None
for n in range(2,8):
    b,_,_=inverse_toeplitz_atom_upper(mom[:n],theta)
    f,_=fejer_atom_upper(mom[:n],theta)
    assert np.linalg.eigvalsh(b-atom).min()>-2e-10
    assert np.linalg.eigvalsh(f-b).min()>-2e-10
    if last is not None:assert np.linalg.eigvalsh(last-b).min()>-2e-10
    last=b
checks['noncommuting_nonzero_phase_and_nested_bounds']='PASS'

r=json.loads((P/'code/closure_results.json').read_text());f=r['freezeout']
K=(2*np.pi**2/45)/1.66*np.sqrt(90)*1.22089e19*300*f['swave_GeV_minus2']
ys=[constant_rate_freezeout(K,xf=x)['Y_infinity'] for x in [2000,4000,8000]]
spread=max(abs(y/ys[-1]-1) for y in ys)
assert spread<3e-8
assert f['post_endpoint_tail_fraction']>.005
checks['freezeout_endpoint_independence']={'relative_spread':spread,'yields':ys}

# Angular and kinetic quadrature refinement tests the newly evaluated amplitude.
original=portal.average(20)
portal.COSTH,portal.ANG_WEIGHTS=roots_legendre(80)
portal.KIN,portal.KIN_WEIGHTS=roots_genlaguerre(80,.5)
refined=portal.average(20)
assert abs(original/refined-1)<2e-8
hh,vis=portal.rates(1e16)
assert abs(float((hh+vis)[0])/(portal.LAMBDA**2/(4*np.pi*1e16))-1)<2e-7
hh0,vis0=portal.rates(4*portal.M**2)
assert abs(portal.average(1e7)/float((hh0+vis0)[0])-1)<2e-6
checks['portal_refined_quadrature_and_high_energy_limit']={'x20_relative_change':abs(original/refined-1)}

# Independent dimensionless top-hat equation includes radiation through H(a).
base=json.loads((P/'code/results.json').read_text());inp=base['inputs']
om,orr,ol=inp['Omega_m'],inp['Omega_r'],inp['Omega_Lambda'];ai=.01;delta=.04
rho=base['halo']['rho_m_Msun_Mpc3'];rlag=(3e14/(4*np.pi*rho))**(1/3)
def rhs(t,y):
    a=np.exp(t);e2=om/a**3+orr/a**4+ol
    friction=2-(3*om/a**3+4*orr/a**4)/(2*e2)
    return [y[1],-friction*y[1]-.5*(om/a**3)/e2*(y[0]**-3-1)*y[0]]
def event(t,y):return np.exp(t)*rlag*y[0]-.01
event.terminal=True;event.direction=-1
y=(1+delta)**(-1/3)
s=solve_ivp(rhs,(np.log(ai),0),[y,-y*delta/(3*(1+delta))],events=event,max_step=.003,rtol=2e-10,atol=1e-12)
ac=float(np.exp(s.t_events[0][0]));reference=base['spherical_top_hat']['collapse_scale_factor']
assert abs(ac/reference-1)<2e-8
bg=solve_ivp(rhs,(np.log(ai),0),[1.,0.],rtol=1e-10,atol=1e-12)
assert np.max(abs(bg.y[0]-1))<1e-14
checks['spherical_collapse_independent_equation']={'a_collapse':ac,'background_identity':'PASS'}

# Source contact curvature by two real Hermitian perturbations (Wirtinger pair).
with np.load(P/'code/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz') as d:
    h0,kq,km,k2=[d[k] for k in ['K0','K1_q','K1_mq','K2_qmq']]
    e0=np.linalg.eigvalsh(h0)[0]
    def energy(x,y):return np.linalg.eigvalsh(h0+x*(kq+km)+1j*y*(kq-km)+(x*x+y*y)*k2)[0]
    eps=.0002
    finite=(energy(eps,0)+energy(-eps,0)+energy(0,eps)+energy(0,-eps)-4*e0)/(4*eps**2)
metric=json.loads((P/'data/RESPONSE_AND_REHEATING_RESULTS.json').read_text())['gravity_finite_source']
assert abs(finite-metric['energy_mixed_curvature'])<1e-7
checks['finite_kato_contact_curvature']={'finite_difference':float(finite),'spectral':metric['energy_mixed_curvature']}

growth=json.loads((P/'data/metric_growth_galaxy.json').read_text())['growth']
assert abs(growth['angular_convergence_B_over_GR_equal_initial_density']['200']-growth['equal_time_Weyl_power_B_over_GR_z0p5'])>.001
checks['angular_and_equal_time_observables_distinct']='PASS'

out={'checks':checks,'all_pass':True,'scope':'Analytic identities and numerical consistency in the explicitly declared finite/effective models.'}
(P/'data/CONSISTENCY_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
