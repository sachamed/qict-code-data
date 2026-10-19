"""Independent solver/grid convergence checks on the explicitly selected toy models."""
import json
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.special import kn
from scipy.integrate import cumulative_trapezoid
from pathlib import Path
root=Path(__file__).parent
old=json.loads((root/'results.json').read_text())
ref=json.loads((root/'closure_results.json').read_text())
sv=ref['freezeout']['swave_GeV_minus2']; Mpl=1.22089e19; m=300.;gs=90.
coef=(2*np.pi*np.pi/45)/1.66*np.sqrt(gs)*Mpl*m*sv
Yeq=lambda x:45/(4*np.pi**4)/gs *x*x*kn(2,x)
from numerical_methods import constant_rate_freezeout
def integrate(rtol, step, x0=8):
 return constant_rate_freezeout(coef,gstar=gs,x0=x0,rtol=rtol,max_step=step)['Y_infinity']
y1=integrate(3e-7,.12); y2=integrate(1e-8,.05); y3=integrate(3e-9,.02)
assert max(abs(y-y3)/y3 for y in (y1,y2))<4e-4
assert abs(y2-ref['freezeout']['Y_final'])/y2<2e-4
# non-identifiability witness exact n moment orthogonality
N=6; Z=np.exp(2j*np.pi*np.arange(N+1)/(N+1))
assert all(abs(np.mean(Z**n))<1e-13 for n in range(1,N+1))
# SI rate scales coupling squared, width scales yield linearly
r=ref['direct_detection']
assert np.isclose((r['portal_coupling_max_given_model']/.01)**2*old['dark_matter_benchmarks']['SI40_cm2'],r['LZ_4p2ty_40_GeV_SI_90pct_cm2'],rtol=1e-13)
assert abs(ref['freezein']['width_for_observed_Omegac_h2_GeV']*old['dark_matter_benchmarks']['freezein_omega_h2']/old['dark_matter_benchmarks']['freezein_width_GeV']-.12)<1e-12
out={'freezeout_three_resolutions_yield':[y1,y2,y3], 'freezeout_max_fractional_spread':max(abs(y-y3)/y3 for y in (y1,y2)), 'spectral_moment_N':N,'pass':True}
(root/'numerical_convergence.json').write_text(json.dumps(out,indent=2))
print('PASS',json.dumps(out))
