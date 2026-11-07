"""Small-amplitude finite-N rotor dark-matter Jeans diagnostics.

Analytical perturbative comparison. A positive quadratic tail with w=0
is not necessarily gravitationally cold on galactic scales. This uses
the *late-time small-amplitude* scalar curvature and the standard
oscillatory scalar fluid dispersion, not a new full LSS likelihood.
"""
import json,math
from pathlib import Path

N=250
F=0.225
V0=201.55504807115258 # Native CLASS scalar potential units: Mpl² Mpc^-2
H0=74.93
OMEGA_M=.2535753809882664
MPC_INV_IN_EV=6.394e-30 # hbar*c / Mpc
COEFF_X=4/N**2
M2=V0*COEFF_X/(F**2)
M=math.sqrt(M2)
H=H0/299792.458
# For dilute quadratic canonical axion field, kJ^4=16*pi*G*rho_m*m²*a^4.
# rho_m=3*H²*Omega_m/(8*pi*G), today a=1 -> kJ=(6*Omega_m)^1/4 sqrt(H*m).
K_J=(6*OMEGA_M)**.25*math.sqrt(H*M)
Ks=(.01,.05,.1,.2,1.0)
CS2={str(k):k*k/(k*k+4*M*M) for k in Ks}
# Fractions of total scalar mass modulation do not imply an odd-sector mass.
g=.02
result={"status":"passed_analytic","scope":"standard WKB dilute quadratic scalar; self-interaction and full eigenmode coupling neglected",
  "N":N,"V0_in_Mpl2_per_Mpc2":V0,"F_over_Mpl":F,
  "m_phi_comoving_unit_Mpc_inverse":M,"m_phi_eV":M*MPC_INV_IN_EV,
  "H0_Mpc_inverse":H,"m_over_H0":M/H,
  "Jeans_k0_Mpc_inverse":K_J,"Jeans_wavelength_2pi_over_k_Mpc":2*math.pi/K_J,
  "effective_sound_speed2_at_k_Mpc_inverse":CS2,
  "maximum_chi_mass_modulation_fraction":math.sqrt(1+g)-1,
  "interpretation":"The scalar tail cannot be equated with freely clustering cold dark matter from w=0 alone; odd chi production and clustering require independent microscopic work.",
  "not_inferred":["relic abundance","odd particle mass","CMB+LSS fit","vacuum value"]}
assert M>0 and K_J<.02 and CS2["0.1"]>.009 and CS2["1.0"]>.4
p=Path("results/qict_scalar_jeans_gate/jeans_gate.json")
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(result,indent=2)+"\n")
print("QICT_FINITE_CHARGE_JEANS_GATE",json.dumps(result),flush=True)
