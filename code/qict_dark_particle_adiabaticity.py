#!/usr/bin/env python3
"""Finite-charge receiver dark-mass adiabaticity bound near EDE release.

Given canonical energy positivity and m_chi=m0 sqrt(1+g sin²(theta/2)),
bound |dot m|/m² for rest frame using observed native CLASS background H.
The result is only about vacuum nonadiabaticity of mass variation, not
thermal freeze-out, scattering, parametric resonance, or primordial DM.
"""
import json,math
from pathlib import Path

G=.02
F=0.225    # F/Mpl from native CLASS QICT model
F_PEAK=.10634829861864978 # high-energy CLASS f_phi at z=3500
H_MPC=39.32200692272184
MPC_INV_EV=6.394e-30
H_EV=H_MPC*MPC_INV_EV

def ln_mass_theta(theta,g):
    return g*math.sin(theta)/(4*(1+g*math.sin(theta/2)**2))

def energy_velocity_bound(f,F):
    if f<0 or f>1 or F<=0:raise ValueError("0<=f<=1 and F>0")
    # dot(phi)^2/2 <= rho_phi=3 Mpl² H² f.
    return math.sqrt(6*f)/F

def adiabaticity_upper(m_ev):
    if m_ev<=0:raise ValueError("m_ev must be positive")
    return (G/4)*energy_velocity_bound(F_PEAK,F)*H_EV/m_ev

def main():
    tests=0
    for th in [i*math.pi/100 for i in range(101)]:
      val=abs(ln_mass_theta(th,G))
      assert val<=G/4+1e-15;tests+=1
    p=energy_velocity_bound(F_PEAK,F)
    assert p>0
    masses=[1.e-28,1.e-25,1.e-20,1.e-12,1.,1.e3,1.e9]
    rows=[{"rest_mass_eV":m,"upper_bound_abs_mdot_over_m2":adiabaticity_upper(m),
           "adiabaticity":"bound_below_one" if adiabaticity_upper(m)<1 else "not_certified"}
          for m in masses]
    assert rows[-1]["upper_bound_abs_mdot_over_m2"]<1e-35;tests+=1
    bound={
     "status":"passed_conditional_adiabaticity_bound",
     "tests":tests,
     "assumptions":"canonical scalar, nonnegative V, f_phi=0.106348 and F/Mpl=.225 at z=3500, smooth QICT mass m=m0 sqrt(1+g sin²(theta/2)), g=.02",
     "derivation":"1/2 F² dot(theta)² <= 3 Mpl² H² f_phi; |d ln m/dtheta| <=g/4",
     "analytic_inequality":"|dot(m)|/m² <= (g/4)*sqrt(6 f_phi)*(Mpl/F)*(H/m0)",
     "H_at_z3500_inv_Mpc":H_MPC,"H_eV":H_EV,
     "upper_dot_theta_over_H":p,
     "maximum_rest_mass_variation_fraction":math.sqrt(1+G)-1,
     "bound_rows":rows,
     "interpretation":"Vacuum production of massive chi quanta by smooth mass modulation at this late EDE episode is strongly adiabatic for m0 >> H, under specified coupling. A separate production epoch or process is needed for standard massive relic DM.",
     "not_proved":["relic abundance zero","no thermal/nonthermal mechanisms exist",
                    "odd mass value","all possible interactions","Hubble tension solved"]
    }
    path=Path("results/qict_dark_particle_adiabaticity/adiabaticity_bound.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(bound,indent=2)+"\n")
    print("QICT_DARK_ADIABATICITY_BOUND",json.dumps(bound),flush=True)

if __name__=="__main__":main()
