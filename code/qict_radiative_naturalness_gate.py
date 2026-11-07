"""QICT dark-sector radiative naturalness gate: amplitude coupling vs phase-only.

Illustrative one-loop Dirac determinant in MSbar at mu=m0.
Not a renormalization-invariant exclusion; finite counterterms or protected
partner sectors can alter the effective scalar potential.
"""
import math,cmath,json
from pathlib import Path

G=.02
MPL_EV=2.435e27
MPC_INV_EV=6.394e-30
H_3500=39.32200692272184 * MPC_INV_EV
FPEAK=.10634829861864978
RHO_EDE_EV4=FPEAK*3*(MPL_EV*H_3500)**2
H0_EV=(67.4/299792.458)*MPC_INV_EV
RHO_DE_EV4=.685*3*(MPL_EV*H0_EV)**2

def mass_amp(theta,m0ev,g=G):
    return m0ev*math.sqrt(1+g*math.sin(theta/2)**2)

def one_loop_magnitude(theta,m0ev,mu=None,g=G):
    if mu is None:mu=m0ev
    m=mass_amp(theta,m0ev,g)
    return -(m**4)/(16*math.pi**2)*(math.log((m/mu)**2)-1.5)

def phase_mass_components(theta,m0ev,q=1.):
    # Scalar and pseudoscalar Dirac masses after a chiral phase rotation.
    return m0ev*math.cos(q*theta),m0ev*math.sin(q*theta)

def phase_mass_squared(theta,m0ev,q=1.):
    ms,mp=phase_mass_components(theta,m0ev,q)
    return ms**2+mp**2

def main():
    scale_abs=abs(one_loop_magnitude(math.pi,1.)-one_loop_magnitude(0.,1.))
    m0_bound_EDE=(RHO_EDE_EV4/scale_abs)**.25
    m0_bound_DE=(RHO_DE_EV4/scale_abs)**.25
    tests=0
    for theta in (0.,.2,1.,2.5,math.pi,4.):
        for m in (.01,1.,1000.,1e9):
            assert abs(phase_mass_squared(theta,m)-m*m)/m**2<4e-16
            tests+=1
    assert scale_abs>0 and RHO_EDE_EV4>RHO_DE_EV4
    assert m0_bound_EDE<5 and m0_bound_DE<.02
    tests+=3
    row={
      "status":"passed_analytic",
      "tests":tests,
      "model_old":"m_chi(theta)=m0*sqrt(1+g sin²(theta/2))",
      "model_candidate":"Dirac mass m0 exp(i gamma5 q theta); constant magnitude and chiral phase",
      "model_chiral_phase_one_loop_constant_background":"phase-independent Dirac determinant in the anomaly-free constant-phase limit",
      "model_chiral_phase_coupling":"(q/2F)partial_mu phi * bar(chi)gamma^mu gamma5 chi after field redefinition",
      "g_amplitude":G,
      "renormalization":"MSbar one Dirac fermion at mu=m0; scalar-independent counterterm subtracted",
      "delta_one_loop_coefficient_absolute_times_m0_eV_power4":scale_abs,
      "EDE_energy_density_eV4_from_native_background":RHO_EDE_EV4,
      "DE_energy_density_eV4_from_reference_Lambda":RHO_DE_EV4,
      "unprotected_mass_naturalness_threshold_EDE_eV":m0_bound_EDE,
      "unprotected_mass_naturalness_threshold_DE_eV":m0_bound_DE,
      "unprotected_GeV_mass_loop_to_EDE_ratio":scale_abs*(1e9**4)/RHO_EDE_EV4,
      "epistemic_warning":"Naturalness indicators only; finite counterterms or a new protecting symmetry may cancel. Phase-only mass removes scalar-dependent perturbative determinant but also removes variable-magnitude background energy exchange and is NOT a fitted cosmology.",
      "new_physics_required":["exact dark parity","anomaly cancellation or controlled nonperturbative breaking","odd-sector relic production","separate mechanism for late-time dark energy"]
    }
    p=Path("results/qict_radiative_naturalness/radiative_audit.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(row,indent=2)+"\n")
    print("QICT_RADIATIVE_GATE",json.dumps(row),flush=True)

if __name__=="__main__":main()
