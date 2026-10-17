#!/usr/bin/env python3
"""C11 Gauss-reduced tangent edge-curvature invariant on the BCC sector.

This certificate computes a parameter-free C11 edge-curvature coefficient in the
explicitly defined Gauss-reduced tangent channel, not a full fermion mass.  It
combines the C11 spectral-measure translation theorem with the C10 quadratic Gauss
minimizer.  A continuous source amplitude a is used only as a
linear-response coordinate in the tangent U(1) Gauss functional; the compact charge
representation itself is not deformed.

For charge q and finite BCC graph-ball radius R,
    chi_q(R) = 2 phi_* q^2 R_eff(R),
where phi_*=pi/63 and R_eff is the unit BCC resistance.  Rayleigh monotonicity and
the exact Watson/Joyce limit give a one-sided certified convergence interval.
"""
from __future__ import annotations
import csv, json, math, hashlib
from pathlib import Path
from scipy.special import gamma

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'; CERT=ROOT/'certificates'
INP=DATA/'c10_bcc_gauss_coulomb_R1_R48.csv'
OUT=DATA/'c11_gauss_reduced_edge_curvature_convergence.csv'
COUT=CERT/'C11_GAUSS_REDUCED_EDGE_CURVATURE.json'
PHI=math.pi/63.0
Q_E=-6

def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

rows=[]
with INP.open(newline='') as f:
    src=list(csv.DictReader(f))
Rinf=float(gamma(0.25)**4/(32.0*math.pi**3))
chi_inf=2.0*PHI*(Q_E**2)*Rinf
# Exact simplification for q_e=-6: Gamma(1/4)^4/(28 pi^2)
chi_exact_simplified=float(gamma(0.25)**4/(28.0*math.pi**2))
assert abs(chi_inf-chi_exact_simplified)<5e-15
prev=-1.0
for r in src:
    R=int(r['R']); Re=float(r['R_eff_unit'])
    chi=2.0*PHI*(Q_E**2)*Re
    err=chi_inf-chi
    assert err>=-2e-14
    assert chi>prev
    prev=chi
    rows.append({
        'R':R,
        'vertices':int(r['vertices']),
        'R_eff_unit':Re,
        'chi_e_tangent':chi,
        'certified_upper_limit':chi_inf,
        'one_sided_abs_error_bound':max(0.0,err),
        'relative_error_bound':max(0.0,err)/chi_inf,
    })
with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# Electromagnetic tangent-response ratios in the manuscript's integer charge normalization.
charges={'electron':-6,'up_type':4,'down_type':-2}
ratios={name:(q*q)/(Q_E*Q_E) for name,q in charges.items()}
cert={
  'status':'PASS_EXACT_C11_GAUSS_REDUCED_EDGE_CURVATURE',
  'observable':'Half-curvature of the C11 support edge in the explicitly defined Gauss-reduced U(1)_em tangent channel',
  'definition':'In the Gauss-reduced tangent channel W_red(a)=exp[-i delta_q,R(a)] W_red(0), so C_red(n;a)=exp[-i n delta_q,R(a)] C_red(n;0) and the C11 support edge translates by delta_q,R(a); chi_q^E=(1/2) d^2 epsilon_edge_red/da^2 at a=0',
  'scope':'Exact parameter-free C11 spectral susceptibility in the explicitly defined Gauss-reduced tangent channel. The unreduced compact C8J product is noncommuting, so this certificate does not identify the susceptibility with the full physical mass edge or with a numerical fermion mass.',
  'phi_star':PHI,
  'electron_charge_integer':Q_E,
  'exact_R_BCC':Rinf,
  'exact_R_BCC_formula':'Gamma(1/4)^4/(32*pi^3)',
  'exact_chi_e':chi_inf,
  'exact_chi_e_formula':'Gamma(1/4)^4/(28*pi^2)',
  'electron_response_over_phi_star':chi_inf/PHI,
  'electromagnetic_response_ratios_to_electron':ratios,
  'charged_lepton_family_splitting_response':'zero for the Gauss-reduced tangent edge curvature because q_e=q_mu=q_tau=-6 and the C10 quadratic dressing is family blind',
  'c11_spectral_translation_identity':'C_red(n;a)=exp[-i*n*delta(a)] C_red(n;0), hence mu_red(a) is the circle translation of mu_red(0) and every continuously tracked support edge shifts by delta(a)',
  'largest_finite_radius':rows[-1]['R'],
  'largest_finite_value':rows[-1]['chi_e_tangent'],
  'largest_radius_one_sided_abs_error_bound':rows[-1]['one_sided_abs_error_bound'],
  'largest_radius_relative_error_bound':rows[-1]['relative_error_bound'],
  'convergence_statement':'The Gauss-reduced C11 edge curvature chi_e(R) increases monotonically under BCC Dirichlet domain exhaustion and is bounded above by the exact Watson/Joyce limit; therefore the displayed difference is a certified one-sided finite-volume error bound.',
  'external_particle_mass_data_used':False,
  'external_mixing_data_used':False,
  'new_free_dimensionless_parameters':0,
  'physical_mass_claimed':False,
  'full_compact_G2_edge_claimed':False,
  'input_sha256':{'data/c10_bcc_gauss_coulomb_R1_R48.csv':sha(INP),'code/closure/verify_c11_gauss_reduced_edge_curvature.py':sha(Path(__file__))}
}
COUT.write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
