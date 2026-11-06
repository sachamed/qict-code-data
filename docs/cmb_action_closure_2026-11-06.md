# QICT receiver closure and CMB inference boundary (6 November 2026)

The appended Symmetry manuscript section and Online Resource 5 in the local submission ZIP distinguish algebraic closure from complete cosmological inference.

## Derivable conditions
Let E=H/H0, q=Hc/H0, x=E/q, f(E)=lambda*x/(1+x*x), and F(E)=E²(1-f(E)). Then
F'(E)=E[2-lambda*g(x)] with g(x)=x(3+x²)/(1+x²)².
The unique global maximum of g is attained at x²=2 sqrt(3)-3. Consequently lambda < 2/max(g) is a sufficient condition that the positive Friedmann branch is unique, with monotonically decreasing E(a) for standard matter and radiation.

For separately conserved r_R=Omega_floor+E²f(E), dr_R/dln(a) <=0 and w_R>=-1. A canonical scalar exists along the cosmological trajectory provided the kinetic term is nonnegative:
(d phi/dln a)²/M_Pl² =3 Omega_R(1+w_R), V=(1-w_R)rho_R/2.
This does **not** prove that the QICT discrete receiver uniquely determines the scalar action or the off-trajectory potential.

For ds²=a²[-(1+2Psi)d eta²+(1-2Phi)d x²] and canonical phi,
delta rho=(phi' delta phi'-phi'² Psi)/a²+V_phi delta phi,
delta p=(phi' delta phi'-phi'² Psi)/a²-V_phi delta phi,
(rho+p)theta=k² phi' delta phi/a² and Pi_phi=0.
The decoupled field perturbation satisfies
delta phi''+2Hconformal delta phi'+(k²+a²V_phiphi)delta phi=phi'(Psi'+3Phi')-2a²V_phi Psi.

For parity-odd cold particles with m_eff²=m_s²+g_s²phi², beta=d ln(m_eff)/dphi and the consistent covariant exchange is
nabla_mu T_c^{mu nu}=-beta rho_c nabla^nu phi, with opposite receiver exchange.
The homogeneous system obeys dot(rho_c)+3Hrho_c=beta rho_c dot(phi) and ddot(phi)+3Hdot(phi)+V_phi=-beta rho_c.
These equations must be evolved jointly; an independently inserted drag violates closure.

## Likelihood selections
Reference: Planck 2018 Plik TTTEEE + low-T Commander + low-E SimAll + standalone Planck lensing.
ACT-only lensing check: Planck primary + ACT DR6 act_baseline, without standalone Planck lensing.
Joint lensing check: Planck primary + ACT DR6 actplanck_baseline, without standalone Planck lensing.
Use primary/lensing correction settings recommended by the official ACT release and verify overlap covariances.

## Current evidence
Seven local algebraic/numerical checks passed. Archived fixed-parameter CAMB spectra and DESI/Union3 compressed fits predate this integration. Neither real Planck/ACT posterior sampling nor a QICT-specific perturbation-to-Boltzmann adapter has been executed. The configurations alone are **not** a likelihood fit.

Official sources:
- https://www.mdpi.com/journal/symmetry/instructions
- https://github.com/ACTCollaboration/act_dr6_lenslike
- https://doi.org/10.1051/0004-6361/201936386
