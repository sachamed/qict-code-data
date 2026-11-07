# Charge-Resolved Receiver Cosmology: Exact Symmetry Obstructions and a Minimal Parity-Safe Dark-Sector Completion

**Research memorandum — 7 November 2026**  
**Repository:** [sachamed/qict-code-data](https://github.com/sachamed/qict-code-data)  
**Numerical status:** Exact symmetry, scale, and adiabaticity gates passed on GitHub Actions. The proposed parity-safe two-field action has **not** undergone a two-field Einstein–Boltzmann likelihood fit. This document distinguishes proven statements, conditional constructions, and untested phenomenological implications.

## Abstract

Finite-charge receiver Hamiltonians provide nontrivial exact coherent-state identities and a calculable parity-resolved excitation spectrum. Their cosmological interpretation nevertheless encounters three distinct structural questions: whether the same discrete reflection can both preserve a stable dark excitation and support two independently active early-energy phases; whether cosmological scalar normalization fixes an absolute odd-sector mass; and whether an internal spectral ground state fixes the gravitationally renormalized vacuum energy. We establish a conditional no-go theorem for the first question within the natural three-channel phase realization and demonstrate continuous microscopic-scale and identity-counterterm ambiguities for the latter two. A consistent extension with a *separate central dark parity* is then constructed and checked, including a positive field-space metric and a derivative dark coupling that does not generate a background amplitude-dependent fermion-mass determinant by construction. A quantitative adiabaticity bound shows why the small, smooth EDE-era mass variation previously implemented cannot produce a conventional massive relic from the vacuum. The observational status is independently characterized by physically calibrated single-field QICT likelihood runs, which contain a competitive fixed point near H0=70 km s^-1 Mpc^-1 but no converged solution of the local-expansion discrepancy. These results define a compact, falsifiable completion problem: derive the independent parity, lattice scale, vacuum selection and relic-production process from receiver primitives, and test the full two-field perturbations against the same cosmological data and separately optimized LambdaCDM reference.

## 1. Minimal finite-charge input

Consider N>=3 two-flavor bosonic charge quanta. Let K=N/2-J_x count conversion-misaligned constituents, with K=0,...,N. The **distinct-triplet** local operator is

\[
H_{\rm tri}/u={8K(K-1)(K-2)\over N(N-1)(N-2)}.
\]

For a spin-coherent phase theta,

\[
\langle H_{\rm tri}\rangle/u=(1-\cos\theta)^3 .
\]

The ordinary-power parent is instead H_pow/u=(2K/N)^3, with the different exact coherent expectation

\[
P_N(\theta)={4x\over N^2}
 +{6(N-1)x^2\over N^2}
 +{(N-1)(N-2)x^3\over N^2},\qquad
x=1-\cos\theta .
\]

The two operators are not identical at finite N, and neither coherent expectation is by itself a derived one-particle-irreducible quantum effective potential. The triple parent has three zero modes K=0,1,2 before inertia is added. A positive, flavor-exchange-invariant charge-inertia term can open a **dimensionless** odd–even energy gap. It does not determine the gap's absolute energy scale.

The triplet construction and its finite-charge alternatives have existing reproducible exact checks; the new results concern the gravitational and symmetry completion.

## 2. Exact three-channel relative-phase geometry

Assume, as a separate microscopic hypothesis, that three independent receiver channels carry phases and that their common phase is gauge redundant. The relative phase manifold is

\[
{\cal M}=U(1)^3/U(1)_{\mathrm{diag}}\simeq T^2.
\]

With coordinates q=(alpha,beta) and link angles (alpha,beta-alpha,-beta), the natural cyclic-invariant kinetic metric and finite-charge potential are

\[
G={1\over3}\begin{pmatrix}2&-1\\-1&2\end{pmatrix},\qquad
U_N(q)=\sum_{\ell=1}^3 P_N(\theta_\ell).
\]

The cyclic and reflection maps may be represented by

\[
M=\begin{pmatrix}-1&1\\-1&0\end{pmatrix},\qquad
R=\begin{pmatrix}0&1\\1&0\end{pmatrix},
\quad M^3=R^2=1,\ RMR=M^{-1}.
\]

The full group is the dihedral group D3 isomorphic to S3. Its order is six, and its center is trivial.

**Theorem 1 (one active even background).** The fixed subspace of any reflection in this irreducible two-dimensional representation has dimension one. Writing alpha=s+d and beta=s-d, the chosen reflection sends (s,d)->(s,-d). A homogeneous solution preserving this exact reflection must have d=dot(d)=0. Only s can then provide an independently rolling *background*. At d=0, the odd fluctuation has zero first-order contribution to the energy-momentum tensor; hence it cannot provide a second independent source to linear CMB perturbations.

**Proof.** Every reflection has eigenvalues +1 and -1. Its fixed subspace therefore has dimension one. In the eigenbasis, both the invariant potential and kinetic metric are even in d, giving dV/dd|d=0=0, mixed quadratic s–d terms zero, and delta T linear in the background-vanishing d mode zero. QED.

**Theorem 2 (transverse instability near the standard release angle).** Along d=0,

\[
U_N(s,d)=P_N(s+d)+P_N(s-d)+P_N(2d)
\]
\[
G_{sd}=\mathrm{diag}(2/3,2),
\]
\[
m_s^2={3V_0\over F^2}P_N''(s),\qquad
m_d^2={V_0\over F^2}\big[P_N''(s)+2P_N''(0)\big].
\]

For N=250, P_N''(0)=0.000064 and P_N''(2.5)=-3.931255531. Thus both effective squared masses are negative when the even field rolls off the conventional tachyonic flank. At the illustrative constant-H release criterion |m_s²|/H²=4, the odd tachyonic magnitude is approximately 1.33329 H², yielding an odd linear growth exponent 0.39296 per Hubble time. An exactly reflection-invariant background remains a mathematical solution, but it is transversely unstable against perturbations in this regime.

The conclusion is conditional on identifying the protective dark parity with this **same reflection** and on the chosen D3 link potential. It is not a general theorem against all QICT multiplets.

## 3. Smallest straightforward symmetry repair

Because Z(D3)={identity}, a *distinct central dark Z2* cannot be obtained as a nontrivial central element of this D3 action alone. One simple 12-element enlargement is D3 x Z2_dark. A bosonic singlet chi transforms as chi->-chi under the new parity, while the two phases q are even under Z2_dark, although they transform nontrivially under D3.

A positive-metric covariant effective action is

\[
\begin{split}
S=\int d^4x\sqrt{-g}\bigg[
&{M_{\rm Pl}^2R\over2}
-{F^2\over2}e^{-\kappa\chi^2/(2\Lambda^2)}
G_{ab}\partial_\mu q^a\partial^\mu q^b
-{1\over2}\partial_\mu\chi\partial^\mu\chi\\
&-V_0 U_N(q)-{m_0^2\over2}\chi^2
-{\lambda_\chi\over4}\chi^4-\rho_{\rm vac}
\bigg]+S_{\rm SM},
\end{split}
\]

with F,V0,m0,Lambda>0 and kappa,lambda_chi>=0. The scalar field-space metric is positive for every finite real chi because G is positive definite and the exponential never changes sign. At chi=0, dark parity is unbroken regardless of the D3 orientation of q; both q fields may now roll as independent backgrounds.

For a homogeneous trajectory X=F² G_ab dot(q)^a dot(q)^b/2>=0, expanding near chi=0 gives

\[
m_{\chi,\mathrm{eff}}^2=m_0^2+{\kappa\over\Lambda^2}X>0.
\]

An odd state is protected from decay into strictly parity-even particles, provided this is the exact full symmetry of the quantum theory and the state is the lightest odd excitation. This action is a **consistent EFT proposal**, not a microscopic derivation of the added central Z2 or of the observed dark-matter abundance. An independently odd scalar does not imply fermionic spin-statistics.

The derivatively modulated q metric avoids giving chi a static q-dependent rest mass by hand. This removes the particular tree-level origin of a phase-dependent heavy-mass determinant but does not establish full quantum radiative protection. D3 invariance permits lower harmonics, such as sum_l(1-cos theta_l), so a sextic-only effective potential also requires a microscopic or renormalization argument beyond D3.

An alternative to an independent dark parity is a receiver construction with at least **four** phase-bearing channels, provided the parity acts by one transposition: after removing the common phase, there are n-2 parity-even relative coordinates and one odd coordinate. Whether this has an independently derived QICT primitive remains open.

## 4. Two independent scale obstructions

**Theorem 3 (local mass scale).** The conditional rotor dictionary

\[
F^2={1\over4\Delta\ell^3},\qquad
V_0={u\over\ell^3},\qquad
v_R^2=2tN\Delta\ell^2=1
\]

does not determine the microscopic length ell. For *every* ell>0 the choices

\[
\Delta={1\over4F^2\ell^3},\quad
t={2F^2\ell\over N},\quad
u=V_0\ell^3
\]

produce the same F,V0 and propagation speed. A local parity gap DeltaE=u delta(lambda) therefore varies as ell³. Moreover, a particle rest mass is not established until a relativistic one-particle dispersion and normalized collective state have been derived. A copy-time-to-spacetime conversion law fixing ell would supply a missing independent physical condition.

**Theorem 4 (gravitational vacuum).** For any finite receiver Hamiltonian H respecting its symmetries, H'=H+c u I has identical eigenvectors, commutation relations, parity assignments and energy gaps. The local energy density shifts by c u/ell³=c V0. Thus the absolute gravitational vacuum energy is not inferred from those symmetries and gaps alone. A renormalized vacuum selection principle must fix the allowed identity counterterm and its radiative stability.

Both theorems are exact in the stated model. They identify what a stronger foundational QICT theory must add; they are not impossibility statements about all ultraviolet completions.

## 5. Adiabatic bound on EDE-era massive particle production

For the previous phenomenological mass-modulation ansatz

\[
m_\chi=m_0\sqrt{1+g\sin^2(\theta/2)},\qquad g\ge0,
\]

one has |d ln(m_chi)/d theta|<=g/4. A nonnegative canonical scalar energy satisfies F² dot(theta)²/2<=3 Mpl² H² f_phi. Hence

\[
\boxed{
\left|{\dot m_\chi\over m_\chi^2}\right|
\le {g\over4}\sqrt{6f_\phi}{M_{\rm Pl}\over F}{H\over m_0}.
}
\]

For the previously computed high-EDE CLASS point, H(z=3500)=39.322 Mpc^-1≈2.5142e-28 eV, F/Mpl=.225, f_phi=.10635 and g=.02. The right-hand side is 4.4631e-30 for m0=1 eV, and 4.4631e-39 for m0=1 GeV. Smooth mass modulation at this epoch is highly adiabatic for such masses. It therefore cannot **by itself** establish the production of a conventional heavy dark relic; reheating, freeze-in/out, an earlier transition, or an independently demonstrated scattering mechanism must be modeled and normalized.

This is only a conditional instantaneous nonadiabaticity bound for the specified mass function. It does not prohibit all possible nonthermal or thermal production processes.

## 6. What the data actually establish

The latest native CLASS + Cobaya **fixed-point** Planck 2018 Plik-lite TTTEEE + lowTT + lowEE + DESI DR2 runs identify one comparatively competitive region in the single-field finite-charge model:

| Configuration | z_peak | f_phi peak | H0 (km/s/Mpc) | total log L | Delta chi2 vs fixed LCDM |
|---|---:|---:|---:|---:|---:|
| LCDM fixed control | — | 0 | 67.4 | -530.239 | 0 |
| QICT, omega_c=.122, ns=.975 | 2029.60 | 0.03871 | 70.0 | -529.627 | -1.22 |
| QICT, omega_c=.126, ns=.985 | 2004.95 | 0.06896 | 71.0 | -592.392 | +124.31 |
| QICT, omega_c=.130, ns=.990 | 2000.35 | 0.09801 | 72.0 | -699.940 | +339.40 |

The redshift and acoustic angle were reconstructed with real CLASS background integration at each point. This table is a **conditional slice**, not a global likelihood minimum, model evidence, converged posterior, or a joint Planck–ACT–DESI–SH0ES solution. The 70.0 point shows a promising physical direction but does not itself remove the tension with the local distance ladder. It should not be extrapolated to H0=73.

This difficulty is consistent with the independently documented empirical fact that physical two-field EDE can alter high-l CMB responses relative to one-field models. The multiple-field mechanism reported by Bella, Poulin, Vagnozzi and Knox (arXiv:2604.13535, 2026) is **prior literature**. QICT novelty, if established, would be a first-principles receiver origin and parameter relation, not merely proposing a second axion-like field. The 2026 Planck–ACT–DESI EDE analysis of Poulin et al. (Phys. Rev. D 113, 063519) also shows that dataset choices and profile-prior effects matter. A similarly calibrated likelihood pipeline is essential.

## 7. The next falsifiable inference

The proposed D3 x Z2dark effective theory is now an explicit candidate for numerical testing. A scientifically decisive programme is:

1. Derive its full background and *coupled two-field gauge-consistent Einstein–Boltzmann perturbations* from the covariant action; include adiabatic/isocurvature initial conditions and stress-energy constraint tests.
2. Construct a shooting map in physically interpretable variables {z_peak, f_peak, phase orientation, relative activation delay} without assuming that the scalar acoustic angle determines high-l CMB peak heights.
3. Evaluate lensed TT/TE/EE and primary + lensing ACT DR6, DESI DR2 and growth data with appropriate nonlinear treatment. Reoptimize LambdaCDM on identical choices and report full profile/posterior convergence.
4. Independently derive the odd mass, primordial population process and elastic clustering from a microscopic scale-fixing condition; evolve dark abundance without setting present Omega_chi by hand.
5. Specify a gravitational vacuum renormalization prescription that eliminates the identity-shift ambiguity and stabilizes the observed late energy; otherwise retain rho_vac as a phenomenological input and do not call it an explanation.

The four mathematical obstructions above prevent a premature unified claim; the constructive action removes one of them **conditionally** and provides an unambiguous object that can be falsified. A positive result requires a truly competitive matched likelihood and successful dark-relic/vacuum calculations, not only a large H0 at fixed parameters.

## 8. Computational provenance and literature

- [108 exact reflection, instability and CMB-active-rank checks](https://github.com/sachamed/qict-code-data/actions/runs/42419255199).
- [164 constructive central-dark-parity and positive-metric EFT checks](https://github.com/sachamed/qict-code-data/actions/runs/42419565734).
- [30 microscopic length and vacuum-shift non-identifiability checks](https://github.com/sachamed/qict-code-data/actions/runs/42419710113).
- [102 canonical-energy dark-mass adiabaticity checks](https://github.com/sachamed/qict-code-data/actions/runs/42419843560).
- [Planck + DESI acoustic compensation scan](https://github.com/sachamed/qict-code-data/actions/runs/42418425583).
- Marco Bella, Vivian Poulin, Sunny Vagnozzi and Lloyd Knox, *Double the axions, half the tension: multi-field early dark energy eases the Hubble tension*, arXiv:2604.13535 (2026), https://arxiv.org/abs/2604.13535.
- Vivian Poulin, Tristan L. Smith, Rodrigo Calderón and Théo Simon, *Impact of ACT DR6 and DESI DR2 for early dark energy and the Hubble tension*, Phys. Rev. D **113**, 063519 (2026), https://doi.org/10.1103/bx25-1g5d; arXiv:2505.08051.
- DESI Collaboration, *DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints* (2025), arXiv:2503.14738.
