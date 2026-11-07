# Finite-Charge Receiver Cosmology: A Symmetry-Consistent Dark-Sector Reconstruction

## Abstract

We investigate which components of a finite-charge quantum receiver admit a mathematically controlled description as a pre-recombination scalar field, a stable dark excitation, and a late-time vacuum sector. Two distinct positive parent operators—an ordinary cubic power and a falling-factorial triple-event operator—have inequivalent finite-charge coherent potentials. The second is uniquely determined, among cubic charge polynomials, by the requirement that fewer than three distinct excitations cannot trigger the interaction; its expectation is exactly the familiar cubic axionlike potential for every integer charge \(N\ge3\). Flavor exchange defines an exact parity grading, and a charge-inertia term produces positive odd-sector gaps in explicitly diagonalized finite systems. These results do not fix the absolute particle mass or abundance. In the previously tested high-energy scalar branch, native Einstein–Boltzmann solutions reproduce the angular acoustic scale at elevated \(H_0\) only with a low matter fraction, while physically reconstructed energy peaks at fixed cosmological parameters remain disfavored by Planck and DESI data. We also establish quantitative radiative-naturalness and scalar Jeans-scale screens and identify a phase-only dark fermion mass as a conditional symmetry-protected alternative. The resulting framework defines sharply separated microscopic and cosmological tests without promoting exploratory parameter points into claims of model selection.

## 1. Exact microscopic ordering

Let \(\mathcal H_N\) be the fixed-\(N\) Schwinger-boson space, \(j=N/2\), \(C=2J_x/N\), and \(K=j-J_x\). Since \(\operatorname{spec}(J_x)=\{-j,\ldots,j\}\), \(\operatorname{spec}(K)=\{0,\ldots,N\}\) and \(0\le K\le N\). On the balanced spin-coherent phase state \(|N,\theta\rangle\), the variable \(K\) is distributed as \(\mathrm{Binomial}(N,p)\), \(p=(1-\cos\theta)/2\).

**Ordinary-power parent.** The positive operator \(H_{\rm pow}=u(2K/N)^3\) has the exact coherent expectation
\[
\langle H_{\rm pow}\rangle/u=
\frac{4x}{N^2}+\frac{6(N-1)x^2}{N^2}
+\frac{(N-1)(N-2)x^3}{N^2},\qquad x=1-\cos\theta.
\]
Consequently the finite-\(N\) theory has quadratic, quartic and sextic small-field contributions. The continuum result \(u x^3\) is reached only as \(N\to\infty\).

**Distinct-triplet theorem.** Suppose a receiver event can occur only when at least three *distinct* constituents have been activated. Demand that \(H_N(K)\) be a polynomial of degree at most three, that \(H_N(0)=H_N(1)=H_N(2)=0\), and that \(H_N(N)=8u\). The polynomial is uniquely
\[
H_N(K)=\frac{8u}{N(N-1)(N-2)}K(K-1)(K-2).
\]
It is nonnegative on the integer spectrum. The factorial-moment identity
\(\langle K(K-1)(K-2)\rangle=N(N-1)(N-2)p^3\) gives
\[
\boxed{\langle H_N\rangle=u(1-\cos\theta)^3}
\]
for every integer \(N\ge3\). The exact identity and positivity were checked numerically across nine charges and six phases, together with the polynomial uniqueness constraints, in [GitHub Actions 42412981507](https://github.com/sachamed/qict-code-data/actions/runs/42412981507), passing 162 checks.

This is a theorem **conditional on the distinct-triplet primitive**. It does not establish that QICT itself uniquely supplies that primitive; this missing derivation is the central microscopic classification task. The resulting potential is a known \(n=3\) axionlike EDE functional form, so novelty lies in the finite-charge operator representation, not in introducing the cosine-cubed potential.

The distinct-triplet operator has a three-dimensional kernel, \(K=0,1,2\), and the state \(K=1\) is odd under constituent exchange. It therefore does not endow the odd state with a mass on its own.

## 2. Flavor-exchange parity and a candidate odd spectral gap

Define flavor exchange \(I:b_1\leftrightarrow b_2\). Then \(I J_z I=-J_z\) and \(I J_x I=J_x\). A compatible conditional Hamiltonian is
\[
\mathcal H_{N,\eta}/u=
\frac{8K(K-1)(K-2)}{N(N-1)(N-2)}
+\eta\left(\frac{J_z}{N}\right)^2,\quad \eta\ge0.
\]
It satisfies \([I,\mathcal H_{N,\eta}]=0\), so the finite matrix decomposes into even and odd sectors. In the 251-dimensional \(N=250\) representation, for \(\eta=1\), numerical diagonalization gives
\[
E_+/u=0.000149244894124,\quad
E_-/u=0.000577298343055,\quad
(E_--E_+)/u=0.000428053448931 .
\]
The parity commutator residual is \(4.44\times10^{-16}\). Positive gaps were observed for the finite grid \(N=10,20,40,100,250\), \(\eta=0.01,0.1,1,10\), not proven for every \(N,\eta\). Source and results: [GitHub Actions 42413121631](https://github.com/sachamed/qict-code-data/actions/runs/42413121631).

An exact unbroken parity can stabilize the *lightest* odd state in an even environment; it does not establish the particle's fermionic statistics, absolute mass, propagation speed, or relic density. In particular, \(\Delta E/u\) is dimensionless. A physical mass requires an independent conversion from the microscopic site energy \(u\), lattice spacing, and relativistic dispersion.

## 3. Why a pressureless late scalar is not automatically cold dark matter

In the ordinary-power model the finite-charge quadratic curvature at \(\phi=0\) is
\[
m_\phi^2=\frac{4V_0}{N^2F^2}.
\]
For the previously executed CLASS input \(N=250,F/M_{\rm Pl}=0.225,V_0/M_{\rm Pl}^2=201.55505\ {\rm Mpc}^{-2}\), this gives \(m_\phi=0.50478\ {\rm Mpc}^{-1}\simeq3.23\times10^{-30}\ {\rm eV}\). A rapidly oscillating dilute scalar can have averaged pressure \(w\simeq0\) but a scale-dependent effective sound speed
\[
c_s^2(k,a)\simeq\frac{k^2}{k^2+4m_\phi^2a^2}.
\]
The standard nonrelativistic Jeans estimate at \(a=1\) is \(k_J\simeq (6\Omega_m)^{1/4}\sqrt{H_0m_\phi}\), which for the tested point gives approximately \(0.0125\ {\rm Mpc}^{-1}\), corresponding to \(2\pi/k_J\simeq500\ {\rm Mpc}\). The assumptions of the dilute quadratic WKB treatment need checking against the full self-interacting scalar perturbation equations. These values are nevertheless incompatible with *identifying the entire late-time scalar tail as freely clustering galactic dark matter*. The separate parity-odd excitation remains the appropriate candidate. Analytic execution: [42412515608](https://github.com/sachamed/qict-code-data/actions/runs/42412515608).

## 4. Radiative stability distinguishes a coupling from a theory

For the previously implemented magnitude modulation
\[
m_\chi(\phi)=m_0\sqrt{1+g\sin^2(\phi/2F)},\qquad g=0.02,
\]
a single Dirac fermion contributes at one loop, in \(\overline{\rm MS}\),
\[
V_{1,\chi}(\phi)=-\frac{m_\chi^4(\phi)}{16\pi^2}
\left[\log\frac{m_\chi^2(\phi)}{\mu^2}-\frac32\right].
\]
At \(\mu=m_0\), the difference between \(\phi=\pi F\) and \(\phi=0\) has magnitude \(2.53286156\times10^{-4}m_0^4\). Using the native high-energy scalar branch as a benchmark, \(\rho_{\rm EDE}\simeq0.1196\ {\rm eV}^4\) and \(\rho_\Lambda\simeq2.518\times10^{-11}\ {\rm eV}^4\). Illustrative unprotected one-loop scales equal those densities at \(m_0\simeq4.66\ {\rm eV}\) and \(m_0\simeq0.0178\ {\rm eV}\), respectively. These are **naturalness diagnostics, not exclusion bounds**: renormalization, finite counterterms, additional particles and exact protecting symmetries change the conclusion. For a 1-GeV fermion with no cancellation, the illustrated one-loop variation is \(2.1\times10^{33}\) times the EDE density. The full sensitivity of interacting quintessence to heavy particle loops is well established (D'Amico, Hamill & Kaloper, PRD 94, 103526, 2016; M. C. D. Marsh, PLB 789, 639, 2019).

A conditional remedy replaces the magnitude modulation by a pure chiral phase,
\[
\mathcal L_{m,\chi}=-m_0\bar\chi e^{iq\gamma_5\phi/F}\chi .
\]
For a constant background with an anomaly-free chiral rotation, the squared Dirac mass remains \(m_0^2\), independent of \(\phi\). The chiral rotation rewrites the interaction as a derivative axial current,
\[
\mathcal L_{\rm derivative}=\frac{q}{2F}(\partial_\mu\phi)\bar\psi\gamma^\mu\gamma_5\psi ,
\]
up to the sign convention for the fermion rotation. This protects the **constant-background perturbative fermion determinant**. It does not protect against unspecified anomalous or explicit shift-symmetry breaking and it does not retain the old homogeneous variable-mass energy exchange. The comparison passed 27 algebraic checks on [GitHub Actions 42412724141](https://github.com/sachamed/qict-code-data/actions/runs/42412724141).

Thus the observed small likelihood gain from \(g=0.02\) is not sufficient to justify abandoning radiative stability. A protected derivative coupling deserves study only as a **different** microscopic realization, with a new independently computed cosmological likelihood.

## 5. Native acoustic and observed-likelihood findings

A six-model native CLASS audit [42412074540](https://github.com/sachamed/qict-code-data/actions/runs/42412074540) computed the actual background expansion and recombination quantities. The high-energy branch at \(H_0=74.93\) gives \(r_d=144.1325\ {\rm Mpc}\), \(100\theta_s=1.041574\), \(\Omega_m=0.253575\), and \(f_\phi(z=3500)=0.106348\), with realized \(H_0\) differing from its input by less than \(10^{-7}\) fractionally. The corresponding standard-CDM control at \(H_0=67.4\) gives \(r_d=147.1142\ {\rm Mpc}\), \(100\theta_s=1.040493\), and \(\Omega_m=0.313400\). Correct geometry alone does not ensure correct acoustic peak heights or BAO distances.

For the high-energy scalar point at \(H_0=74.93\), the previously executed matched Planck+DESI fixed-point comparison gave \(\Delta\chi^2\simeq+727.79\) relative to the separate reference \(\Lambda\)CDM point \(H_0=67.4\). An exploratory bounded optimization of the five parameters \((H_0,\omega_b,\omega_c,n_s,A_s)\) still recorded \(\Delta\chi^2\simeq+498.88\) between its best encountered points, with **neither** search converged.

A subsequent physically reconstructed scalar-peak experiment [42412842684](https://github.com/sachamed/qict-code-data/actions/runs/42412842684) matched the peak redshift and scalar fraction to three target pairs, using a continuous interpolation of the peak and a finite-difference shooting of the actual CLASS background. The resulting cases had:
- \((H_0,z_{\rm peak},f_{\rm peak})=(70.5,3399.55,0.05006)\), \(\Delta\chi^2\simeq+684.15\);
- \((71.5,3499.56,0.07009)\), \(\Delta\chi^2\simeq+1534.82\);
- \((73.04,3499.82,0.09508)\), \(\Delta\chi^2\simeq+3444.08\).

These are comparisons to one fixed \(\Lambda\)CDM input, not optimized evidence ratios. The corresponding acoustic parameters \(100\theta_s=1.04799,1.05093,1.05584\) visibly miss the Planck acoustic angle around 1.0411. The next [acoustic-anchored reconstruction](https://github.com/sachamed/qict-code-data/actions/workflows/qict-acoustic-anchored.yml) therefore **solves for** the physical peak redshift and observed acoustic angle, allowing the scalar fraction to emerge.

## 6. The vacuum cannot be inferred from zero microscopic ground-state energy

Both positive receiver parents have a zero-energy unexcited state in their isolated fixed-charge algebra. This alone does not calculate the renormalized gravitational cosmological constant: the absolute vacuum gravitation involves the full effective action and every coupled sector. The claimed \(N^{-4}\) relation between pre-recombination scalar density and late vacuum density is currently an unproved ansatz; its numerical closeness to selected EDE fractions is not evidence.

A viable late-time completion has to specify either a separately protected positive vacuum sector or a self-consistent late dynamical field. For a single canonical scalar with the positive periodic \(V_N\) whose minimum is zero, and no external energy source, \(\dot\rho_\phi=-3H\dot\phi^2\le0\). A field that has already entered rapid small-amplitude oscillations cannot dynamically regenerate a late-time positive cosmological constant without new physics. If it remains potential-dominated at late times, its trajectory must instead be demonstrated to satisfy the earlier EDE and CMB constraints.

## 7. Next high-value proof and numerical experiments

The most valuable mathematical step is to derive the **distinct-triplet condition itself** from the copy-time and receiver distinguishability primitive, rather than choosing the factorial operator to produce a desired scalar potential. The second is to show that relative-charge inertia, exchange parity and a relativistic continuum can be derived with fixed microscopic coefficients. Without fixing \(u\), \(N\), the spatial scale and production interactions, the parity gap cannot be converted into an absolute relic mass or abundance.

For cosmological inference, the **observed acoustic angle and physically located scalar peak** must jointly determine the allowed scalar potential region, followed by Planck TT/TE/EE, ACT–Planck DR6 lensing, DESI DR2, structure growth, and optionally SH0ES. Each benchmark must use the same neutrino and nuisance conventions as a separately optimized \(\Lambda\)CDM reference. Numerical convergence, initialization and gauge constraints are not dispensable.

The central result appropriate for publication is a classification and validation of exact finite-charge receiver operators and their dark-sector implications, together with observed-data limits on their minimal cosmological realization. Claiming a definitive three-problem cosmological solution would require additional theorems and an actual competitive joint fit.

## References

1. T. L. Smith, V. Poulin, M. A. Amin, "Oscillating scalar fields and the Hubble tension: A resolution with novel signatures," *Phys. Rev. D* **101**, 063523 (2020), https://doi.org/10.1103/PhysRevD.101.063523.
2. V. Poulin, T. L. Smith, R. Calderón, T. Simon, "Impact of ACT DR6 and DESI DR2 for early dark energy and the Hubble tension," *Phys. Rev. D* **113**, 063519 (2026), https://doi.org/10.1103/bx25-1g5d.
3. A. Reeves et al., "Multiprobe constraints on early and late time dark energy," *Phys. Rev. D* **113**, 103512 (2026), https://doi.org/10.1103/w737-nk6r.
4. G. D'Amico, T. Hamill, N. Kaloper, "Quantum field theory of interacting dark matter and dark energy: Dark monodromies," *Phys. Rev. D* **94**, 103526 (2016), https://doi.org/10.1103/PhysRevD.94.103526.
5. L. A. Ureña-López and A. X. Gonzalez-Morales, "Towards accurate cosmological predictions for rapidly oscillating scalar fields as dark matter," *JCAP* **07**, 048 (2016), https://doi.org/10.1088/1475-7516/2016/07/048.
6. M. C. D. Marsh, "The swampland, quintessence and the vacuum energy," *Phys. Lett. B* **789**, 639 (2019), https://doi.org/10.1016/j.physletb.2018.11.001.
