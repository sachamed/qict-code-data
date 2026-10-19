# Quantum Information Copy Time: Haar Obstructions, Central Quotients, and Finite Spectral Tests

## 1. Microscopic action and its representation

The canonical lattice action fixes the ordered product

\[
U_{\rm micro}=U_{\rm int}^{rf}\,e^{-i\pi K_B/63}\,
 e^{-i\pi K_E/63}\,U_{\rm cell}\,
 \Gamma(S_x)\Gamma(S_y)\Gamma(S_z).
\]

Operators act from right to left. Its electric generators are
\(K_{E,Y}=2E_Y^2+E_Y\), \(K_{E,2}=4C_2(E_2)\), and
\(K_{E,3}=3C_2(E_3)\). For a single plaquette on the product cover,
\[
K_B(\theta,u,v)=4(1-\cos\theta)
+32\left(1-\tfrac12\operatorname{ReTr}u\right)
+36\left(1-\tfrac13\operatorname{ReTr}v\right).
\]

The authenticated action JSON specifies these generators and the ordering, but it does not supply complete, indexed route holonomies, all six CAR-conditioned matter amplitudes, or executable and independently certified global gates. In particular, a symbolic factor label does not determine the corresponding matrix elements.

## 2. Finite-rank Floquet obstruction at the Haar endpoint

Let \(\mathcal H_{\rm phys}\) be a closed invariant subspace of a compact-link Haar direct integral. Assume that a nonzero Abelian cycle \(\eta\) remains and that the multiplication unitary
\[
T_\eta=M_{\exp(i126\eta\cdot\theta)}
\]
leaves \(\mathcal H_{\rm phys}\) invariant and commutes with the full microscopic evolution \(U\). The level sets of the nonconstant character are Haar-null, so \(T_\eta\) has no eigenvectors in \(\mathcal H_{\rm phys}\). Every Riesz spectral projection \(P\) of \(U\) commutes with \(T_\eta\). If \(0<\operatorname{rank}P<\infty\), the finite-dimensional invariant range of \(P\) would contain an eigenvector of \(T_\eta\), a contradiction. Therefore no nonzero finite-rank Floquet Riesz projection exists **under these assumptions**.

The same argument rules out a nonzero positive trace-class operator commuting with \(T_\eta\): its positive eigenspace would be finite-dimensional and invariant. It does **not** rule out every normal state, a translation-breaking density, an algebraic state in a distinct representation, or a global infinite-rank spectral projection. Establishing \([U,T_\eta]=0\) for a fully realized gate remains part of the microscopic verification; the cited derivation T28.42 supplies a conditional factor-by-factor argument for the specified gate class.

### Infinite-rank projectors and finite compressed residues

On \(\mathcal H=L^2(\mathbb T)\otimes\mathbb C^2\otimes\mathbb C^6\), take
\[
U=I\otimes\operatorname{diag}(1,i)\otimes I_6,
\quad P=I\otimes|1\rangle\langle1|\otimes I_6,
\]
with normalized source columns \(Ve_a=f\otimes|1\rangle\otimes e_a\).
Then \(\operatorname{rank}P=\infty\), \(V^*V=I_6\), and
\[
V^*PV=I_6,\qquad
G(z)=V^*(z-U)^{-1}V=\frac{I_6}{z-1}.
\]
Thus an infinite-rank spectral projection may have a finite nonzero compressed residue. This exact auxiliary construction establishes logical compatibility, not an interacting QICT excitation, nor physical localization or a dark particle.

## 3. Central quotient and magnetic non-descent

For the diagonal central lift
\(\xi=(e^{i\pi/3},-I_2,e^{2\pi i/3}I_3)\), the magnetic costs at representatives \(\xi^k\) are exactly

| \(k\) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| \(K_B(\xi^k)\) | 0 | 120 | 60 | 72 | 60 | 120 |

The original gate distinguishes representatives of the same central class:
\(\exp(-i120\pi/63)\ne1\). It therefore acts on the product-group cover, not as a function on the faithful quotient. Its use is consistent with the product-cover plaquette convention of derivation T1473, provided this **link-variable convention** is maintained throughout the interacting construction and the matter quotient is implemented separately. The phrase *gauge group modulo the central kernel* in the action header does not itself supply a compatible quotient-link operator. This distinction must be explicit when specifying Gauss constraints and the physical link carrier.

The distinct centre-invariant potential
\[
\widetilde K_B=\tfrac19(1-\cos6\theta)
+12\left(1-\tfrac13\chi_{\rm ad,2}(u)\right)
+16\left(1-\tfrac18\chi_{\rm ad,3}(v)\right),
\quad \chi_{\rm ad,n}=|\operatorname{Tr}|^2-1,
\]
has the same quadratic weak-field expansion, but different nonperturbative minima. It is an alternative action, **not** an admissible silent replacement under the frozen SHA-256.

## 4. Primitive Hamiltonian and its finite cutoffs

On the oriented Abelian plaquette, the separate unwrapped Hamiltonian is
\[
H_s=8r^2+4s-2s(S+S^*),\qquad0\le s\le1,\quad r\in\mathbb Z.
\]
Since \(4sI-2s(S+S^*)\succeq0\), the min–max principle gives \(E_1(H_s)\ge8\). The trial vector \(|r=0\rangle\) yields \(E_0(H_s)\le4s\), hence \(E_1-E_0\ge4\) uniformly. The contour \(|z|=6\) separates the primitive ground state from the excited spectrum by at least 2. The published Gram factors at flux cutoffs \(N=8,12,16,20\) are reproduced byte-for-byte and rechecked against their original SHA-256 certificates.

These are four cutoffs of **one primitive single-plaquette problem**. They are neither four independent spatial Kato buffers nor four interacting charge-sector Floquet states.

## 5. Exact electric Floquet failure of trace-distance convergence

In the L24-oriented single-plaquette convention, \(K_E(r)=8r^2\). The phase-one subspace of \(e^{-i8\pi r^2/63}\) is spanned precisely by \(r\in21\mathbb Z\). On \(|r|\le R=63m\), its rank is \(6m+1\). With the canonical embedding into \(\ell^2(\mathbb Z)\), its normalized projector \(\rho_R\) satisfies
\[
\frac12\|\rho_R-\rho_{2R}\|_1
=1-\frac{6m+1}{12m+1}=\frac{6m}{12m+1}
\longrightarrow \frac12.
\]
Therefore **this specified projector-density sequence** is not trace-norm Cauchy. The statement does not extend to every possible primitive or algebraic state-selection rule.

## 6. Six-source static GNS controls

Eight finite arrays for static external charges \(q=-3,-6\) and cutoffs \(N=8,12,16,20\) provide matrix-valued moments of depth 12. Their original SHA-256 checks and independent double-precision \(C_0\)/Toeplitz eigenvalue screening are supplied. Small negative eigenvalues at the level of rounding error are not rigorous interval proofs of positive semidefiniteness. These static external-charge probes are distinct from the dynamical left/right chiral fermion sources of the complete C10 contraction.

## 7. Finite moments do not identify a spectral atom

For each \(N\ge1\), Haar measure on the unit circle and the uniform atomic measure on the \(N+1\) roots of unity both obey
\(C_0=1\), \(C_n=0\) for \(1\le n\le N\). Only the atomic measure has a nonzero atom at \(z=1\), of weight \(1/(N+1)\). Tensoring by \(I_6\) gives the corresponding positive matrix-valued examples. Thus a truncated positive Toeplitz moment sequence cannot, by itself, certify a unique physical isolated pole.

## 8. Quantitative spectral stability

For normal unitary operators \(U,U'\) on one common Hilbert space and a closed contour \(\Gamma\) at distance at least \(g>0\) from both spectra, the resolvent identity gives
\[
\|P_\Gamma(U)-P_\Gamma(U')\|
\le\frac{\operatorname{length}(\Gamma)}{2\pi g^2}\|U-U'\|.
\]
For source isometries \(V,V'\),
\[
\|V^*PV-V'^*P'V'\|
\le \frac{\operatorname{length}(\Gamma)}{2\pi g^2}\|U-U'\|
+2\|V-V'\|.
\]
The bound is conditional on a common positive gap and compatible source embeddings. A positive trace lower bound suffices to retain at least one source-coupled atom; a lower bound on \(\lambda_{\min}\) of a six-channel residue is a strictly stronger condition and is **not** required for a single dark particle.

For the two-phase auxiliary model \(\{1,i\}\) and a midpoint contour quadrature with radius \(a<|1-i|\), the exterior-channel quadrature error, in exact arithmetic, is bounded by \(q^M/(1-q^M)\), \(q=a/\sqrt2\), for \(M\) midpoint nodes. Floating-point rounding error must be bounded separately. Numerical agreement of three methods in this elementary model is not independent physical evidence.

## 9. Odd sector and physical pole conditions

For a physical self-adjoint unitary involution \(Q\) satisfying \([U,Q]=0\), the odd projector is \(P_{\rm odd}=(I-Q)/2\). An isometric, Gauss-compatible, copy-odd source must obey \((I+Q)V_\chi=0\). At an isolated spectral point of the same full evolution, the compressed residue is \(Z_\chi=V_\chi^*P_\chi V_\chi\succeq0\). Its nonvanishing requires \(\operatorname{Tr}Z_\chi>0\), not merely a spectral eigenvalue elsewhere in the Hilbert space. Finite-dimensional unitary and source checks in the accompanying code exercise these identities without identifying a QICT state.

## 10. Scale and physical interpretation

A Floquet phase fixes a quasi-energy only modulo \(2\pi\hbar/\tau_*\). With an independently measured \(\tau_*\), a vacuum-subtracted and branch-resolved phase difference \(\Delta\theta\) yields
\[
\Delta E_\chi(\mathrm{GeV})=
\frac{6.582119569\times10^{-25}\,\Delta\theta}{\tau_*(\mathrm{s})}.
\]
A mass assignment further requires a physical rest frame, stability, and the appropriate dispersion relation; the corresponding mass unit is \(\mathrm{GeV}/c^2\). The included interval calculation is dimensional arithmetic, not an independently calibrated QICT mass prediction.

## 11. Conditions for a complete interacting calculation

A complete physical computation requires an unambiguous common gauge-link/matter Hilbert representation, the certified route-first operator \(U_{\rm int}^{rf}\), compatible Gauss and CAR projections, an endpoint state-selection rule respecting the Haar obstruction, quantum C10 maps, six interacting chiral source columns, physical matrix moments, a copy-odd neutral source, a uniform gap/overlap condition across regulators, and an independent metrological clock. The computations in this package establish precise tests and mathematical constraints for those objects; their physical values are not encoded in the supplied canonical action or finite primitive evidence.

## 12. Certified infinite-flux primitive ground state

Consider the *unwrapped* one-plaquette Hamiltonian on \(\ell^2(\mathbb Z)\),
\[
H=8D^2+4I-2(S+S^*),\qquad De_r=r e_r.
\]
Its quadratic form is
\[
\langle f,Hf\rangle=8\sum_r r^2|f_r|^2+2\sum_r|f_{r+1}-f_r|^2.
\]
The form is nonnegative, its domain has finite-support sequences as a form core, and the confining diagonal makes the resolvent compact. The strictly connected nearest-neighbour hopping gives a positivity-improving semigroup, hence the ground-state eigenvalue \(E_0\) is simple with a strictly positive eigenvector. The min–max principle applied to the diagonal operator \(8D^2\) gives \(E_1\geq8\), whereas the normalized site-zero trial vector has energy 4. Thus \(E_0\leq4\) and \(E_1-E_0\geq4\). The contour \(|z|=6\) isolates this *Hamiltonian* ground state from all other eigenvalues, with a separation of at least 2.

For **any** normalized finite-support trial vector \(v\), let
\[
\mu=\langle v,Hv\rangle<8,\qquad
\epsilon^2=\langle v,(H-\mu)^2v\rangle.
\]
Temple's inequality, using \(E_1\geq8\), and the spectral theorem give
\[
\boxed{\mu-\frac{\epsilon^2}{8-\mu}\leq E_0\leq\mu},\qquad
\boxed{\|(I-P_0)v\|\leq\frac{\epsilon}{8-\mu}},
\]
where \(P_0\) is the rank-one spectral projector onto the infinite-flux ground state. For the associated pure-state density \(\rho_v=|v\rangle\langle v|\),
\[
\frac12\|\rho_v-P_0\|_1
=\sqrt{1-\langle v,P_0v\rangle}
\leq\frac{\epsilon}{8-\mu}.
\]
The second inequality follows since every vector in the orthogonal complement of the ground state has spectral support in \([8,\infty)\). Temple's inequality follows by integrating \((\lambda-\mu)(\lambda-E_0)\) against the spectral measure. These arguments apply to the infinite-dimensional Hamiltonian, not merely to its finite matrix truncations.

In `src/primitive_infinite_volume_certificate.py`, each stored binary64 component is interpreted as an **exact dyadic rational**, and all sums for \(\langle v,v\rangle\), \(\langle v,Hv\rangle\), and \(\|Hv\|^2\)—including the two omitted boundary fluxes—are evaluated with integer arithmetic through `fractions.Fraction`. Consequently the Rayleigh and Temple endpoints and the square of the state-overlap bound are rationally certified for the *precise stored trial vectors*. Their decimal representations are rounded outward. The construction requires no trust in a claimed floating-point eigenvalue error or an extrapolation model.

The normalized finite-cutoff Galerkin ground-state sequence converges to the infinite-flux ground state: the finite-support subspaces increase to a form core, so their Rayleigh minima decrease to \(E_0\). The uniform positive gap then forces the excited-state weight to vanish. The four source-backed numerical cutoffs \(N=8,12,16,20\) additionally provide explicit finite error bounds and direct pairwise density distances in the common integer-flux basis. Their complete machine-readable rational enclosures appear in `results/PRIMITIVE_INFINITE_VOLUME_RATIONAL_CERTIFICATE.json`.

This is a convergence theorem for the **primitive Abelian Hamiltonian**. It neither resolves the full Haar/Floquet finite-rank obstruction nor constructs four spatial \(\lambda=1\) charged Kato buffers, and no certificate for either of those claims follows from it.

## 13. Central-character selection rule for the faithful quotient

Let
\[
G=\bigl(U(1)\times SU(2)\times SU(3)\bigr)/\mathbb Z_6,
\qquad \xi=(e^{i\pi/3},-I_2,e^{2\pi i/3}I_3).
\]
A product-cover irreducible representation with integer Abelian charge \(Y\), \(SU(2)\) spin \(j\) (\(2j\in\mathbb N_0\)), and \(SU(3)\) triality \(t\in\{0,1,2\}\) transforms under \(\xi\) by the scalar
\[
\exp\!\left(\frac{i\pi}{3}\left[Y+3(2j)+2t\right]\right).
\]
It therefore defines a representation of the **quotient** if and only if
\[
\boxed{Y+3(2j)+2t\equiv0\pmod 6}. 
\]
The selection rule is exact; it needs neither a weak-field expansion nor numerical sampling. In particular, the mixed character
\[
X(\theta,u,v)=e^{i\theta}\frac{\operatorname{Tr}_2u}{2}
\frac{\operatorname{Tr}_3v}{3}
\]
is invariant under each of the six central lifts, while isolated characters \(e^{i\theta}\), \(\operatorname{Tr}_2u\), and \(\operatorname{Tr}_3v\) fail the selection rule separately. A magnetic action built from suitable real parts of quotient-admissible mixed and adjoint characters would be well-defined on the quotient. **Such an action would not be the frozen QICT magnetic action:** the original independent fundamental plaquette terms are not centre invariant, and replacing them changes the full nonperturbative theory and its action SHA-256. The quotient criterion alone does not determine the chiral link carrier, a physical C10 contraction, or any physical mass.

## 14. Certified nonvanishing local Hamiltonian residue and split-step distinction

For the unwrapped primitive Hamiltonian of Section 12, the resolvent
\[
G_{00}(z)=\langle e_0,(z-H)^{-1}e_0\rangle
\]
has a **simple isolated pole** at the genuine infinite-flux eigenvalue \(E_0\), with spectral residue \(Z_0=|\langle e_0,\psi_0\rangle|^2\). Positivity does not require a numerical extrapolation: because \(H\ge 8D^2\),
\[
\sum_{r\ne0}|\psi_0(r)|^2\le E_0/8\le1/2,
\qquad\boxed{Z_0\ge1/2}.
\]
The source-backed rational dyadic certificate gives a sharper enclosure of this *actual infinite-flux* residue. For any normalized approximate vector \(v_N\), let \(p_N=|\langle e_0,v_N\rangle|^2\) and \(\delta_N=\epsilon_N/(8-\mu_N)\). Since rank-one projector operator-norm distance equals pure-state trace distance,
\[
\boxed{p_N-\delta_N\le Z_0\le p_N+\delta_N}.
\]
Each term is evaluated from exact binary input coefficients and bounded by outward-rounded rational comparisons. For \(N=20\),
\[
0.9058301980893397686289319480855
\;\le Z_0\le\;
0.9058301980893398126289319480856.
\]
The pole location has the simultaneously certified energy enclosure in `results/PRIMITIVE_LOCAL_RESOLVENT_POLE_CERTIFICATE.json`. A finite parity-even Jacobi diagonalization and an independent continued-fraction Schur reduction check its location and residue, with float64 discrepancies separated from the rational proof. The finite Jacobi solver uses full-spectrum `stev`: a selected-eigenvalue bisection driver was observed to lose approximately \(10^{-12}\) absolute accuracy for the widest tested diagonal range and was not used to support the infinite-volume certificate.

**Essential distinction.** The actual ordered elementary Floquet gauge update in this same subsystem is
\[
U_{BE}=\exp(-i\phi_*K_B)\exp(-i\phi_*K_E),\quad
K_B=4-2(S+S^*),\quad K_E=8D^2,
\]
not \(\exp[-i\phi_*(K_B+K_E)]\). Using the exact infinite-link Bessel kernel
\[
\langle r|e^{-i\phi_*K_B}|s\rangle=
  e^{-4i\phi_*}i^{r-s}J_{r-s}(4\phi_*),
\]
the finite trial vectors yield an optimized Floquet eigenphase residual
\[
\min_{|\zeta|=1}\|U_{BE}v_N-\zeta v_N\|
=\sqrt{2-2|\langle v_N,U_{BE}v_N\rangle|}
\simeq0.03211158.
\]
The infinite-link convolution and a padded matrix exponential agree numerically to approximately \(10^{-16}\). The measured defect is many orders above the rationally certified source-to-primitive-ground errors. This rules out identifying the Hamiltonian eigenstate with a stationary state of the split-step dynamics **on numerical evidence in this exact elementary gauge sector**. It does not settle the full \(U_{\rm micro}\), whose route-first interacting factor remains unspecified. Most importantly, the primitive Hamiltonian pole is not thereby a physical Floquet or dark-sector particle.

## 15. Exact elementary Floquet shift obstruction

The primitive Hamiltonian and its ordered elementary Floquet update live in the same infinite electric-flux Hilbert space but have different spectral symmetries. In the single-plaquette subsystem, let \(T_{126}e_r=e_{r+126}\). The electric gate has the exact integer identity
\[
K_E(r+126)-K_E(r)=8[(r+126)^2-r^2]=126(16r+1008).
\]
Consequently \(e^{-i\pi K_E/63}\) commutes with \(T_{126}\). The magnetic generator \(K_B=4-2(S+S^*)\) commutes with all integer flux shifts. The ordered product \(U_{BE}=e^{-i\pi K_B/63}e^{-i\pi K_E/63}\) therefore commutes with \(T_{126}\) **exactly on \(\ell^2(\mathbb Z)\)**.

The bilateral shift \(T_{126}\) has no nonzero \(\ell^2\) eigenvectors: any eigenvector would repeat nonzero modulus over an infinite arithmetic progression. If a spectral Riesz projector \(P\) of \(U_{BE}\) were nonzero and finite rank, its finite-dimensional invariant range would contain an eigenvector of \(T_{126}\), a contradiction. Thus **the elementary ordered single-plaquette Floquet unitary has no nonzero finite-rank Riesz spectral projection**, whereas the unwrapped Hamiltonian \(H=K_E+K_B\) has the simple isolated rank-one ground-state projection constructed above. There is no contradiction: the unwrapped \(H\) is not invariant under \(T_{126}\).

This theorem settles the *elementary Abelian sector* of the Haar/Floquet rank obstruction. It does not independently establish that the unknown full route-first matter operator \(U_{\rm int}^{rf}\) commutes with the same shift on the Gauss-constrained physical Hilbert space; that remains a separate condition for applying T28.43 to the complete microscopic construction.
