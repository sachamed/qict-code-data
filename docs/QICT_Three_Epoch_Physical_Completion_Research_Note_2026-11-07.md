# Charge-Resolved Dark-Sector Cosmology: Physical Failure Modes and a Falsifiable Completion Programme

**Research note — 7 November 2026.** Repository: sachamed/qict-code-data.  
**Status:** Microscopic analytic identities and selected Einstein–Boltzmann likelihood evaluations are reproducible. No joint cosmological inference establishing superiority to ΛCDM has yet been obtained.

## Scientific position

The central issue is no longer a failure to integrate the coupled Einstein–Boltzmann system. It is a failure of the tested high-energy trajectories to reproduce the full acoustic and distance observables simultaneously. An elevated geometrically inferred Hubble parameter must not be interpreted as a resolved tension unless the corresponding temperature, polarization, lensing, baryon acoustic oscillation, and matter-power likelihoods remain competitive with a separately fitted reference.

The completed 7 November 2026 bounded profile, [GitHub Actions 42406075849](https://github.com/sachamed/qict-code-data/actions/runs/42406075849), evaluated Planck 2018 Plik-lite TTTEEE, low-TT, low-EE and DESI DR2 BAO, with five varying parameters in each model. The best evaluated ΛCDM point was H0=67.922905 and logL=-511.557700. The best evaluated finite-charge QICT point was H0=75.369024 and logL=-760.997035, giving a **provisional** Δχ²=+498.878671. **Neither optimizer converged**: both exhausted the 65-evaluation budget, and the scalar potential, coupling and τ remained fixed. The result therefore identifies the present branch as unsuccessful, not QICT as a theory as a whole.

## I. Exact finite-charge construction and the late-tail effect

For a fixed charge N≥3, let C=2Jx/N, and define the positive parent operator u(1-C)^3. Its spin-coherent expectation is the polynomial

[
V_N(phi)=V_0 left(rac{4x}{N^2}+rac{6(N-1)x^2}{N^2}
+rac{(N-1)(N-2)x^3}{N^2}ight),qquad
x=1-cos(phi/F).
]

This is an exact coherent-state expectation, **not** a demonstrated fully quantum 1PI effective potential. The factorial-ordered parent producing exactly V0 x³ at finite N is a distinct operator; the two must not be silently identified.

For N=250 the coefficients of (x,x²,x³) are (0.000064,0.023904,0.988032). Equality of the x³ and x² terms occurs at x=0.0241935484 or amplitude φ/F=0.2204165935; equality of the x² and x terms is x=0.0026773762 or amplitude φ/F=0.0731925098. These are **potential-term crossovers**, not measured cosmological transition redshifts. At small amplitude, the exact curvature is

[
m_{phi,min}^2=rac{4V_0}{N^2F^2}.
]

In the rapid-oscillation regime of V∝|φ|^p, the averaged equation of state is w=(p−2)/(p+2). The coherent finite-N polynomial thus permits a progression of approximate dominant monomial behaviors φ⁶ (w=1/2), φ⁴ (w=1/3), and φ² (w=0). Consequently, the finite-charge coherent rotor is **not** a model whose early energy necessarily redshifts faster than radiation forever. The exact cubic-potential parent lacks these lower-order terms. Neither conclusion by itself proves a viable dark-matter abundance.

This late-tail effect is principally a prediction for later evolution. It cannot be presumed to explain the **high-multipole CMB** likelihood penalty without first demonstrating that the field actually traverses the corresponding amplitude regimes near recombination.

## II. Small dark-mass modulation is quantitatively bounded

For the implemented ansatz

[
m_chi(phi)=m_0sqrt{1+gsin^2(phi/2F)},qquad gge0,
]
[
0lerac{m_chi(phi)}{m_0}-1lesqrt{1+g}-1.
]

At g=0.02, the largest possible fractional excursion from m0 is 0.00995049, or 0.995049%. A model attributing a large reorganization of the cosmic expansion to this coupling must demonstrate the effect through the full evolution; the bare mass change is intrinsically small.

Defining β(φ)=∂φ ln mχ, the covariant homogeneous equations for a pressureless χ number fluid are

[
dotho_chi+3Hho_chi=eta(phi)dotphi,ho_chi,qquad
ddotphi+3Hdotphi+V_{,phi}=-eta(phi)ho_chi.
]

A crucial density-convention identity for linear perturbations is
δρχ/ρχ = δnχ/nχ + β δφ. If a code variable is the number contrast, the mass perturbation must appear in the stress-energy; if it is the energy contrast, that contribution must not be added a second time. The actual variable definitions must be followed through every Einstein and Boltzmann source before claiming gauge consistency.

## III. A minimal covariant three-component interpretation

A conditional model with an exact dark parity can be organized as

[
S=int d^4xsqrt{-g}left[
 rac{M_{m Pl}^2}{2}R-rac12(partialphi)^2
 -V_N(phi)-ho_{m vac}
 +archi(islashed
abla-m_chi(phi))chi
ight]+S_{m SM}.
]

- The even receiver phase φ can supply a **transient pre-recombination energy component**. Its charge-dependent oscillatory tail must be bounded by lensing and structure data.
- An odd state χ, stable under an exact preserved dark Z2 parity and lightest in its sector, is a **dark-matter candidate**. Stability is not a relic calculation: its absolute mass m0 and its production collision term remain independently necessary.
- A vacuum term ρvac can provide **late-time acceleration**, but its observed small value and radiative stability are not explained merely by writing a constant. The positive polynomial V_N with minimum V_N=0 does not generate sustained late acceleration once φ enters rapid small-amplitude oscillations.

If future data establish genuinely dynamical late dark energy requiring effective w crossing −1, a coupled effective description may exhibit this without a canonical scalar kinetic ghost. Explicitly, since Q=βρ_χ dotφ,
[
w_{phi,mathrm{eff}}=w_phi+rac{Q}{3Hho_phi}.
]
Here wφ≥−1 for a healthy canonical field. The effective crossing depends on a specified interaction and on the observational inference model; it cannot be asserted from DESI BAO alone.

## III-b. Conditional acoustic feasibility bound

A useful exact constraint exists when the non-scalar cosmological history, baryon sound speed and acoustic integration endpoints are **held fixed**. If a positive scalar contributes a fraction 0<=f_phi(z)<=fmax to the total energy at each relevant redshift, then H_new(z)=H_bg(z)/sqrt(1-f_phi(z)), so the sound horizon is an average of sqrt(1-f_phi(z)) weighted by positive background integrand. Therefore

[
sqrt{1-f_{max}}lerac{r_s^{m new}}{r_s^{m bg}}le1.
]

For fmax=0.10105 this implies the largest possible reduction of r_s under those assumptions is 1-sqrt(0.89895)=5.187%, compared with 1.901% in the previously reported simplified QICT background. The naive inverse H0 ratio 67.4/74.93 would require a 10.049% reduction if the distance to last scattering scaled strictly as 1/H0, which it does **not** across freely varying cosmological parameters. The illustrative fraction needed for that naive scale would be at least 19.089% even if sustained throughout the acoustic integral.

The previous result is deliberately **not** a strict coupled-QICT bound: when CDM mass evolves, the non-scalar background is not fixed; and in a full likelihood the angular-diameter distance and sound horizon respond together. The correct test compares the full H(z), recombination and the angular/diffusion scales at each fitted point. This bound demonstrates why a narrow 10%-peak alone is insufficient evidence of a high-H0 solution.

## IV. A speculative but sharp test of inter-epoch charge hierarchy

A proposed *microscopic question*, not a derived result, is whether a receiver vacuum projector carries a first nonvanishing suppression of order N⁻⁴ relative to the excitation scale. If a microscopic Hamiltonian demonstrated

[
ho_{m EDE}(z_c)=N^4,ho_{m vac},
]
then, with a matter-plus-radiation pre-episode density (E_{m bg}^2(z)=Omega_m(1+z)^3+Omega_r(1+z)^4), a conditional prediction would follow:

[
f_{m EDE}(z_c)=
rac{Omega_{m vac}N^4}
{E_{m bg}^2(z_c)+Omega_{m vac}N^4}.
]

Illustration only: for (Ωm,Ωr,Ωvac,z_c,N)=(0.315,9.3×10⁻⁵,0.685,3500,250), fEDE=0.08870555. Requiring exactly fEDE=0.10 at this redshift instead gives N≈258.408. This numerical proximity is **not evidence** of the hierarchy; N=250 and zc=3500 were selected as model inputs. Moreover, nothing in the proved finite-N cubic moment currently implies a fourth-order suppression of the renormalized vacuum energy. The proposed relation becomes a scientific prediction only after a separate microscopic derivation fixes the exponent, normalization and charge without fitting them to cosmological data.

## V. A direct, discriminating computational experiment

The most informative next step is a **controlled source ablation**, not a declaration of a newly fitted H0. Hold the cosmological parameters and scalar initial data fixed, and compare:
1. finite N=250, g=0.02;
2. finite N=250, g=0;
3. large N=10⁶ (continuum cubic-potential surrogate), g=0.02;
4. large N=10⁶, g=0;
5. an ordinary ΛCDM control.

The code and GitHub workflow in this branch implement precisely that matched **fixed-input** diagnostic for official Planck 2018 and DESI DR2 likelihoods. Large-N potentials do **not** identify the finite-N factorial parent at the operator level. The ablation separates the effects of the extra coherent-state terms and of DM coupling at fixed input, without fitting away either signal.

An observed large penalty common to all four scalar cases points toward the scalar acoustic history or initialization rather than the finite-N coefficients or small g alone. A penalty restricted to finite N would implicate the coherent correction and motivate comparison to the factorial parent. A penalty dependent on g would motivate a fresh audit of momentum exchange and the gauge-invariant perturbation sources.

## VI. Model development that could genuinely change the empirical result

The next full theory comparison must sample variables tied to physical features of the energy injection rather than hold the original scalar normalization fixed. A controlled shooting map should solve for scalar amplitude and field scale against specified (z_c), (f_phi(z_c)), and initial phase, while computing curvature and the total energy budget at each point, **not freezing a present scalar density from one reference point**. A complete inference varies cosmological densities, primordial amplitude and tilt, optical depth, physically motivated charge and interaction parameters, and nuisance parameters, using matched neutrino assumptions.

Every candidate passing background screening must pass (i) covariant stress-energy and number-current checks, (ii) linear perturbation and gauge diagnostics, (iii) lensed TT/TE/EE/BB and matter-spectrum comparisons, (iv) full Planck/ACT–Planck lensing/DESI joint likelihood with the intended nonlinear treatment, and (v) independent ΛCDM reoptimization and posterior or multi-start profile convergence. SH0ES should first be held out as a predictive tension test and then explicitly included in a second, separately labeled fit.

## VII. Publication criterion and defensible main result

The publishable mathematical contribution is currently the **finite-charge, symmetry-constrained potential and its predicted cascade of effective oscillatory regimes**, together with a reproducible, covariant, quantitative test of its cosmological relevance. The stable odd dark sector becomes an absolute dark-matter prediction only after derivation of its energy calibration and particle production. The late vacuum is not explained until its suppression follows from the independent microscopic model.

A combined Hubble-tension claim requires a physically admissible region that actually fits the spectra, not merely a high-H0 solution of the homogeneous geometry. A physically meaningful negative result, supported by a no-go or a quantitative exclusion of a well-defined parameter region, is also a substantive theoretical advance.

### Academic reference points

1. DESI Collaboration, *DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints* (2025), arXiv:2503.14738, https://arxiv.org/abs/2503.14738.
2. V. Poulin, T. L. Smith, R. Calderón and T. Simon, *Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble Tension*, Phys. Rev. D 113 (2026), arXiv:2505.08051, https://arxiv.org/abs/2505.08051.
3. R. Chen, J. M. Cline, V. Muralidharan and B. Salewicz, *Quintessential dark energy crossing the phantom divide*, arXiv:2508.19101, https://arxiv.org/abs/2508.19101.
