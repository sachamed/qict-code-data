# QICT observed CMB calculations — 6 November 2026

## Verified native likelihood calculations

The following are measured-data log-likelihood evaluations from checked-in CAMB 2.0.5 source. No distance-prior surrogate has been substituted for a CMB spectrum.

- Planck 2018 Plik-lite high-ell TTTEEE: ln L = -326.3311920360; Commander low-ell TT: -12.0380235739; SimAll low-ell EE: -197.9890044900; native Planck lensing: -4.8833602442; total: -541.2415803441. GitHub Action 42314656579.
- ACT DR6 foreground-marginalized primary TTTEEE (CMB-only): ln L = -100.8587921093 at the same QICT fixed point. GitHub Action 42315437097.
- Official ACT DR6 lensing v1.2: chi2 = 23.8890843 (10 bandpowers) at mnu = .06 eV. Published **joint** ACT--Planck lensing product: chi2 = 30.5945317 (19 bandpowers), as an alternative, not an additional factor. GitHub Action 42313015146.
- Conditional 12-point lambda grid on real Planck primary plus either ACT-only or joint ACT--Planck lensing: largest observed grid likelihood at lambda=.03 **when H0=69.28 and all remaining parameters fixed**. GitHub Action 42314925339.

## H0--receiver degeneracy: 63 official Planck points

An independent 63-point Cartesian grid of nine H0 values and seven lambda values evaluated Plik-lite TTTEEE+Commander low-TT+SimAll low-EE. The highest sampled log likelihood is **-507.4683106726 at H0=67.7 km/s/Mpc, lambda=0**, the receiver-free LambdaCDM limit.

After maximization over sampled H0 only, best log likelihood at each amplitude:

| lambda | best sampled H0 | ln L(Planck primary) | Delta chi2 relative best grid |
|---:|---:|---:|---:|
| 0.00000 | 67.7 | -507.46831 | 0.00 |
| 0.01000 | 68.3 | -511.61887 | 8.30 |
| 0.02000 | 69.0 | -521.61150 | 28.29 |
| 0.03000 | 69.28 | -524.33566 | 33.73 |
| 0.03776 | 69.8 | -533.41994 | 51.90 |
| 0.04500 | 69.8 | -539.64471 | 64.35 |
| 0.06000 | 70.5 | -559.26555 | 103.59 |

This controls the earlier fixed-H0 preference, which does **not** imply a global QICT detection. Other five cosmological parameters, high-ell foreground nuisance and full model comparison remain outside this 2D conditional scan. Do **not** label the grid a multidimensional MCMC posterior. GitHub Action 42315939089 contains the measured likelihood evaluations.

## Algebraic normalization of the supplied QSL transport branch

For the distinct saturation-transport branch, the copy-horizon equation Lcopy^2 = xi D_H/H and collapse-bound density rho_Q = 3 kappa Mpl^2 S/Lcopy^2 imply

    rho_Q = 3 Mpl^2 H0^2 [kappa/(xi D_H H0)] E S.

Thus the normalized, dimensionless coefficient must be nu_tilde = kappa/(xi D_H H0), optionally multiplied by a defined propagation factor c_Q^2. The supplied Supplement I Eq. (8) incorrectly prints kappa H0/(xi D_H); the latter violates the dimension of a physical diffusivity and its own Eq. (10). The submitted consolidated manuscript corrects this algebra, without changing the original PDF. This is the **separate QSL tanh transport branch**, not the rational one-pole history used by the evaluated CAMB likelihood.

## Statistical scope

Results are official measured-data *pointwise* likelihoods and two conditional grids. The uncompressed multifrequency Plik nuisance likelihood, a common primary Planck/ACT overlap-aware joint fit, a 7+ dimensional converged posterior, evidence/Bayes factors and a Boltzmann fork for the exchanging CDM branch have **not** been run. The lensing alternatives use the published covariance and are not double-counted.

The associated *Symmetry* research manuscript includes 53 pages, eight online-resource designations (Resources 2–8), exact code, raw JSON and CSV, and covariant microphysical checks. The final source archive also records 17/17 locally passing consistency tests.
