# Verified QICT coupled CLASS spectra and survey-likelihood scope (7 November 2026)

## Reproducible numerical evidence

An interacting-quintessence CLASS fork was patched with the finite-`N` coherent receiver potential and the varying-mass cold component `m_chi(phi)=m_chi0 sqrt(1+g sin^2(phi/(2F)))`. These modifications reach the scalar background equations, CDM continuity/Euler equations, scalar perturbation source, and total Einstein density source in Newtonian gauge.

A GitHub Actions simulation (run **42349237753**) compiled native CLASS and computed the fully coupled CMB TT/TE/EE/BB/lensing potential and linear matter spectra for `N=250`, `F=0.35 Mpl`, `Omega_scf(z=0)=1e-8`, `g=0` versus `g=0.02`. Lensed spectra contain ell=2..4000 and matter output contains 639 k values. The dimensionless comoving number invariant `rho_c a^3/m_chi(phi)` drifts no more than **1.71e-12** for g=0.02.

At redshift z=1100 the g=0.02 scalar contributes only approximately **1.735e-6** of the critical density. Therefore this is a verified **weak-amplitude integration smoke test**, not a cosmology capable of addressing the Hubble-tension r_s reduction.

For the g=0.02 test, global RMS relative differences from g=0 are roughly 0.0144% (TT), 0.02225% (EE), 0.03916% (TE), 0.04093% (CMB lensing), all conditioned on the fixed cosmological and scalar-parameter input.

## Observed survey baseline independent of QICT

GitHub Actions run **42347896772** successfully evaluated Planck 2018 high-l Plik-lite TTTEEE, Commander low-l TT, SimAll low-l EE, published ACT DR6–Planck **joint** lensing bandpowers, DESI DR2 BAO and an independent SH0ES Gaussian H0 prior. At the H0=67.4 ΛCDM fixed point total lnL is -541.1665264 (SH0ES normalisation omitted); at H0=73.04 with other parameters frozen it is -2411.4073641.

These values are **not** best-fit Δχ², posterior constraints, or a direct QICT model comparison. A QICT-Cobaya coupled CLASS observed-data workflow is maintained separately and all new measurements must be tied to a successful official likelihood run.

## Remaining evidence gate

Before a claim of cosmological superiority: establish a physical receiver fraction near the acoustic era; test full CLASS Newtonian/CDM perturbations and Einstein metric equations against an independent implementation and convergence/precision settings; run sampled/marginalized Planck/ACT/DESI and a proper optimisation of ΛCDM; treat SH0ES as both included and held out; constrain matter P(k) by Lyman-alpha and structure with appropriate thermodynamics/baryonic physics.

An absolute chi-particle mass in eV and relic number density cannot emerge solely from the cold-fluid evolution due to the exact rescaling `m0->a m0; n0->n0/a`. A microscopically computed energy calibration and particle-production mechanism are required.

Upstream third-party code: `https://github.com/kabeleh/iDM` (GPL-3.0), cloned and patched during CI; not copied into this repository.
