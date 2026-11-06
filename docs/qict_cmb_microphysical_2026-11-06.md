# QICT microphysical CMB closure — verification branch (6 November 2026)

This branch adds a public verification kernel for the one-pole receiver background, its global monotonicity domain, the mass-varying cold-dark-matter number current and the linear Klein–Gordon / continuity / Euler source terms.

Run in the repository root:

    python -m unittest discover -s tests -p 'test_qict_cmb_microphysics.py' -v

The analysis has three separate evidence levels.

1. **Exact local action identities.** A canonical scalar has unit rest-frame sound speed and vanishing intrinsic linear shear. A conserved odd-particle number current with mass varying through the field implies a fixed covariant scalar–matter exchange current.
2. **Executed checks.** The kernel verifies the unique positive single-pole Friedmann branch and integrates a nonzero-coupling scalar–particle background. The particle-number invariant is conserved. The tests do not use cosmological survey likelihoods.
3. **Official likelihoods.** Full Planck 2018 TT/TE/EE/low multipoles/lensing and ACT DR6 require released survey data, nuisance parameters, a Boltzmann runtime at each proposed cosmological point and converged posterior chains. They are not computed by this verification kernel.

The complete Symmetry research submission contains thermal massive-neutrino background integrals, 17 verified tests, Online Resource 7 and a separate CAMB / Cobaya source adapter. The minimal public kernel does not replace those larger files or claim a complete CAMB/CLASS implementation of interacting cold dark matter.

For lensing, the published joint ACT–Planck product replaces the corresponding overlapping standalone lensing likelihoods. It is not multiplied by them.

Methodological references: Ma and Bertschinger (1995); Planck Collaboration A&A 641 A5 (2020); ACT DR6 lensing collaboration likelihood release, https://github.com/ACTCollaboration/act_dr6_lenslike .
