# Charge-normalized QICT receiver: exact finite-N result and cosmological test

**Scientific distinction.** QICT's published copy-time inequality does not, by itself, determine a gravitational scalar action. The n=3 potential below is the known axionlike EDE potential, already implemented in CAMB EarlyQuintessence. The new result is an operator-level correction and a finite-charge coherent-state identity, which are independent of Planck fitting.

## Local microscopic correction

On a two-flavor bosonic site define
- N = n1+n2
- X = b1^†b2+b2^†b1
- C = N^(-1/2) X N^(-1/2), with the Moore–Penrose inverse set to zero on vacuum.

In each full fixed-N Schwinger spin representation, C = 2 J_x/N, spec(C) is in [-1,1]. Therefore the cubic local Hamiltonian u(1-C)^3 is positive semidefinite, and the full local Hamiltonian with number-conserving hopping and relative-charge inertia still commutes with total charge and flavor exchange.

For the spin-coherent relative-phase state |N,theta> = ((b1^†+exp(i theta)b2^†)/sqrt(2))^N|vac>/sqrt(N!), let x=1-cos theta. The **exact** binomial identity is:

    <(1-C)^3> = [(N-1)(N-2)/N^2] x^3
                 + [6(N-1)/N^2] x^2 + [4/N^2] x.

Proof: Each constituent gives a sigma_x=-1 outcome with probability p=x/2; if K~Binomial(N,p), then 1-C=2K/N on spectral outcomes. Evaluate E[K^3]=N(N-1)(N-2)p^3+3N(N-1)p^2+Np. The large-N limit is (1-cos theta)^3; the finite-size terms are fixed without fitting.

The old conversion normalization X/(2nmax) has mean eta*cos theta with eta=N/(2nmax), and its cubic mean-field potential has curvature at the origin 3u eta(1-eta)^2. That is an unphysical cutoff-dependence if the target continuum is cubic EDE. The corrected normalization removes this at leading coherent order, but leaves the *physical finite-N variational corrections* above. They have not been shown to equal the renormalised one-particle-irreducible potential.

## Conditional rotor dictionary

For the balanced large-filling rotor:
- inertia I ≈ 1/(4 Delta)
- phase gradient coefficient J ≈ t N/2
- F^2 = I/ell^3
- V0 = u/ell^3
- v_R^2 = J ell^2/I ≈ 2 t N Delta ell^2

Relativistic matching v_R=1 and minimal coupling to Einstein gravity are additional assumptions, not results of the copy-time theorem. The microscopic QCA gates must still determine t, Delta, u, N and ell to generate a parameter-reducing physical prediction.

## Direct Einstein–Boltzmann test, 7 November 2026

The exact continuum potential V=V0(1-cos(phi/F))^3 was tested with CAMB 2.0.5's **EarlyQuintessence** (not the effective-fluid approximation), Cobaya and official Planck 2018 Plik-lite TTTEEE, Commander low-TT, SimAll low-EE. A fixed LambdaCDM control with (H0,ombh2,omch2,ns)=(67.7,.02237,.120,.965) gave ln L=-509.84894.

- Nine conditional (H0,zc,fde_zc) nonzero receiver points with other parameters fixed: best ln L=-548.37740, delta chi2=77.05693 vs that fixed control (H0=70, zc=4000, fde_zc=.05).
- A second 24-point grid varying H0, physical omch2, and primordial tilt ns at zc=3400, fde=.10, theta_i=2.5 found ln L=-530.35638 at H0=70.5, omch2=.128, ns=.990. Delta chi2=41.01488 vs the fixed LambdaCDM control. This is a strong parameter degeneracy, not evidence of a global exclusion or detection.
- The CAMB calibrated F differs from the earlier background-only F/Mpl=.40 illustration: for H0=70, zc=3400, fde=.05, CAMB returns F/Mpl=.150607, m=1.54794e-55 (CAMB internal units). It is erroneous to treat a CAMB input guess as the calibrated field scale.

The complete input grids and outputs are archived as workflow artifacts in GitHub Actions. A full Planck--ACT--BAO--SH0ES posterior and independently maximised LambdaCDM reference are NOT part of this conditional experiment.

Source references: Sacha, Quantum Rep. 8, 33 (2026), doi:10.3390/quantum8020033; Smith, Poulin and Amin, Phys. Rev. D 101, 063523 (2020), doi:10.1103/PhysRevD.101.063523; CAMB EarlyQuintessence documentation, https://camb.readthedocs.io/en/latest/dark_energy.html .
