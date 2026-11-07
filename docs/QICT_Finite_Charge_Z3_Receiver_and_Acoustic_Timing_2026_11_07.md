# Finite-Charge Receiver Cosmology: Acoustic Timing and a Conditional Two-Phase Symmetry Construction

7 November 2026 — Research continuation in English. Code and original experimental logs are available in the QICT repository branch qict/three-epoch-physical-consistency-2026-11-07.

## Abstract

Two independent numerical and mathematical developments refine the finite-charge quantum-receiver model. A new physical acoustic-peak reconstruction shows that, at H0 = 70 km/s/Mpc, moving the scalar-energy maximum from redshift 3500 to 2000 improves the fixed-parameter Planck 2018 plus DESI DR2 fit by approximately 95 chi-square units. The model remains disfavored relative to the fixed standard cosmology at that point; an adequate acoustic angle does not establish an adequate temperature, polarization and BAO fit. Independently, a conditional three-channel phase receiver naturally has two relative phases after removal of the diagonal U(1), with a positive kinetic metric and a D3-invariant finite-charge potential. Its two small-amplitude scalar masses are exactly degenerate when all three channels respect the permutation symmetry. A cyclic-invariant reflection-odd composite exists, but its existence is not proof of an elementary dark-matter particle. These results identify the physical and microscopic conditions needed for a controlled two-field cosmological extension.

## 1. Observed acoustic-surface redshift sensitivity

The actual scalar fractional-energy peak and 100 theta_s = 1.04109 were calibrated separately at each point using the modified coupled CLASS source. Official Planck 2018 Plik-lite TTTEEE, Planck low-TT and low-EE, and DESI DR2 BAO likelihoods were evaluated at those points. All other cosmological parameters were held at the values assigned to each sample; this is neither a posterior nor an independently maximized model-selection statistic.

Run 42418134966: https://github.com/sachamed/qict-code-data/actions/runs/42418134966

| H0 km/s/Mpc | Actual zpeak | fpeak | Planck+DESI logL | Delta chi2 relative to a fixed LCDM control |
|---:|---:|---:|---:|---:|
| 70 | 2005 | 0.04713 | -550.94004 | +41.40 |
| 70 | 2500 | 0.05101 | -568.51980 | +76.56 |
| 70 | 3000 | 0.05541 | -585.70362 | +110.93 |
| 71 | 2002 | 0.07748 | -637.71199 | +214.95 |
| 72 | 2000 | 0.09801 | -699.93959 | +339.40 |

The earlier calibrated H0 = 70, zpeak = 3500 point gave Delta chi2 about +136.8. Thus the redshift of energy release has a large independent effect on the observed CMB even after the acoustic peak position has been matched. Near-H0 and early-late compromise models require an independently optimized LCDM benchmark, converged parameter fits and compatible nonlinear lensing before physical superiority can be asserted.

## 2. A conditional Z3-to-D3 receiver phase doublet

Assume three independent complex receiver-channel amplitudes with nonzero fixed magnitudes and phases theta0, theta1 and theta2. Remove their common phase, and write alpha = theta1-theta0 and beta = theta2-theta0. **On this fixed-amplitude phase submanifold**, the relative phase space is U(1)^3/U(1)_diag, a two-torus. An unconstrained normalized three-component complex quantum amplitude instead has projective state space CP^2, with four real dimensions. The two-torus is therefore a dynamical phase-only reduction, not a consequence of merely having three discrete end-class labels.

A cyclic channel permutation acts on the real phase doublet q=(alpha,beta) as M=[[-1,1],[-1,0]], with M^3=I. A reflection interchanging channels 1 and 2 acts as R=[[0,1],[1,0]], with R^2=I and RMR=M^-1. If both transformations are microscopic symmetries, they generate D3, isomorphic to the permutation group S3.

Removing the diagonal phase from an equal-channel kinetic term produces the positive metric G=(F0^2/3) [[2,-1],[-1,2]]. This metric is invariant under both generators. The three link angles are alpha, beta-alpha and -beta. An invariant potential is the sum of the three identical finite-charge link potentials V0 P_N(1-cos gamma), where P_N(x)=4x/N^2 +6(N-1)x^2/N^2 +(N-1)(N-2)x^3/N^2. The distinct-triplet parent instead has P_N(x)=x^3 and is a different microscopic Hamiltonian.

At the symmetric vacuum, the finite-N potential Hessian is H=(4V0/N^2) [[2,-1],[-1,2]]. Thus G^(-1)H=(12V0/(N^2 F0^2)) I, and the two canonically normalized small-amplitude masses are exactly degenerate. The mere existence of two symmetry-related fields does not create two separate linear transition times. A physically useful staggered excitation requires independent nonlinear phase initial conditions, a derived source of controlled symmetry breaking, or additional microscopic eigenmodes.

A cyclic-invariant reflection-odd function exists:

O_minus(alpha,beta) = sin(alpha)+sin(beta-alpha)-sin(beta)
                    = 4 sin(alpha/2) sin((beta-alpha)/2) sin(beta/2).

It obeys O_minus(Mq)=O_minus(q) and O_minus(Rq)=-O_minus(q). This establishes an allowed sign representation, not the existence, mass, binding or relic abundance of an elementary odd dark particle. The original Schwinger-spin link is bosonic; flavor-exchange parity does not by itself imply a Dirac fermion.

The identifications of three abstract receiver end-classes with three physical phase channels and of their links with independent charge operators are additional assumptions requiring derivation.

## 3. Parity gap and vacuum hierarchy

A separate finite-charge Hamiltonian with distinct-triplet conversion and charge inertia has exact coherent expectation u[(1-cos theta)^3+eta/(4N)]. The constant is not the exact interacting many-body ground-state energy and cannot be directly equated with a renormalized cosmological constant. A putative N^-4 vacuum hierarchy from this term imposes eta=4/N^3, which is an input unless independently derived. At N=250 the required eta is 2.56e-7 and the finite-matrix odd-even gap is about 6.25e-10 times the microscopic energy u. The absolute mass still depends on the microscopic energy calibration and continuum spacetime normalization.

A variable-magnitude heavy dark-fermion coupling also requires a radiative stabilization mechanism. A constant-magnitude chiral phase eliminates a field-dependent constant-background Dirac determinant in an anomaly-free setting, but simultaneously removes the original variable-mass background energy exchange. These are distinct physical models and should not be conflated.

## 4. Immediate empirical falsification programme

The most informative observation-level study is a joint fit that controls the actual scalar peak epoch and strength, physical baryon and dark-matter densities, primordial tilt and amplitude, optical depth, late expansion and the full coupled perturbation equations. A true two-phase proposal requires an Einstein-Boltzmann solver evolving both scalar perturbations; the existing single-field CLASS run is not that solver. Full Planck and Planck-low-ell plus ACT DR6 are distinct, correlated-data strategies and must not be naively multiplied. Each comparison needs an independently optimized LCDM baseline with identical data and nuisance conventions. The local H0 likelihood should first be held out and then included in a separately labeled second inference.

Full two-field early dark energy has precedent in Bella et al. (2026), arXiv:2604.13535, and must not be claimed as a new general mechanism. The potential QICT contribution is a microscopic derivation fixing the doublet, its dynamical couplings and its zero-mode spectrum with fewer freely chosen parameters, followed by a reproducible observed-data demonstration.

## Academic references

1. M. Bella, V. Poulin, S. Vagnozzi and L. Knox, Double the axions, half the tension: multi-field early dark energy eases the Hubble tension, arXiv:2604.13535 (2026), https://arxiv.org/abs/2604.13535.
2. V. Poulin, T. L. Smith, R. Calderón and T. Simon, Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble Tension, Physical Review D 113 (2026), arXiv:2505.08051, https://arxiv.org/abs/2505.08051.
3. G. D'Amico, T. Hamill and N. Kaloper, Quantum Field Theory of Interacting Dark Matter/Dark Energy: Dark Monodromies, Physical Review D 94, 103526 (2016), arXiv:1605.00996, https://arxiv.org/abs/1605.00996.

## Reproducibility

Source scripts for the exact parity, Z3, two-phase homogeneous and acoustic redshift studies and their GitHub Actions runs are committed in the referenced branch. The exact Z3 and D3 algebra and finite-charge vacuum compatibility tests are automated. The homogeneous two-phase numerical results, when run, do not include linear CMB perturbations or observational likelihoods.
