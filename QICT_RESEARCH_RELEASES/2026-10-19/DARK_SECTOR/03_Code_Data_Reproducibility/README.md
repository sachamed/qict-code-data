# QICT spectral tests and cosmological calculations

This package contains the main article, supplementary proofs, a standalone prediction and result register, and all inputs and programs used by the reported calculations. The physical interpretation of each result is stated in its source document and in the register.

## Package contents

The archive has exactly four top-level folders:

1. `01_Main_Manuscript`: `manuscript.pdf`, the master LaTeX source, eight thematic source modules, `references.tex`, and fifteen vector figures.
2. `02_Supplementary_Information`: `supplement.pdf`, its complete LaTeX source and references. The supplement proves quantitative pole-persistence bounds and connected-cumulant identities, and identifies the state information entering cosmological observables.
3. `03_Code_Data_Reproducibility`: this README, the pinned Python requirements, `code/` with programs and original finite arrays, and `data/` with processed results and certificates.
4. `04_Theoretical_Predictions`: `predictions.pdf`, its LaTeX source and generated table. The 87-entry register distinguishes finite results, conditional physical relations, reference-model outputs and input parameters.

`SHA256SUMS.txt` at the archive root records the distributed files. Compilation auxiliaries, caches, temporary plots and internal review files are excluded.

## Scientific scope

The supplied C9 arrays describe six static chiral matter-source channels at four finite representation cutoffs. The exact recorded-moment bounds and the polar-unitary operator certificate apply to their explicitly specified finite problems. The nine-state metric array is a separate response test. The supplement gives sufficient hypotheses for a physical odd-sector pole and its residue to persist in a common continuum limit; it does not assert that the finite chiral-source arrays establish those hypotheses for a dark particle.

The scalar-portal masses and couplings, reference background parameters, halo profiles and thermal histories are declared inputs to the numerical examples. The package does not establish a QICT dark-particle mass, physical continuum gravitational coefficients, a first-principles relic abundance, a galaxy-sample fit or full QICT CMB spectra. It supplies the mathematical identities, finite certificates and reproducible calculations actually supported by the included inputs. The reference calculations retain their model provenance throughout all three documents.

## Environment and full reproduction

Use Python 3.12. Tested NumPy, SciPy and Matplotlib versions are pinned in `requirements.txt`; `data/environment.json` records the runtime environment. PDF generation additionally requires pdfLaTeX and standard TeX Live packages: geometry, fontenc, inputenc, lmodern, microtype, AMS packages, mathtools, bm, graphicx, booktabs, tabularx, longtable, array, hyperref, fancyhdr, xcolor and caption. The tested TeX distribution is TeX Live 2023. The documents use the standard article class and require no custom class or style files.

Run the following commands from `03_Code_Data_Reproducibility` after extracting the complete archive:

```bash
python3 -m venv /tmp/qict-reproduction-env
. /tmp/qict-reproduction-env/bin/activate
python3 -m pip install -r requirements.txt
python3 code/reproduce.py
```

Alternatively, the reproduction entry point can be invoked from the archive root:

```bash
python3 03_Code_Data_Reproducibility/code/reproduce.py
```

The command regenerates all fifteen figures, all four numerical article tables, the prediction register, JSON results, particle-mesh fields and the 200/300-step comparison. It runs the mathematical and numerical validators and then rebuilds all three PDFs. The programs use dense finite matrices, a bounded-memory spectral-product scan, quadrature and ordinary differential equations, and the declared `32^3` particle-mesh model. Computation time depends on the machine and numerical-library configuration.

The document build uses temporary directories and publishes only the final PDFs. It checks for unresolved or multiply defined references and overfull typesetting boxes. `reproduce.py` disables Python bytecode output for its child programs. For numerical reproduction without TeX, from this folder:

```bash
python3 code/reproduce.py --calculations-only
```

For a document-only rebuild, using the distributed numerical outputs:

```bash
python3 code/build_manuscript.py
```

The canonical main-article register is stored in `data/PREDICTION_LEDGER.csv`. `generate_tables.py` updates its numerical entries from the computed results. `build_manuscript.py` adds the supplementary entries and resolves cross-references from the current document compilations. A calculations-only run writes the 76 main-article entries; the PDF build produces the complete 87-entry register.

To regenerate the particle-mesh figure from its stored fields:

```bash
python3 code/render_particle_mesh_figure.py
```

## Principal expected outputs

| Calculation | Expected value | Domain |
|---|---:|---|
| R4 left-source atomic-weight upper bound | 0.001956273015 | Exact rational, explicitly normalized recorded moments |
| Zero-phase weight at R1–R4 | Exactly 0 | Outward-certified polar-unitary finite operator model; every stationary state |
| R4 left relative chord gap | At least 1.92308e-7 | All 59,541,075 spectral-centre combinations checked |
| R4 right source-visible chord gap | At least 3.19912e-4 | 6,561 combinations; exact common spectator cancellation |
| Constant-rate freeze-out abundance | 0.1003221390 | Specified constant rate and degrees of freedom; asymptotic tail included |
| Thermal tree-level portal abundance | 0.0234521570 | 300-GeV scalar, portal 0.20, constant `g*=90`, open two-body channels |
| Decay-source freeze-in abundance | 0.0995126375 | One daughter, chosen parent mass and partial width, constant degrees of freedom |
| Spherical-collapse threshold scale factor | 0.4614217772 | Defined overdensity, velocity, mass and radius threshold; smooth radiation |
| Galaxy correlation at 20 Mpc | 0.4661378766 | Smooth BBKS spectrum and selected bias |
| Angular scalar-response/GR ratio at ell=200 | 1.2709504742 | Source plane z=1; equal density amplitude at a=0.01 |
| Particle-mesh final RMS density contrast | 10.4690242761 | Coherent three-wave initial condition, finite mesh |
| Scalar Legendre third derivative | 147/500 | Exact cumulants of the declared three-outcome control |
| Scalar Legendre fourth derivative | 13573/12500 | Same control; checked by independent source-map inversion |

Validators state their numerical tolerances. The independent collapse equation agrees with the radius equation; the constant-rate asymptotic yield agrees across endpoints 2000, 4000 and 8000. The portal check refines angular and thermal quadrature and verifies its high-energy and low-temperature limits. The metric check compares spectral curvature with independent Hermitian finite differences. The particle-mesh comparison tests time resolution at one spatial mesh; it retains its explicit coherent-wave initial condition.

## Certificates and mathematical controls

`certify_moment_bounds.py` proves positivity and a polynomial upper bound for the precisely defined normalized stored moment problem using exact rational arithmetic. Independent physical moment uncertainties and regulator errors enter through the article's perturbation formulas.

`derive_extended_results.py` writes `data/ANALYTIC_EXTENSION_RESULTS.json`. Its controls recover a seven-dimensional cyclic unitary from flat-rank moments, compare the Chebyshev enclosure with noncommuting matrix weights, test source leakage, continue the abundance equation with its equilibrium source, and recover gravity parameters by differentiating integrated growth histories. These floating-point controls complement the analytic proofs.

`reconstruct_c9_moments.py` independently reconstructs the thirteen recorded C9 moments from the finite matrices, primitive vectors and joint dephasing prescription. It retains degenerate-cluster coherences and reproduces all entries within 2.68e-14 in the tested environment. Its clustering tolerance is a numerical convention; the operator separation certificate is independent of that choice.

`certify_operator_spectra.py` defines each exact unitary as the polar factor of the corresponding stored binary matrix. Schur decompositions propose spectral centres. Outward-rounded elementary operations enclose unitarity defects, eigenbasis defects, residuals and every relevant product centre. `data/OPERATOR_SPECTRAL_CERTIFICATE.json` records the positive separation from relative eigenvalue one and hence the exactly zero atomic weight there. `validate_operator_certificate.py` checks interval matrix products against exact rational arithmetic, tests subnormal operations, and exercises known spectra and zero-phase controls. This certificate requires IEEE-754 binary64 round-to-nearest arithmetic with gradual underflow; its verifier uses no BLAS reductions.

The recorded-moment extremum and the finite operator certificate concern different specified data sets. The latter uses the complete supplied factors and their polar convention. Common-source convergence and a physical spectral gap enter the separately stated continuum theorems.

`verify_physical_identities.py` writes `data/PHYSICAL_IDENTITY_CHECKS.json`. A declared dimensionless Hermitian sequence tests the supplementary pole, energy and residue bounds against independently integrated Riesz projectors. Repeating the calculation after common negative and positive energy shifts checks the energy-origin invariance of the centred bound. A convergent source family tests the full source-error term and the positive lower residue certificate. An orthogonal source gives exactly zero residue at the same isolated eigenvalue, testing the distinction between an operator eigenvalue and a source-visible pole. Exact scalar cumulants are checked against derivatives of an independently inverted generating function. These controls test the identities; their numerical parameters are not QICT particle predictions.

The same program tests the complementary square-root residue enclosure. A separate weak-overlap source has a positive limiting weight enclosed by the amplitude bound even though the linear lower bound is negative. Its source overlap and rotated Hermitian matrix are declared control inputs. The two enclosures can be intersected; neither is claimed to dominate the other for every source.

## Data and figure provenance

`code/FINITE_C9_COMMON_STATE_AND_MOMENTS.npz` and `code/SPECTRAL_GRAVITY_SELFTEST_INPUT.npz` preserve the original supplied arrays byte for byte. The C9 state prescription was checked against the author's `compute_finite_C9_common_state.py` in `QICT_C9_STATIONARY_SCIENTIFIC_SUBMISSION_2026-10-15.zip`: it uses primitive ground vectors followed by joint neutral Cesaro dephasing. Every array needed by the present calculations is included. Particle-mesh arrays are regenerated from the initial condition defined in the code.

Figure producers are `compute_cosmology.py`, `compute_closure_tests.py`, `compute_operator_closure.py`, `compute_portal_thermal.py` and `render_particle_mesh_figure.py`. They write vector PDFs to the main manuscript's figure directory. The scalar and background parameters are explicit in source; there is no external data download during reproduction.

Expected agreement is numerical; creation timestamps and PDF or compressed-container metadata can differ between runs. Check the distributed files before regeneration from the archive root:

```bash
sha256sum -c SHA256SUMS.txt
```
