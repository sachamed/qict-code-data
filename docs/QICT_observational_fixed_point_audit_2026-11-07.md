# QICT finite-charge cosmology: verified observational fixed-point gate (7 November 2026)

## Measurement context
These are GitHub Actions executions of official Planck 2018 Plik-lite high-ell TT/TE/EE, Planck low-ell TT+EE, DESI DR2 BAO and, where included, ACT–Planck DR6 joint lensing likelihoods. All entries are fixed-point log likelihoods, not optimized inferences, evidence ratios or Hubble-tension solutions.

## Coupled CLASS, weak scalar amplitude
Run: https://github.com/sachamed/qict-code-data/actions/runs/42404824107

At H0=67.4 and Omega_scf=1e-8, g=0 gives total Planck+ACT–Planck+DESI log L = -528.564527089, while g=0.02 gives -528.580857148. Thus delta chi-square = +0.032660117. The weak-coupling control passed all computation steps but neither fit was optimized.

## QICT high-early-energy branch, ACT-inclusive evaluation
Run: https://github.com/sachamed/qict-code-data/actions/runs/42405185832

| likelihood | H0=67.4 | H0=74.93 |
|:--|--:|--:|
| Planck high-ell | -3894.237967942 | -564.039503158 |
| Planck lowTT | -65.332517951 | -67.457562863 |
| Planck lowEE | -200.665852268 | -200.440616252 |
| ACT–Planck lensing | -19.788149512 | -32.088207133 |
| DESI DR2 | -74.348771221 | -62.604497309 |
| total excluding SH0ES | -4254.373258894 | -926.630386715 |

This diagnostic uses linear lensing: CLASS HMCode is disabled. The underlying neutrino assumptions differ from the weak-amplitude control; therefore do not subtract log likelihoods between these separate setups.

## Matched standard-CDM baseline on Planck+DESI
Run: https://github.com/sachamed/qict-code-data/actions/runs/42405710766

The upstream fork option model_cdm=h denotes *Hubbleian* CDM, NOT standard CDM. The corrected control omits the option. Neutrino convention is N_ncdm=0, N_ur=3.044 for all cases below.

| case | Planck high-ell | lowTT | lowEE | DESI | total |
|:--|--:|--:|--:|--:|--:|
| LCDM, H0=67.4 | -304.926163 | -11.756472 | -198.052678 | -15.503497 | -530.238810 |
| LCDM, H0=74.93 | -2899.652093 | -12.656935 | -198.004703 | -152.391203 | -3262.704933 |
| QICT high-energy, H0=67.4 | -3895.160503 | -65.333279 | -200.665852 | -74.348771 | -4235.508405 |
| QICT high-energy, H0=74.93 | -563.631904 | -67.457453 | -200.440616 | -62.604497 | -894.134471 |

For QICT(H0=74.93) versus LCDM(H0=67.4), delta chi-square equals **+727.791322** for the same Planck+DESI data. Including the *separate*, unnormalized SH0ES Gaussian diagnostic yields **+701.684166**. These comparisons are restricted to the listed fixed points.

## Independent bounded profile experiment
Workflow: https://github.com/sachamed/qict-code-data/actions/workflows/qict-h0-profile-limited.yml

Both theories are permitted to adjust H0, omega_b, omega_cdm, n_s and log(1e10 A_s) independently, while tau_reio, neutrino settings and QICT potential parameters are fixed. Every evaluation is saved, including failures. This is a bounded exploratory numerical profile, not a posterior or a definitive reoptimization. No local-H0 prior enters its objective; SH0ES is held out for subsequent tension diagnostics.

Scientific completion requires independent converged multistart optimization or sampling, perturbation and gauge tests, consistent nonlinear lensing, and a matched joint Planck–ACT–DESI LCDM reference before any Hubble-tension solution can be claimed.
