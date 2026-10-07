# Quantum Copy-Time Geometry — Code and Data

Reproducibility repository for the manuscript:

**Quantum Copy-Time Geometry: Standard-Model Gauge Algebra, Family Structure, and a Parameter-Free BCC Photon Signature**

Author: **Mohamed Sacha**  
Independent Researcher, Casablanca, Morocco  
ORCID: 0009-0005-2078-2032

## Repository scope

This repository contains the computational and machine-readable materials supporting the submission. It is intentionally limited to reproducibility assets rather than publication PDFs.

- `code/` — executable calculations, symbolic checks, validation scripts, and numerical pipelines.
- `data/` — primary numerical inputs, finite-regulator payloads, SU(2) state data, and source archives used by the calculations.
- `results/` — regenerated numerical outputs and machine-readable certificates.
- `validation/` — publication-claim, source-consistency, and scientific-framing audits.
- `REPRODUCE_ALL.py` — top-level reproduction entry point.
- `MANIFEST_SHA256.txt` — original submission-package checksum manifest.
- `GITHUB_EXPORT_SHA256.txt` — checksum manifest for this repository export.

## Main reproducibility targets

The distributed calculations cover, among other components:

- finite-algebra and gauge-structure checks;
- anomaly and hypercharge consistency;
- BCC photon angular invariants;
- finite-regulator Kato/Riesz calculations;
- charged and neutral spectral observables;
- flavor-sector diagnostics;
- RG operator-space bookkeeping and convergence diagnostics;
- phenomenology and machine-readable statistical outputs.

The normalized BCC photon signature used in the manuscript is fixed geometrically, with high-symmetry pattern `9:21:25`, spherical variance `1/525`, and invariant `R_gamma = 3/4`.

## Reproduction

From the repository root:

```bash
python REPRODUCE_ALL.py
```

Individual calculations can also be run from `code/`. Some finite-regulator/HPC stages use precomputed payloads distributed under `data/` so that the numerical provenance is preserved exactly.

## Data integrity

All files in this export are covered by `GITHUB_EXPORT_SHA256.txt`. Verify with:

```bash
sha256sum -c GITHUB_EXPORT_SHA256.txt
```

## Scientific status

The repository separates exact algebraic/theorem-level results, finite-regulator numerical certificates, and continuum/RG diagnostics in machine-readable form. The corresponding JSON certificates under `results/` and `validation/` record the status and provenance of each computational layer.

## Citation

When citing the code or data, cite the associated manuscript title above and this repository. A journal/DOI citation can be added after publication.

## Correspondence

Mohamed Sacha  
Independent Researcher, Casablanca, Morocco  
ORCID: 0009-0005-2078-2032  
Email: www.sachamed@gmail.com
