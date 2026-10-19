# QICT scientific releases — 19 October 2026

This directory contains **exact byte-for-byte copies of every one of the 137 files** distributed in two original scientific ZIP archives received on 19 October 2026. Original directory structures and filenames have been preserved. The 55 primitive Haar/Z6 files and 82 dark-sector files include 47 Python scripts, mathematical notes, 14 LaTeX sources, PDF manuscripts and figures, matrices, prediction tables and numerical certificates.

## Release components

- [PRIMITIVE_HAAR_Z6](PRIMITIVE_HAAR_Z6/QICT_PRIMITIVE_INFINITY_CERTIFIED_2026_10_19/README.md) — finite-dimensional and primitive infinite-flux consistency checks, Haar and Z6 selection rules, primitive Hamiltonian local pole, and diagnostics excluding an unwarranted Floquet dark-particle inference. Reproduce with:
  `cd PRIMITIVE_HAAR_Z6/QICT_PRIMITIVE_INFINITY_CERTIFIED_2026_10_19 && python -m unittest discover -s tests -v`.
- [DARK_SECTOR](DARK_SECTOR/03_Code_Data_Reproducibility/README.md) — theoretical calculations for spectral diagnostics, cosmological reference models, freeze-in/out, structure growth, gravitational lensing and comparison domains. Reproduce numerical tests with the README instructions in this directory; some parts are computationally intensive.

## Verification performed prior to publication

- Both ZIP containers passed ZIP CRC tests; their original SHA256SUMS manifests had no missing or mismatched files.
- All **47 Python scripts** passed Python AST parsing.
- Primitive mathematical package: **66/66 unit tests passed**.
- Dark-sector package: the targeted finite-operator validation and operator-closure test both passed.
- The dark-sector `python code/reproduce.py --calculations-only` run completed in a separate extracted working copy, including the finite spectral tests, 3-D particle-mesh calculations and regeneration of the 76-entry calculations-only prediction register. The full LaTeX PDF regeneration step was not part of this command.
- Every distributed file's Git blob SHA has been matched against its original bytes. Full SHA-256 manifests are in [UPLOAD_VERIFICATION_MANIFEST.json](UPLOAD_VERIFICATION_MANIFEST.json).

## Scientific interpretation

These are reproducible **finite, primitive and conditional** computations. A primitive Hamiltonian resolvent pole is not a physical interacting dark-sector pole. The dark-sector package explicitly uses externally selected benchmark parameters in several cosmological calculations. The full interacting physical `U_cone`, four certified Kato buffers, physical C10 maps and neutral-sector pole are not established by either archive.

See [HPC fail-closed handoff](../../QICT_FULL_PHYSICAL_EXECUTION/README.md) before planning full physical mass calculations. No QICT physical dark-matter mass has been extracted by this publication.

Use a specific Git commit to cite immutable numerical inputs and scripts.
