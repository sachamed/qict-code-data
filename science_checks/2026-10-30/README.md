# QICT — independently reproducible validation checks (30 October 2026)

This folder contains **new, canonical, stand-alone validation materials**, not a duplicate of the historical submission snapshots. Its contents are deliberately scoped to the mathematical or experimental-reference tests that can be reproduced directly from the committed files.

## Charged-lepton experimental-reference test

Run `python3 charged_leptons/double_ratio_independent.py` (Python 3.11+). The reference value of the fixed reduced coefficient is retained, and no charged-lepton mass is fitted. The resulting `delta_mu`, `delta_tau`, and `D` are **targets**, not predictions from the complete C8/C10 transfer.

## Finite-moment spectral nonuniqueness

Run `python3 spectral_moments/code/finite_moment_nonuniqueness.py` with NumPy. The script constructs two positive spectral supports with identical first nine nontrivial moments and different minimum positive supports. It overwrites the local results JSON. The six-source extension follows by tensoring with the identity on the source space.

## Scope and provenance

`SCIENTIFIC_STATUS.json` documents validated checks and open requirements. The full scientific release also includes the 24-link SU(2) 2,081-state auxiliary Galerkin data; those large binary matrices have **not** been included in this small GitHub commit. Their omission is explicitly recorded; the auxiliary returns are not charged C8/C10 amplitudes.

Two prior GitHub commits serve different roles:
- `02480ed023a9137eb05f12cb9c173a51df747daa`: historical full-source snapshot.
- `78cfd3f07f1e28cd0101365bb69b3776afe8bb29`: C10 coherent-recursion validation.

No change to the full microscopic action or to existing historical files is implied by this targeted update. Before consolidating the remaining local programs and data, compare blob hashes, follow imports, and confirm that each retained certificate has its required inputs.
