# Quantum Information Copy Time — Reproducible Code and Data

This is the public computational repository for **Sacha Mohamed's Quantum Information Copy Time (QICT)** research. It contains finite calculations, benchmark comparisons, source archives and an explicitly **fail-closed** HPC preflight. It is not a validated execution of the complete interacting QICT operator or a physical dark-matter mass prediction.

## Scientific status

- Finite receiver, SU(2) rotor, BCC transport and reduced-flavor checks are available under their stated assumptions.
- Neutrino-shape results compare a **reduced** finite Takagi spectrum with external oscillation benchmarks; they are not a prediction of the interacting physical neutrino masses.
- The Kato buffers, quantum C10 maps, boundary-to-path maps, certified global U-cone and physical CN6 moments needed for a fully interacting spectral extraction are **not** provided in complete certified form.
- No isolated physical dark-sector pole, positive physical residue or mass has been established by the available preflight.

## Repository entry points

- `code/`, `data/`, `results/`: currently available computational resources.
- `snapshot-2026-10-17/`: frozen historical source snapshot.
- `QICT_FULL_PHYSICAL_EXECUTION/`: HPC preflight, missing-input inventory and fail-closed guards.
- `REPRODUCE_ALL.py`: reproduces finite numerical calculations. Optional publication audits require separately supplied `paper/` and `cover_letter/` sources.

To set up and reproduce the supported numerical calculations:

```bash
git clone https://github.com/sachamed/qict-code-data.git
cd qict-code-data
python -m pip install -r code/requirements.txt
python REPRODUCE_ALL.py
python code/verify_neutrino_shape_obstruction.py
python -m unittest discover -s QICT_FULL_PHYSICAL_EXECUTION/tests -v
```

The numeric pipeline may be computationally demanding. Do not interpret successful arithmetic or synthetic/diagnostic tests as full interacting physics.

The optional publication-source audit is invoked by `python REPRODUCE_ALL.py --publication-audits`. It intentionally refuses to run without the corresponding manuscript source files. The optional `--compile-pdfs` requires a TeX toolchain and those same source packages.

## Integrity and provenance

Git commit IDs identify exact immutable source revisions. Files `MANIFEST_SHA256.txt` and `GITHUB_EXPORT_SHA256.txt` document a **historical export**, not every later addition. Generate and verify an up-to-date tree-wide SHA-256 manifest with `tools/repository_integrity.py`, as described in its help and the CI workflow. Do not treat an old checksum list as proof that the current checkout is complete.

## HPC handoff

```bash
cd QICT_FULL_PHYSICAL_EXECUTION
python src/stage_guard.py --stage mass
```

This command must return a non-zero exit code and `BLOCKED_MISSING_INPUT` until all physical contracts have independent evidence. The supplied SLURM files are preflight guards, not actual full-volume HPC executions.

## Source citations

Use the exact repository commit when citing code. Mathematical statements and assumptions belong to the associated QICT manuscript or book, rather than being inferred from file names.

Author: **Sacha Mohamed** · ORCID: 0009-0005-2078-2032.
