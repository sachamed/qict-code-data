# QICT — Independent full physical execution

**STATUS: BLOCKED_MISSING_INPUT (19 October 2026).**

This is a fail-closed HPC execution **preflight**, not a verified dark-matter particle or mass prediction. No numerical pole, residue, or mass has been derived.

The parent repository at immutable inspected commit `fd664503ca92d2d88007d68d6e3873f7df2beeb8` contains the canonical lattice action, Kato diagnostic data and an L24 rank-six **precontraction**, but not the 19 independently verified complete physical inputs required for executing the full pipeline.

## Reproducible starting point

```bash
git clone https://github.com/sachamed/qict-code-data.git
cd qict-code-data
git checkout qict/full-physical-execution-2026-10-19
cd QICT_FULL_PHYSICAL_EXECUTION
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/stage_guard.py --stage mass
```

The final command MUST return nonzero and record `BLOCKED_MISSING_INPUT`. This is correct scientific behavior, not an implementation failure. The guard does not permit a file name or a synthetic diagnostic to authorize physical computation.

## Map of this delivery

- `COMPLETENESS_AUDIT.json`, `MISSING_INPUTS_REPORT.md`, `HPC_BENCHMARK.json`, `RUN_PLAN.md`, `FAIL_CLOSED_STATUS.json` — status reports.
- `ACTION/` — canonical, hash-checked real-time action and metadata; author-approved commit and full operator are **not** frozen.
- `input/evidence/` — exact existing Git data blobs copied by object identity, **not** promoted to complete physical inputs.
- `input/verified/` — intentionally empty apart from its explanatory README.
- `src/`, `tests/`, `hpc/` — guards, verification primitives, refusal tests and preflight-only SLURM templates.
- `hashes/` — source Git inventory and SHA-256 values for certified available evidence.
- `docs/` — links to the reference manuscript sources. The two large PDF reference copies remain available in the accompanying local delivery archive; their binary bytes have not been pushed by this connector.

## Non-negotiable scientific gate

Build and independently certify physical nested Kato densities, complete C10 Kraus/transfer maps, boundary-to-path maps, core recoupling, rank-six embedding, full interacting `apply_U` and `apply_U_dagger`, physical matrix moments and an independently defined neutral dark-sector source `V_chi`. Only then study multi-volume isolated poles, positive residues and scale conversion.

**Do not relabel geometric buffers or diagnostic/synthetic CN6 files as physical observations.**

## Reproducibility note

This GitHub publication is a **source-addressable execution handoff**: the original archive additionally contains two reference PDFs and the offline audit's large archive-by-archive CSV inventory. The repository-native inventory in `hashes/` catalogs this Git commit's files and does not pretend that remote paths were individually SHA-256 downloaded.
