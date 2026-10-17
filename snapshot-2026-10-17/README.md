# QICT Computational Source Snapshot — 17 October 2026

This directory preserves all **428 source-code files** from the computational section of the revised scientific submission, maintaining the original relative paths under `03_Code_Data_Reproducibility/`.

The snapshot includes reproduction drivers, analytical and numerical source programs, Kato/GNS spectral tests, charged operator and tensor calculations, and the Task evidence scripts. It is additional to the repository's previous `code/`, `data/`, `results/`, and `validation/` directories; those are not overwritten.

## Contents and checksums

- `SHA256SUMS.txt`: original file SHA-256 hashes, with paths relative to this snapshot directory.
- All 428 source-code files are located at the exact paths listed in the checksum manifest.
- `reproduce.py` is a top-level reproduction entry point. Some runs require numerical data, state checkpoints, or external packages that are included in the submission archive and are not all duplicated in this Git repository.

Verify locally:

```sh
cd snapshot-2026-10-17
sha256sum -c SHA256SUMS.txt
```

The complete scientific submission bundle also includes numerical payloads, source tables, LaTeX documents and extended proofs. The source snapshot does **not** imply that every calculation has been independently rerun, that the entire scientific dataset is mirrored here, or that continuum physical interpretations have been established beyond the statements in the manuscript.
