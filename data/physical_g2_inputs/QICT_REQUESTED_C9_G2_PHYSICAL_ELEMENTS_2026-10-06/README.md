# QICT requested C9/G2 physical elements bundle — 2026-10-06

This archive collects every currently available artifact relevant to the requested L=24 physical payload chain:
KATO_BUFFER, C10_SHELL_MAPS, BOUNDARY_TO_PATH, CORE_RECOUPLING, RANK6_EMBEDDING, GLOBAL_CHARGED_UCONE, and CN6_MOMENTS.

## Scientific status
The archive is fail-closed. It does NOT fabricate or rename control/synthetic objects as physical payloads.
The current hard-stop scan reports that the physical BOUNDARY_TO_PATH, volume-matched CORE_RECOUPLING/RANK6 join, full charged U_cone provider, and physical CN6 moments are not serialized/instantiated. The included CN6 moment file is explicitly under `synthetic_c11` and is retained only as a backend/control dataset.

## Layout
- `00_CONTRACT_AND_STATUS/`: authoritative physical contract and hard-stop/status files.
- `01_SOURCE_PACKAGES/`: complete small relevant packages extracted from the two QICT_ALL_RESULTS archives.
- `02_EXTRA_CHECKPOINTS/`: additional B0 Kato/Riesz checkpoints from Ucone/Auth81 packages.
- `03_REQUESTED_ELEMENTS/`: per-element view with all available inputs/candidates and a STATUS.txt.
- `REQUESTED_ELEMENTS_STATUS.json`: machine-readable physical availability.
- `SHA256_MANIFEST.csv`: hashes for every file in this bundle.

The bundle is intended as the exact restart package for the remaining numerical contraction, not as a false FULL_PASS certificate.


## Repository-clean layout
Exact duplicate payloads are canonicalized in this public repository. 01_SOURCE_PACKAGES
retains provenance paths; repeated bytes are represented by Git symbolic links to the canonical
files. The fully redundant 02_EXTRA_CHECKPOINTS mirror was removed. 03_REQUESTED_ELEMENTS
retains the element-status records and the CN6 compatibility path used by the contract audit.
DEDUPLICATION_MAP.csv at repository root records every removed or aliased path.
