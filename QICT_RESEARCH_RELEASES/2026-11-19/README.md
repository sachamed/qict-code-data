# QICT — Causal Dynamics, Spectral Structure, and Emergent Geometry
## Research update · 19 November 2026

This directory reproduces a selected set of **finite, independently executable checks** from the corresponding QICT/C8–C10 submission. The checkers and frozen JSON outputs preserve the source classification and do not constitute physical mass or gravity predictions.

### Provenance

Source: `QICT_Causal_Dynamics_Spectral_Structure_Emergent_Geometry_Submission_Audited_2026-11-19.zip`

- SHA-256: `223cff38d7a1dbb1cb4cd9f90e3875f2aa0726e47a27bdd11edfc7e869eea8da`
- Archive size: 199,192,790 bytes; 2,248 ZIP entries
- The complete ZIP is **not hosted here**. This update publishes a curated, executable subset; do not identify this directory with a byte-complete copy of the submission.
- Existing files, historical research releases, and the fail-closed physical-execution preflight are preserved.

### Published checks

| File | Corresponding path in submission | Scope |
| --- | --- | --- |
| `code/verify_clock_holonomy.py` | `4_Code_and_Data/Clock_Geometry_Holonomy/verify_clock_holonomy.py` | Cycle-phase reconstruction on a fixed 4-vertex graph; 200 random comparisons |
| `code/verify_phase_floquet.py` | `4_Code_and_Data/Clock_Geometry_Holonomy/verify_phase_floquet.py` | Relative-phase and projective-metric invariance; Floquet aliases |
| `code/verify_passive_schur.py` | `4_Code_and_Data/Flavor_Kinetic_Audit_2026/verify_passive_schur.py` | Finite Hermitian Schur-slope identity and positivity |
| `results/clock_holonomy_checks.json` | `4_Code_and_Data/Clock_Geometry_Holonomy/clock_holonomy_checks.json` | Archived graph-comparator output |
| `results/PASSIVE_SCHUR_CHECK.json` | `4_Code_and_Data/Flavor_Kinetic_Audit_2026/results/PASSIVE_SCHUR_CHECK.json` | Archived 720-check numerical summary |

### Reproduce

Install `numpy` and `scipy`; then run from this directory:

```bash
python code/verify_clock_holonomy.py
python code/verify_phase_floquet.py
python code/verify_passive_schur.py
```

These programs write their generated JSON files alongside the scripts (and `code/results/` for the Schur test). Compare the generated files with the archived `results/` or inspect the numerical diagnostics. The scripts' output paths were retained from the scientific source archive to preserve exact source bytes.

### Scientific interpretation

The clock–holonomy test proves phase identifiability *if* all independent loop holonomies of a connected graph are physically determined. It does not calculate the seven-cell holonomies from the interacting QICT evolution. The passive Schur-slope identity applies to finite Hermitian energy-independent Hamiltonians and an orthonormal projected source space; identifying its kinetic matrix with the QICT physical pole slope is a separate calculation. The neutrino and quark kinetic fits elsewhere in the submission use external measurements for calibration and must not be described as blind mass predictions.

The full C8/C10 interacting charged kernel, regulator-stable physical pole residues, geometrical phase action, positive massless tensor pole and absolute clock normalization remain the physical validation targets.
