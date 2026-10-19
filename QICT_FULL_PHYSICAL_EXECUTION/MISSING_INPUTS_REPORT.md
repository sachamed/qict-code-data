# Missing physical inputs — independent HPC handoff

**Decision: BLOCKED_MISSING_INPUT.**

The 19 canonical file names are absent **as independently certified complete physical objects**. This statement does not mean that similarly named diagnostics, partial contractions, templates, or geometry maps are nonexistent.

1. `input/verified/KATO_BUFFER_0.npz` — **BLOCKED_MISSING_INPUT**
2. `input/verified/KATO_BUFFER_1.npz` — **BLOCKED_MISSING_INPUT**
3. `input/verified/KATO_BUFFER_2.npz` — **BLOCKED_MISSING_INPUT**
4. `input/verified/KATO_BUFFER_3.npz` — **BLOCKED_MISSING_INPUT**
5. `input/verified/BUFFER_CONTRACTION_CERTIFICATE.json` — **BLOCKED_MISSING_INPUT**
6. `input/verified/C10_SHELL_MAPS_0.npz` — **BLOCKED_MISSING_INPUT**
7. `input/verified/C10_SHELL_MAPS_1.npz` — **BLOCKED_MISSING_INPUT**
8. `input/verified/C10_SHELL_MAPS_2.npz` — **BLOCKED_MISSING_INPUT**
9. `input/verified/C10_SHELL_MAPS_3.npz` — **BLOCKED_MISSING_INPUT**
10. `input/verified/BOUNDARY_TO_PATH_0.npz` — **BLOCKED_MISSING_INPUT**
11. `input/verified/BOUNDARY_TO_PATH_1.npz` — **BLOCKED_MISSING_INPUT**
12. `input/verified/BOUNDARY_TO_PATH_2.npz` — **BLOCKED_MISSING_INPUT**
13. `input/verified/BOUNDARY_TO_PATH_3.npz` — **BLOCKED_MISSING_INPUT**
14. `input/verified/CORE_RECOUPLING_L24.npz` — **BLOCKED_MISSING_INPUT**
15. `input/verified/RANK6_EMBEDDING_L24.npz` — **BLOCKED_MISSING_INPUT**
16. `input/verified/GLOBAL_CHARGED_UCONE_PROVIDER.py` — **BLOCKED_MISSING_INPUT**
17. `input/verified/GLOBAL_CHARGED_UCONE_PASS_CERTIFICATE.json` — **BLOCKED_MISSING_INPUT**
18. `input/verified/CN6_MOMENTS.npz` — **BLOCKED_MISSING_INPUT**
19. `input/verified/CN6_SHAPE_AND_C0_CERTIFICATE.json` — **BLOCKED_MISSING_INPUT**

## Why evidence is insufficient

1. `KATO_STRICT_K0_K12_48_CP.npz` is not the four specified normalized density matrices with uniformly certified Riesz gaps and independent tail bounds.
2. The extant C10 shell geometry data are not verified physical CP/Kraus maps for all buffers.
3. The L24 N9 rank-six payload is a precontraction, not a complete globally interactive `U_cone` matvec and adjoint.
4. A control/synthetic CN6 moments file must not enter a physical pole analysis.
5. Physical neutral/dark-sector source `V_chi`, stability/selection rules, common-volume branch and time calibration remain to be demonstrated.
6. No physical kernel-level benchmarks can be obtained without `apply_U`.

**Validation gate:** each required artifact needs an independent SHA-256 provenance statement, compatible action hash, non-synthetic classification, physical normalization/positivity bounds and a second independent sign-off.

**Consequences:** do not claim a pole, positive residue, mass (GeV), dark-matter candidate, or cosmological relic prediction.
