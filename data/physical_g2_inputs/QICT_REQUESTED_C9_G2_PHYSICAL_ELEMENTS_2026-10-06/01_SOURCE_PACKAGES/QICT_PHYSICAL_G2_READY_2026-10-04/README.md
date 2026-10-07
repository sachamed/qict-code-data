# QICT — Physical G2 ready checkpoint (2026-10-04)

This package is the conservative handoff after the certified four-incident-plaquette charged SU(2) cluster.

## Closed in this package

- charged cluster pids 2908, 2925, 8972, 8973;
- Step-4 strict adaptive output: 77,036 retained states;
- output norm squared: 0.0034703350287145765 (reference certificate);
- strict cluster L2 error bound: 4.761260928942132e-7 < 1e-6;
- specialized native 8973 kernel regression: support-identical, max amplitude error 1.11e-16;
- consolidated checkpoint SHA-256 is recorded in `STEP4_STRICT_CHECKPOINT_MANIFEST.json`.

## Not closed / must not be promoted

This checkpoint is **not** the complete charged `U_cone`, not a lambda=1 Riesz/Kato-selected state, not `KATO_BUFFER`, not `C9^(6)`, and not G2.

The source tree contains the consumer/provider API and exact reconstruction machinery, but no generator or serialized physical L=24 objects for the global charged `U_cone`, Kato buffers, C10 shell CP maps, or boundary-to-path insertion. Therefore these objects cannot be synthesized from the local cluster without adding new physical dynamics.

## Fastest safe continuation

1. Keep the local cluster frozen; do not recompute it.
2. Produce the complete matrix-free L=24 charged `U_cone` externally/HPC, preserving `(qY,2j,iota,PiCAR=1)` and the six source columns.
3. Select the protected lambda=1 subspace with a Riesz contour **only with independently certified node lower bounds**.
4. Generate at least four nested `KATO_BUFFER_<b>.npz` plus a summable tail certificate.
5. Generate volume-specific C10 shell CP/Kraus maps and boundary-to-path insertion tensors.
6. Project to the six-column core and compute `C_n^(6)(L)`; then run the C11 positive matrix-measure reconstruction.
7. Repeat incrementally for L=32,48,64,96,128, reusing caches and moments rather than restarting each n.

Run `python verify_step4_strict.py` to reverify the checkpoint. Run `python run_next_physical_stage.py --physical-dir <L24 payload directory>` to fail-closed check the next physical inputs.
