# QICT observable-first HPC finalizer V2 — 2026-10-04

This is the fail-closed production continuation after the certified charged local Step-4 state.

## Best production architecture

Do **not** materialize the global charged Hilbert-space operator and do not propagate a huge full-state MPS only to project it back to six observables. The external physical provider supplies only:

- `metadata()`
- `dimension()`
- `source_matrix_into(out)` (or `source_matrix()`)
- `matmat_into(X,out)` (or `matmat(X)`)

The engine propagates the six source columns `(L1,L2,L3,R1,R2,R3)` and records

`C_n^(6)(L) = V^† U_L^n V`.

This changes the production memory model from dense `O(D^2)` to disk-backed `O(6D)`.

## Memory/crash policy

- all BLAS thread counts = 1;
- at most four physics workers inside a provider;
- global process-tree RSS guard = 4200 MiB on the present ~5.81 GiB host;
- three consecutive over-limit samples trigger a clean abort before OOM;
- source and propagated six-column blocks are `.npy` memmaps;
- only the two latest large state checkpoints are retained;
- moments are tiny `(nmax+1,6,6)` arrays and are retained in full;
- blockwise Gram/projection calculations avoid materializing extra copies;
- every state and `LATEST.json` commit is atomic, so rerunning resumes from the latest verified moment.

Rust is not installed in the current environment. Rewriting orchestration in Rust would not remove the missing physics. For hotspots, native C++/NumPy kernels or an external Rust implementation can sit behind `matmat_into()` without changing the scientific contract.

## Frozen physical acceptance contract

A provider is rejected unless it certifies the frozen action, correct volume, `PiCAR=1`, fixed six-column source order, complete charged `U_cone`, lambda=1 Riesz/Kato selection, at least four nested Kato buffers, a certified infinite-buffer tail, volume-specific C10 shell CP maps, boundary-to-path insertion, rank-six carrier, and an independently bounded operator/state error below `1e-6`.

The supplied 77,036-state Step-4 checkpoint is a certified **local charged cluster component**, not the global selected Kato state and not G2.

## Six-volume schedule

`VOLUME_SCHEDULE.json` freezes:

- L=24: nmax=9
- L=32: nmax=13
- L=48: nmax=21
- L=64: nmax=29
- L=96: nmax=45
- L=128: nmax=61

The six trajectories produce all 130 required matrix moments without 130 independent restarts.

## Automatic downstream chain

Once a physical provider exists for a volume:

1. `compute_cn6_matrix_free.py` generates `CN6_MOMENTS_L<L>.npz` and checkpoints.
2. `run_c11_after_cn6.py` runs numerical block-pencil/support diagnostics.
3. Rigorous C11 positivity is promoted **only** when `CN6_ERROR_CERTIFICATE_L<L>.json` supplies independent validated moment errors and a validated lower bound for the computed Toeplitz minimum eigenvalue.
4. Numerical pencil poles are never promoted as physical masses.
5. `g2_physical_acceptance_gate.py` additionally requires a rigorous spectral-support/edge (or exact flat-extension) certificate, a physical per-volume G2 result, and a six-volume limit certificate.

Run all volumes sequentially under the memory guard:

```bash
python code/run_pipeline.py \
  --providers-dir providers \
  --results-root results/physical \
  --schedule VOLUME_SCHEDULE.json
```

Provider location convention:

`providers/L24/GLOBAL_CHARGED_UCONE_PROVIDER.py`, ..., `providers/L128/GLOBAL_CHARGED_UCONE_PROVIDER.py`.

## What is still irreducibly missing from the supplied corpus

The corpus does not contain the complete volume-specific global charged `U_cone` / selected Riesz-Kato density and quantum C10 shell maps needed to implement the physical provider. Those tensors cannot be reconstructed honestly from the local four-plaquette cluster alone. V2 therefore closes the computation, memory management, checkpointing, C11 diagnostics and final acceptance software, while deliberately refusing to synthesize the missing dynamics.
