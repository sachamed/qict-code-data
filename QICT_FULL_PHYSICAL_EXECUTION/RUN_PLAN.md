# Execution protocol — gated HPC handoff

The audited source of the action is `c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7` at GitHub commit `fd664503ca92d2d88007d68d6e3873f7df2beeb8`. Freeze the **author-designated** commit, boundary convention, Kato contour and scale choice before any production run.

1. Record environment: Python, NumPy/SciPy, MPI, OpenMP, compiler, BLAS, CUDA/GPU, nodes and memory; check action SHA.
2. Independently construct four Kato/Riesz physical buffers, verify positive trace-one densities, Riesz gap and convergence **tail bound**.
3. Construct exact physical C10 shell maps; verify CP/Kraus and trace maps; build boundary-to-path.
4. Complete SU(2)/SU(3)/CAR core recoupling and six-column isometric rank-six embedding.
5. Implement **full physical** matrix-free interacting `apply_U`/`apply_U_dagger` with Gauss/Haar projection and error control.
6. Compute actual CN6 matrices with `C0=I6`, error bars, correct six-source order and rigorous Toeplitz positivity.
7. Separately establish a neutral, copy-odd physical source `V_chi`; charged CN6 alone cannot identify a hidden odd-sector pole.
8. Compute projected resolvent; independently certify an isolated pole, positive nonzero residue and separation from continuous support.
9. Repeat at L=24,32,48,64,96,128, depths 9,13,21,29,45,61 with buffer, regulator and pole extraction checks.
10. Convert the dispersion to physical energy/mass only after a separately calibrated QICT clock; propagate all errors.
11. Run actual L=8,12,16 benchmarks **after** the full provider is available; only then select CPU/GPU/SLURM allocations.

The existing SLURM templates perform refusal/preflight only. They are not real physics job submissions, nor are they HPC performance benchmarks.

**Acceptance:** physical prediction requires independently verified source, global operator, convergence, pole/residue, multi-volume limit and no target mass injected. Otherwise status must remain `BLOCKED_MISSING_INPUT`.
