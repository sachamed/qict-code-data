# QICT numerical update — 22 November 2026

This release contains source-identical verified code and small numerical certificates
from the journal submission "Quantum Information Copy Time: Causal Dynamics,
Spectral Structure, and Emergent Geometry."

## Source archive
- Archive: QICT_Soumission_Finale_5_Dossiers_2026-11-22.zip
- SHA-256: dbdc894dd9c3f4c2e9db3e73d350618ecf6bc85f63a7b6a787ff990565bb19a0
- Size: 128620804 bytes
- This directory is a curated source/data update, **not** a complete copy of
  the code/data directory in the journal ZIP.

## What changed
- Coupled SU(2) shared-link theta-graph code now uses a seeded eigenvector
  and three sparse-LU inverse iterations to precondition a rational witness.
  The current source SHA-256 is 8530d86d60a151b22bfa6a13aaa32f2656258917e6039483a496ccd674048da5.
  The historical source SHA-256 was c7b3eadf3b4ab9d37db14d6de1c8dbaab2a7f2ef056ef1f3cd1e0fe0a2ddecf7.
- The independent CG-contraction verifier and its two source dependencies are
  retained byte-for-byte from the archived reproducibility code.
- The finite Schur-resolvent checker tests the norm bound for a spectrally
  separated Hermitian block, and tests the operator-order statement only when
  the block is positive with M >= Delta I. It also supplies an indefinite
  counterexample to the unjustified operator-order claim.
- The SU(2) validation report records 360 independent contractions (208
  off-diagonal), an exact certified threshold of 59, and a finite-model
  ground-gap lower bound of approximately 21.78416.

## Reproduction
From the release directory with numpy and scipy installed:

    python schur/verify_schur_resolvent.py
    python reproducibility/code/compute_coupled_su2_spin_network.py
    python reproducibility/code/verify_coupled_su2_vertices.py

The SU(2) scripts expect their dependencies in the same code directory.
They write outputs under reproducibility/results/. The archived numerical
binary state .npz and full 91 KB SU(2) certificate remain in the journal ZIP,
not in this curated GitHub update. The JSON summaries here are not substitutes
for the omitted binary state when checking byte-for-byte reproduction.

## Scope and limitations
These checks concern finite SU(2) and Schur benchmarks; they do not establish
the six-source interacting C8/C10 pole closure, absolute physical masses,
or the emergent massless-graviton sector. Existing research snapshots are
preserved. This release does not include the separate QICT spin-transport article.
