# Quantum Information Copy Time — Spectral Consistency and Reproducibility

The package contains the exact source action and a reproducible set of mathematical and finite-sector calculations: the Haar/Floquet finite-rank obstruction, central-quotient compatibility, primitive Abelian Hamiltonian cutoffs, static external-charge GNS matrices, finite-moment nonuniqueness, and norm-controlled spectral perturbations.

**Scope.** These results concern specified theorems and finite diagnostic operators. They do not supply the full interacting charged QICT evolution, four physical spatial Kato buffers, quantum C10 Kraus contraction, copy-odd physical particle, or a mass derived from independently calibrated metrology.

## Contents

- `MATHEMATICAL_FOUNDATIONS.md` — precise statements, hypotheses, proofs, and physical interpretation.
- `evidence/` — canonical frozen action, original primitive and static-GNS certificates, and all twelve matching finite NPZ datasets.
- `src/` — algebraic checks, numerical witnesses, independent finite-evidence recomputation, and the HPC safety gate.
- `tests/` — unit, adversarial, mathematical, and reproducibility tests.
- `results/` — regenerated numerical certificates, scope classification, and source-data crosschecks.
- `SHA256SUMS.txt` — file integrity manifest.

## Reproduction

Python 3.11+ with NumPy is required. From the package root:

```sh
python src/mathematical_checks.py
python src/electric_phase_obstruction.py
python src/quantitative_stability.py
python src/verify_finite_evidence.py
python src/contract_inspection.py
python src/independent_analytical_verification.py
python src/generate_status.py
python -m unittest discover -s tests -v
sha256sum -c SHA256SUMS.txt
```

`src/hpc_gate.py --certificates certificates` deliberately refuses a physical production calculation without independently verified inputs. A successful mathematical test does not authorize a physical spectral/mass claim.

The canonical action bytes are preserved exactly. Its SHA-256 is `c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7`.

Source identification: QICT extended derivations T1472–T1475, T2231, T2240–T2242, T28.43; primitive single-plaquette and external-charge finite certificates. 

## Additional certified results

The source-backed primitive one-plaquette Hamiltonian \(8r^2+4-2(S+S^*)\) has a unique infinite-flux ground state with a gap of at least 4. `src/primitive_infinite_volume_certificate.py` certifies the four stored finite-cutoff approximations against that infinite-dimensional state using exact dyadic-rational Rayleigh and residual calculations, Temple's inequality, and the spectral theorem. The complete energy enclosures and trace-distance upper bounds are in `results/PRIMITIVE_INFINITE_VOLUME_RATIONAL_CERTIFICATE.json`.

`src/central_character_rule.py` derives the exact \(\mathbb Z_6\) central-character selection rule and exhibits a genuine quotient-invariant mixed Wilson character, without changing the frozen magnetic action. See `results/CENTER_CHARACTER_SELECTION_RULE.json`.

Both mathematical certificates retain their restricted scope: no full interacting charged Kato buffers or physical dark-sector residue have been derived. These additions make the primitive spectral convergence and quotient compatibility independently reproducible, while preserving the scientific fail-closed gate.

## Isolated primitive pole and ordered-gate consistency

The infinite-flux primitive **Hamiltonian** local resolvent has a genuinely isolated ground-state pole with certified positive residue. The exact analytic lower bound is \(Z_0\ge1/2\), sharpened using source factors and rational arithmetic to approximately \(Z_0=0.90583019808933979\). Its energy and residue enclosures and the independent finite Schur/Jacobi cross-checks are recorded in `results/PRIMITIVE_LOCAL_RESOLVENT_POLE_CERTIFICATE.json`.

The same state's overlap with the ordered elementary gauge **Floquet** update \(U_{BE}=e^{-i\phi_*K_B}e^{-i\phi_*K_E}\) demonstrates a nonzero optimized eigenphase defect near 0.03211. `src/primitive_splitstep_diagnostic.py` computes this using both the exact infinite-link Bessel convolution formula and a padded matrix exponential. This explicitly prevents promoting the Hamiltonian pole to an interacting Floquet or dark-sector pole.

The exact electric-flux identity \(K_E(r+126)-K_E(r)=126(16r+1008)\) further proves that the **elementary ordered Abelian Floquet** update commutes with a bilateral shift with no point spectrum. Therefore it has **no nonzero finite-rank Riesz spectral projection**, notwithstanding the primitive Hamiltonian's isolated rank-one pole. The exact proof and regression are in `src/primitive_floquet_shift_no_go.py` and `results/PRIMITIVE_FLOQUET_SHIFT_NO_GO.json`.
