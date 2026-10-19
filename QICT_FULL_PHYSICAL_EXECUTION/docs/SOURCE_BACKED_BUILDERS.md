# Source-supported input construction for QICT

The original canonical action has SHA-256 `c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7`. The metadata file `ACTION/frozen_action.json` already exists and identifies the same action. Its `author_git_commit`, complete operator, physical clock and Kato contour remain unset; do not replace them with guesses.

`python src/reconstruct_b0_inputs.py --out results/b0_source_backed` verifies the original rank-one factorized neutral **b=0** Riesz/Kato input arrays for truncations J31, J60 and J80, checks each payload against its original certificate and emits six **diagnostic** Gram factors, with provenance. The output is deliberately named `B0_FACTORIZED_..._DIAGNOSTIC.npz`; these are *not* the four physical buffers `KATO_BUFFER_0.npz` through `KATO_BUFFER_3.npz`.

To obtain the latter, one must compute the actual interacting charged lambda=1 spectral subspaces and volume-compatible density matrices, independently prove a uniform Riesz gap, quantify quadrature and solver errors, and bound the trace-distance tail in a compatible Hilbert embedding. Neither positive finite B0 certificates nor the geometric nesting files establish those requirements.

The remaining C10 Kraus maps, quantum boundary-to-path contraction, SU(2)/SU(3)/CAR core recoupling, certified rank-six charged carrier, full Ucone and physical CN6 moments cannot be created from the available partial precontraction without supplying their missing physical operator definitions. The copy-odd neutral dark channel remains a separate input.

**Status: BLOCKED_MISSING_INPUT; no physical pole, residue, or mass predicted.**
