# Verification of the QICT Haar and Spectral Consistency Calculations

## Scope and provenance

The assessed source is the QICT canonical lattice action, authenticated by SHA-256
`c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7`.
The corresponding source derivations are T1472–T1475, T2231, T2240–T2242 and T28.43. Primitive-state and static-source numerical evidence was restored from its original datasets and checked against the issued certificates.

## Verified mathematical results

| Result | Mathematical statement | Verification |
|---|---|---|
| Haar endpoint | Under invariant-carrier and commutation hypotheses, a finite-rank nonzero Floquet Riesz projector cannot exist | Direct spectral argument; assumption stated |
| Central quotient | The frozen magnetic phase differs under the central generator | Exact six-lift cost computation |
| Primitive state | The one-plaquette unwrapped Hamiltonian has a uniform gap at least 4 | Min–max bound and four finite cutoffs |
| Electric Floquet | Normalized phase-one projectors at radius R and 2R have trace distance tending to 1/2 | Exact integer arithmetic |
| Compressed residue | Infinite rank can coexist with a finite positive source residue | Exact auxiliary direct-integral construction |
| Moment depth | Finite identical moments can represent atomic or continuous measures | Roots-of-unity counterexample |
| Stability | Common-gap resolvent estimate bounds changes in projectors and residues | Resolvent identity and finite perturbation controls |
| Static GNS | Six-channel C0 and Toeplitz matrices are consistent to floating-point tolerance | Independent screening of eight original datasets |

## Corrections incorporated

1. The action/gauge convention distinguishes the product-group link cover from the faithful matter quotient. The central quotient is tested for **all six lifts**. No action parameters or original action bytes were altered.
2. The physical endpoint theorem is explicitly conditional on preservation of the Haar cycle and commutation of the full unitary. Its conclusion is not extended to other state representations.
3. All **twelve original** primitive and static-GNS binary datasets referenced by the source certificates are included and checked byte-for-byte.
4. The odd-source validation enforces finite matrix entries and the source Gram identity, in addition to copy parity and unitarity.
5. Schur–Feshbach comparisons use linear solves and reject near-singular complementary blocks rather than relying on explicit inversion.
6. Dimensional phase/clock conversion rejects infinite inputs and is distinguished from a mass certificate based on independent metrology.
7. The contour quadrature truncation estimate is separated from floating-point roundoff.
8. Primitive positive-density assertions use Gram factors; small negative eigenvalues of computed floating-point densities are not misclassified as physical negative states.
9. The static-source Toeplitz check is labeled numerical screening, not rigorous interval certification.
10. A self-declared reviewer string is not accepted as cryptographic evidence. The HPC gate remains closed.
11. Source checks on primitive states now include finite-dimensional Hamiltonian residuals and the two omitted exterior flux amplitudes.
12. All computations can be rerun through a single script with a source-data hash check and an adversarial test suite.

## Distinction of states and observables

The four primitive eigenvectors are **four flux cutoffs of one Abelian plaquette Hamiltonian**. Their Gram factors and gaps do not establish four compatible interacting charged Kato buffers. The eight static GNS source arrays describe two external charge sectors at four cutoffs, rather than the six chirality- and Gauss-conditioned charged matter sources.

The auxiliary matrix pole at z=1 has a nonzero residue and agrees under three exact or numerical evaluations on that auxiliary operator. This remains a test of the extraction procedure and does not identify a QICT dark particle. In particular, the test does not certify the spectral overlap or a uniform gap of the interacting QICT operator.

## Physical conditions not yet verified

The provided action source does not uniquely specify a physical matrix-free implementation of every route-first SU(2)/SU(3)/CAR gate. The physical interacting Kato-state selector, complete C10 maps, common six-source embedding, odd neutral source, exact physical moment sequence, uniform multi-volume spectral separation, nonvanishing physical pole weight, independent time-scale calibration, and blinded physical replication have no complete physical evidence in the assessed package.

**Scientific disposition:** the mathematical and finite-data verification passes within its explicit scope. A full interacting spectral mass prediction is not established by these calculations. HPC production on that basis remains conditional on independently verified physical input tensors and operators.

## Reproducibility

The published suite contains 66 executable tests, including adversarial failures, exact arithmetic identities, independent numerical screening, and checks of all source binary hashes. `python reproduce.py` regenerates the results. The file integrity manifest is generated after reproduction. Numeric metrics are double-precision results in the current environment; byte-identical artifacts are guaranteed only by the recorded content hashes, not presumed across different BLAS implementations.

## Further mathematical closure: infinite primitive state and central descent

The primitive one-plaquette Hamiltonian has a unique positive infinite-flux ground state and a spectral gap of at least four. Its rank-one Riesz projector is separated from the rest of the infinite-dimensional Hamiltonian spectrum by a gap of at least two along the circle \(|z|=6\). For the four authenticated primitive source vectors, exact dyadic-rational evaluation of the full infinite-operator residual—including boundary terms—provides outward-rounded Rayleigh/Temple energy enclosures and rigorous trace-distance bounds. The weakest of the four bounds is \(4.425\times10^{-15}\); the most accurate reported one is \(2.2\times10^{-17}\). Exact dyadic pairwise source overlaps independently verify common flux-basis embedding and yield sharper numerical comparisons.

The central quotient compatibility condition for representations with integer hypercharge normalization, twice-spin, and SU(3) triality is \(Y+3(2j)+2t\equiv0\;(\mathrm{mod}\;6)\). A centre-admissible mixed Wilson character was constructed and checked against all six central lifts. This criterion does not alter the original plaquette action, whose non-descent is already certified.

Neither theorem creates the missing physical charged gates, four interacting Kato buffers, quantum C10 maps, physical six-chiral-source transfer moments, nonvanishing dark pole, or independent clock. Those requirements remain blocked, and all existing fail-closed safety gates remain enabled.

## Isolated primitive pole and the distinction from Floquet evolution

The infinite-flux primitive Hamiltonian has a simple isolated pole in the local resolvent \(G_{00}(z)\). The exact inequality \(H\ge8r^2\), combined with \(E_0\le4\), proves a local residue \(Z_0\ge1/2\) without any finite-volume hypothesis. Rationally certified trial-state overlap yields the sharper interval \(0.9058301980893397686289319480855 \le Z_0\le0.9058301980893398126289319480856\); the joint energy and residue certificate is supplied with the source SHA-256. The Schur continued-fraction and even-parity Jacobi diagonalization are independent algorithmic checks. A numerical precision issue in the selected-eigenvalue bisection driver has been avoided by using the full tridiagonal eigensolver.

For the ordered elementary gauge Floquet update \(U_{BE}=\exp(-i\phi_*K_B)\exp(-i\phi_*K_E)\), the same primitive ground-state trial vectors show an optimized eigenphase defect around \(0.03211158\). The Bessel-kernel and padded-matrix computations agree to about \(10^{-16}\). This explicit contrast prevents the physical interpretation of the primitive Hamiltonian pole as an interacting Floquet particle.

## Exact single-plaquette Floquet obstruction

The frozen electric phase obeys the integer shift identity \(K_E(r+126)-K_E(r)=126(16r+1008)\). The elementary magnetic and electric gates therefore commute with the bilateral shift \(T_{126}\), which has no nonzero \(\ell^2(\mathbb Z)\) eigenvectors. Consequently the ordered elementary Floquet update has no nonzero finite-rank Riesz projector, in sharp contrast with the isolated ground-state projector of the unwrapped Hamiltonian. The proof does not presume that the full route-first interacting gate preserves this shift symmetry.
