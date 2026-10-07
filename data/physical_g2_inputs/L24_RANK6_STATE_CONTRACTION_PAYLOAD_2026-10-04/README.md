# L24 charged rank-6 state/contraction payload — 2026-10-04

This bundle advances the first physical QICT G2 task at `L=24, n=9` without promoting synthetic or perturbative surrogates.

## What is now assembled

`L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz` combines the authentic n=9 covering-lattice geometry, the exact first Floquet-Kato tangent, the exact SU(2) connected second-Krylov pair payload (losslessly decompressed from the accepted 48-class representation), and the exact C10 R12 geometric/shell IR. Sidecar JSON files contain the exact charged local U(1)xSU(2) matter/Gauss/chiral witnesses and the exact second-order Kato seed.

This is a **precontraction payload**, not a selected state. It is deliberately marked `physical_promotable=false`.

## Exact new progress

The previously failed Windows re-execution of the connected SU(2) second Krylov layer is bypassed without approximation. The accepted class-compressed result already serialized every one of the 261,346 connected plaquette pairs by a `pair_class_id` into 48 exact translation/orientation classes. The bundle expands that map into explicit per-pair `branch_count`, `branch_code`, and `branch_amplitudes` arrays. The decompression certificate checks exact class populations and branch norm preservation.

## What still prevents a physical C9^(6)(24)

The source submission requires the following genuinely dynamical objects that are not present in the supplied bytes: a complete charged `U_cone` matrix-free application with PiCAR=1; a lambda=1 selected Riesz/Kato density and at least four nested certified buffers; quantum C10 shell CP/Kraus maps; a boundary-density-to-core-path insertion tensor; and a volume-matched six-column core embedding. Geometry or perturbative tangents cannot replace them.

Place future physical files under `physical/L24/` and run:

```bash
python code/check_physical_payload_completion.py
```

The checker fails closed until every physical dependency needed for `CN6_MOMENTS.npz` is present.

To rebuild the exact precontraction NPZ and certificate from the included components:

```bash
python code/build_precontraction_payload.py
```
