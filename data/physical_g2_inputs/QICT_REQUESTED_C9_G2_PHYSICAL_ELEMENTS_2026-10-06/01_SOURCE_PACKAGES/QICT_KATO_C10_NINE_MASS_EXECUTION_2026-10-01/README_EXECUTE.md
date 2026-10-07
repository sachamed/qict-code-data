# QICT Kato/C10 — executable nine-mass gate

This package is the compact execution bundle for the staged chain

`frozen QICT action -> Kato stationary state -> C10/Gauss compact transfer -> physical G2 -> nine charged-fermion masses`.

## Frozen inputs already included

- `input/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json` — frozen microscopic action. Required SHA-256: `c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7`.
- `input/KATO_STRICT_K0_K12_48_CP.npz` — executed Kato checkpoint, dimension 122677, 48 steps, gap 14.425846446299971.
- `input/TASK3_PHYSICAL_FULL_PASS_CONSOLIDATED_2026-09-28.zip` — executed C10/Task-3 compact full-pass evidence; 3/3 coarse/fine inequalities pass.

## Execute

```bash
python3 -m pip install -r requirements.txt
./run_all.sh
```

The first run validates the frozen action, Kato checkpoint and C10/Task-3 archive and writes `results/PREFLIGHT_STATUS.json`.

## Physical nine-mass extraction

The package intentionally contains **no invented nine-mass payload**. After the interacting physical G2 calculation has produced positive values and uncertainties for all nine charged flavors on the genuine volumes `L=4,8,12`, copy:

```bash
cp input/G2_NINE_FLAVOR_PAYLOAD.template.json input/G2_NINE_FLAVOR_PAYLOAD.json
```

and fill only the calculated G2 values/errors plus the frozen quark schemes. Then execute `./run_all.sh` again.

The extractor fits two declared finite-volume models, freezes a SHA-256 **before any comparison**, and writes:

- `results/NINE_MASS_RESULT.json`
- `results/NINE_MASS_PRECOMPARISON.sha256`

For charged leptons the observable is an inclusive dressed support edge unless an isolated pole is independently certified. For confined quarks the values must be short-distance renormalized masses in one frozen scheme and scale. `1:3:5` is never used as an input, fit target, ordering rule or acceptance criterion.
