#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parameters', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p = Path(args.parameters)
    d = json.loads(p.read_text())
    L = int(d['lattice_L'])
    order = int(d['receiver_central_order'])
    source = list(d['source_order'])
    rb = d['reduced_lepton_benchmark']
    route_lengths = [int(x) for x in rb['route_lengths']]
    defects = [int(rb['route_cycle_length']) - 2*x for x in route_lengths]
    assigned = sorted(int(v) for v in rb['channel_defect_assignment'].values())
    checks = {
        'plaquette_count_equals_3L3': int(d['plaquette_count']) == 3*L**3,
        'receiver_phase_equals_pi_over_63': math.isclose(2*math.pi/order, math.pi/63, rel_tol=0.0, abs_tol=1e-15),
        'six_unique_sources': len(source) == 6 and len(set(source)) == 6,
        'source_order_is_declared_family_chirality_order': source == ['L1','L2','L3','R1','R2','R3'],
        'route_defects_match_channel_assignment': sorted(defects) == assigned,
        'primitive_positive_bridge': int(rb['primitive_positive_bridge_multiplier']) == 1,
        'positive_representation_cutoffs': all(int(v) >= 0 for v in d['representation_cutoffs'].values()),
        'positive_moment_depth': int(d['moment_depth']) >= 1,
    }
    result = {
        'classification': 'MODEL_PARAMETER_CONSISTENCY',
        'parameters_sha256': sha256(p),
        'checks': checks,
        'derived': {
            'plaquette_count': 3*L**3,
            'receiver_phase_radians': 2*math.pi/order,
            'route_defects': defects,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if not all(checks.values()):
        raise SystemExit('Model-parameter validation failed.')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
