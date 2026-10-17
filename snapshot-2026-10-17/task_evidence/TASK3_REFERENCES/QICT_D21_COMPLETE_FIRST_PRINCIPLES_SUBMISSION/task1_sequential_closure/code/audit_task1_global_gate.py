#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, os
from pathlib import Path

ROOT=Path('/mnt/data/qict_src/final_submission')
V6=Path('/mnt/data/QICT_D21_TASK1_SEQUENTIAL_CHECKPOINT_2026-09-17_v6')
NEW=Path(__file__).resolve().parents[1]

def load(p):
    return json.loads(Path(p).read_text())

def exists(p): return Path(p).is_file()

def pass_status(p,prefix='PASS'):
    if not exists(p): return False,None
    o=load(p); return str(o.get('status','')).startswith(prefix),o

checks=[]
def add(name,ok,evidence,detail):
    checks.append({'requirement':name,'pass':bool(ok),'evidence':evidence,'detail':detail})

add('Exact microscopic operator formula present',
    'U_{\\rm micro}' in (ROOT/'supplements/QICT_Supplement_AH_Microscopic_UV_Leakage_Kernel_and_Exact_Scale_Criterion.tex').read_text() or
    'mathbb U_{\\rm micro}' in (ROOT/'manuscript/QICT_D21_MAIN_MANUSCRIPT_FINAL.tex').read_text(),
    'manuscript/QICT_D21_MAIN_MANUSCRIPT_FINAL.tex; Supplement AH',
    'Route-first macro-step is explicitly defined as a product of interaction, magnetic, electric, collision and three covariant streams.')

ok,o=pass_status(NEW/'results/TEMPORAL_SEMIGROUP_ROUTE_UNIQUENESS.json')
add('Route completion unique',ok and 'only within' in o.get('scope',''),
    'task1_new/results/TEMPORAL_SEMIGROUP_ROUTE_UNIQUENESS.json',
    'PASS only within the declared temporal-semigroup class; equivalence to independently specified spatial paths is not proved. This is a class-relative closure, not an unconditional uniqueness theorem.')

ok,o=pass_status(V6/'results/U1_TEMPORAL_BESSEL_INFINITY_FREEZE_CERTIFICATE.json')
add('U(1) repeated-time character contraction converged',ok,
    'task1_v6/results/U1_TEMPORAL_BESSEL_INFINITY_FREEZE_CERTIFICATE.json',
    'L=2 two-macrostep pure-U(1) benchmark is frozen to infinite Bessel cutoff with a certified error bound.')

ok1,_=pass_status(V6/'results/SU2_FUSION_SEPARATOR_CERTIFICATE.json'); ok2,_=pass_status(V6/'results/SU2_CHARACTER_TAIL_CERTIFICATE.json')
add('SU(2) local Haar/intertwiner algebra and representation tail converged',ok1 and ok2,
    'task1_v6/results/SU2_FUSION_SEPARATOR_CERTIFICATE.json; SU2_CHARACTER_TAIL_CERTIFICATE.json',
    'Local/separator algebra passes to j_max=6 with analytic tail control, but no global defect spin-foam contraction is contained in the payload.')

su3=load(V6/'results/SU3_CHARACTER_CUTOFF.json')
add('SU(3) local character regulator controlled',su3.get('status')=='EXECUTED_SU3_LOCAL_CHARACTER_EXPANSION',
    'task1_v6/results/SU3_CHARACTER_CUTOFF.json',
    'Local representation cutoff max(p,q)=15 has an explicit tail bound; global SU(3) intertwiner contraction is not executed for quark channels.')

ok1,_=pass_status(V6/'results/CAR_ONE_PARTICLE_REDUCTION.json'); ok2,_=pass_status(V6/'results/CAR_COLLISION_LOWRANK_CERTIFICATE.json')
add('CAR reduction exact in vacuum one-particle charged correlator',ok1 and ok2,
    'task1_v6/results/CAR_ONE_PARTICLE_REDUCTION.json; CAR_COLLISION_LOWRANK_CERTIFICATE.json',
    'Exact for the declared number-preserving vacuum one-particle channel; does not replace the coupled electroweak defect contraction.')

ok,_=pass_status(V6/'results/LEPTON_COLOR_SUPERSELECTION_CERTIFICATE.json')
add('Leptonic color superselection exact',ok,
    'task1_v6/results/LEPTON_COLOR_SUPERSELECTION_CERTIFICATE.json',
    'SU(3) is trivial on the charged-lepton matter line. Pure-color dynamics may cancel in normalized leptonic relative observables, but not in all unnormalized/full-SM observables.')

ok,_=pass_status(NEW/'results/PRODUCT_GROUP_WORLDLINE_FACTORIZATION.json')
add('Fixed-worldline product-group factorization proved',ok,
    'task1_new/results/PRODUCT_GROUP_WORLDLINE_FACTORIZATION.json',
    'Exact only conditional on a fixed matter path. Coherent recombination over paths/routes remains required, so Z_gauge*T_matter factorization is forbidden.')

# Required global gates: deliberately strict.
add('Full coupled U(1)xSU(2)+CAR rank-six defect contraction executed',False,
    'No machine-readable payload found',
    'Missing: deterministic contraction carrying the fundamental SU(2) defect and hypercharge defect on the same worldline through the Kato-selected rank-six causal cone.')
add('Full SU(3) matter-intertwiner contraction executed for quark/full-SM channels',False,
    'No machine-readable payload found',
    'Local SU(3) character cutoff exists, but normalized SU(3) Clebsch/recoupling tensors and global quark contraction are absent.')
add('Three-dimensional Kato vacuum converged on the same network',False,
    'results/kato_vacuum_tracking.json is a strip/finite diagnostic',
    'The supplied Kato verifier does not establish the 3D R>=2 protecting gap for the coupled network.')
add('Spatial-volume regulator L converged',False,
    'No L sequence of the full coupled rank-six moments',
    'Only the periodic L=2 local/minimal cell and lower-dimensional controls are executed. No full coupled L increase is present.')
add('Dressing-radius R converged in the full compact correlator',False,
    'C10 tangent R sequence exists, but not inside the compact rank-six contraction',
    'The BCC Coulomb tangent dressing is closed separately; the same R limit is not executed inside the full compact electroweak rank-six moments.')
add('Tensor-compression chi_TN convergence certified for the full network',False,
    'No completed boundary-MPS/TRG run with accumulated discarded-norm certificate',
    'Stage A specifies the budget but does not execute the global symmetry-preserving contraction.')
add('Frozen full microscopic transfer operator with all required labels',False,
    'No global contracted transfer payload',
    'The tensor-network IR and local blocks exist, but the globally contracted transfer operator is not present.')

hard=[c for c in checks if not c['pass']]
status='PASS_TASK1_GLOBAL' if not hard else 'FAIL_TASK1_GLOBAL_NOT_FROZEN'
report={
 'status':status,
 'date':'2026-09-17',
 'external_particle_data_used':False,
 'rule':'A FAIL remains a FAIL. PASS_TASK1_GLOBAL is emitted only if every required global contraction and regulator gate is machine-evidenced.',
 'checks':checks,
 'passed':sum(c['pass'] for c in checks),'total':len(checks),
 'blocking_requirements':[c['requirement'] for c in hard],
 'scientific_conclusion':('Task 1 is fully closed.' if not hard else 'The supplied and newly executed calculations substantially reduce Task 1 but do not satisfy the requested full nonfactorized 3D contraction and regulator-convergence gate. Marking Task 1 PASS would be scientifically false.'),
}
out=NEW/'results/TASK1_GLOBAL_GATE.json'; out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
