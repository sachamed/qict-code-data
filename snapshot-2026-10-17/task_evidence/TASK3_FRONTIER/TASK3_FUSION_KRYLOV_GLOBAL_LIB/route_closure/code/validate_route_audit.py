from pathlib import Path
import json,sys
R=Path(__file__).resolve().parents[2]
A=json.load(open(R/'certificates/PEPO3D_REPRESENTATION_TRUNCATION.json'))
H=json.load(open(R/'route_closure/certificates/ROUTE_SOURCE_H0.json'))
N=json.load(open(R/'route_closure/certificates/SU2_CURRENT_ORDER_COUNTEREXAMPLE.json'))
S=json.load(open(R/'route_closure/certificates/STATIC_CONTROL_CLASSIFICATION.json'))
G=json.load(open(R/'certificates/TASK3_PHYSICAL_PROMOTION_GATE.json'))
checks={
 'pepo_representation_truncation':A['status'].startswith('PASS'),
 'h0_fixed':H['status'].startswith('PASS') and H['h0']==[0,1],
 'nonabelian_ordering_counterexample':N['status'].startswith('PASS') and N['trace_difference_abs']>0.1,
 'static_control_not_misidentified':S['status'].startswith('PASS') and not S['promotion_allowed'],
 'physical_gate_remains_truthful':not G['checks']['repeated_electric_kernel_in_common_nonabelian_route_contracted'] and not G['checks']['physical_poles_and_residues_promoted']
}
out={'status':'PASS_ROUTE_AUDIT_INTEGRITY' if all(checks.values()) else 'FAIL_ROUTE_AUDIT_INTEGRITY','checks':checks,'physical_promotion_status':G['status']}
print(json.dumps(out,indent=2));sys.exit(0 if all(checks.values()) else 1)
