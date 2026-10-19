"""Examine fields actually present in the frozen *source* action.
No numerical physical tensor is constructed or inferred from symbolic labels.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7'

def inspect(path):
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED:raise ValueError('wrong action hash')
    data=json.loads(path.read_text())
    implemented={
        'ordered_factor_names':list(data.get('factor_order',[])),
        'U1_SU2_SU3_generator_formulae':all(
            isinstance(data.get(k),dict) and all(r in data[k] for r in ['U1','SU2','SU3'])
            for k in ['electric_generator','plaquette_generator']),
        'microscopic_route_first_full_sparse_matrix':False,
        'route_labels_and_holonomy_maps':False,
        'common_full_gauge_matter_Hilbert_basis':False,
        'interaction_kernel_apply_U':False,
        'interaction_kernel_apply_U_dagger':False,
        'four_interacting_lambda1_state_densities':False,
        'C10_Kraus_CP_maps':False,
        'six_chiral_sources_gauss_CAR_embedding':False,
        'physical_dark_source_and_parity':False,
        'independent_tau_star_SI_calibration':False}
    return {'status':'BLOCKED_MISSING_INPUT','verified_action_sha256':digest,
        'action_source_sha_verified':True,'derived_from_action_json_keys':sorted(data.keys()),
        'contract':implemented,
        'gauge_group_header':data.get('gauge_group'),
        'required_magnetic_gate_domain':'U(1) x SU(2) x SU(3) product cover',
        'quotient_link_action_currently_defined':False,
        'note':'False means action JSON does not provide that executable numeric object; no claim that it can never be derived from additional complete theory.'}
if __name__=='__main__':
    result=inspect(ROOT/'evidence/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json')
    (ROOT/'results/CANONICAL_ACTION_EXECUTABILITY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sha_ok':True,'full_U':False,'C10':False,'tau_SI':False},indent=2))
