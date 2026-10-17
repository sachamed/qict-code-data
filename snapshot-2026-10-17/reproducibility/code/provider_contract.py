#!/usr/bin/env python3
ACTION_SHA256='c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7'
SOURCE_ORDER=('L1','L2','L3','R1','R2','R3')

def validate_provider(mod, expected_L:int):
    for name in ('metadata','dimension'):
        if not hasattr(mod,name): raise ValueError(f'provider missing {name}()')
    if not (hasattr(mod,'source_matrix') or hasattr(mod,'source_matrix_into')):
        raise ValueError('provider needs source_matrix() or source_matrix_into(out)')
    if not (hasattr(mod,'matmat') or hasattr(mod,'matmat_into')):
        raise ValueError('provider needs matmat(X) or matmat_into(X,out)')
    meta=dict(mod.metadata())
    required={
      'physical_payload':True,'action_sha256':ACTION_SHA256,'L':int(expected_L),
      'PiCAR':1,'source_order':list(SOURCE_ORDER)
    }
    for k,v in required.items():
        if meta.get(k)!=v: raise ValueError(f'provider metadata mismatch: {k}={meta.get(k)!r}, expected {v!r}')
    if not meta.get('charged_U_cone_complete',False): raise ValueError('charged_U_cone_complete is not certified true')
    if not meta.get('lambda1_riesz_kato_selected',False): raise ValueError('lambda1_riesz_kato_selected is not certified true')
    if int(meta.get('KATO_BUFFER_count',0))<4: raise ValueError('at least four KATO buffers required')
    if not meta.get('buffer_tail_certified',False): raise ValueError('buffer tail is not certified')
    if not meta.get('C10_shell_CP_maps_present',False): raise ValueError('C10 shell CP maps missing')
    if not meta.get('boundary_to_path_present',False): raise ValueError('boundary-to-path tensor missing')
    if not meta.get('rank6_source_carrier_certified',False): raise ValueError('rank-six source carrier not certified')
    err=float(meta.get('operator_or_state_error_bound',float('inf')))
    if not (0<=err<1e-6): raise ValueError(f'uncertified/too-large operator_or_state_error_bound={err}')
    D=int(mod.dimension())
    if D<=0: raise ValueError('invalid dimension')
    return meta,D
