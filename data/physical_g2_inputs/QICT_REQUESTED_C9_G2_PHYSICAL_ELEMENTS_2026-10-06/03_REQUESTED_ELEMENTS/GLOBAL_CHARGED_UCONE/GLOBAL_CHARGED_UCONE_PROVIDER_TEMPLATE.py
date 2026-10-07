#!/usr/bin/env python3
"""Template only. It MUST remain non-physical until every metadata claim is backed by certificates."""
import numpy as np

def metadata():
    return {
      'physical_payload': False,
      'action_sha256':'c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7',
      'L':24,'PiCAR':1,'source_order':['L1','L2','L3','R1','R2','R3'],
      'charged_U_cone_complete':False,'lambda1_riesz_kato_selected':False,
      'KATO_BUFFER_count':0,'buffer_tail_certified':False,
      'C10_shell_CP_maps_present':False,'boundary_to_path_present':False,
      'rank6_source_carrier_certified':False,
      'operator_or_state_error_bound':float('inf'),
      'implementation_note':'Implement source_matrix_into(out) and matmat_into(X,out) with disk-backed/sharded kernels. Do not set physical_payload true until all certificates exist.'
    }

def dimension(): raise NotImplementedError('physical dimension')
def source_matrix_into(out): raise NotImplementedError('fill D x 6 source carrier')
def matmat_into(X,out): raise NotImplementedError('apply complete relative charged U_cone to six columns')
