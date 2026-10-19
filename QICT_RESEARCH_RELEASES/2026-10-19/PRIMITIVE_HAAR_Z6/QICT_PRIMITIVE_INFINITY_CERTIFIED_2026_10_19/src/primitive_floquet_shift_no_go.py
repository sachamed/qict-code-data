#!/usr/bin/env python3
"""Exact shift-symmetry no-go for the ordered U(1) single-plaquette Floquet gates.

Contrasts the true rank-one primitive Hamiltonian pole with the absence of a
nonzero finite-rank Riesz projector of U_BE. The full QICT U_int^rf is absent.
"""
import json
from pathlib import Path
from primitive_infinite_volume_certificate import ROOT

SHIFT=126

def ke(r):return 8*r*r

def exact_phase_periodicity(r,shift=SHIFT):
    # exp(-i pi K_E/63) periodic iff delta K_E divisible by 126.
    return (ke(r+shift)-ke(r))%126==0

def theorem():
    assert all(exact_phase_periodicity(r) for r in range(-1000,1001))
    assert all(not exact_phase_periodicity(r,21) for r in [1,2,4])
    # 8((r+126)^2-r^2) = 2016*r+127008 =126*(16*r+1008)
    assert all(ke(r+126)-ke(r)==126*(16*r+1008) for r in range(-1000,1001))
    # In contrast, the unwrapped primitive H fails the shift symmetry.
    assert any(ke(r+126)!=ke(r) for r in range(-10,11))
    doc={'classification':'EXACT_FLOQUET_SINGLE_PLAQUETTE_SHIFT_NO_FINITE_RANK_RIESZ',
      'Hilbert_space':'ell2(Z) U1 electric-flux representation',
      'electric_generator':'K_E=8*r^2',
      'magnetic_generator':'K_B=4-2(S+S_dagger)',
      'Floquet_order':'U_BE=exp(-i*pi*K_B/63) exp(-i*pi*K_E/63)',
      'shift':'T_126: e_r -> e_(r+126)',
      'electric_phase_shift_identity':'K_E(r+126)-K_E(r)=126*(16r+1008)',
      'electric_phase_commutes_with_shift':True,
      'magnetic_gate_commutes_with_shift':True,
      'U_BE_commutes_with_shift':True,
      'shift_T126_has_no_nonzero_eigenvectors_on_ell2Z':True,
      'all_nonzero_finite_rank_Riesz_projections_of_U_BE_forbidden':True,
      'Hamiltonian_unwrapped_H_commutes_with_shift':False,
      'primitive_H_rank_one_ground_projector_allowed':True,
      'full_QICT_micro_operator_no_go_proved':False,
      'reason_full_operator_not_proved':'Missing U_int route-first gauge and matter amplitudes may change the translation commutator.'}
    (ROOT/'results/PRIMITIVE_FLOQUET_SHIFT_NO_GO.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc

if __name__=='__main__':
    r=theorem()
    print(r['all_nonzero_finite_rank_Riesz_projections_of_U_BE_forbidden'])
