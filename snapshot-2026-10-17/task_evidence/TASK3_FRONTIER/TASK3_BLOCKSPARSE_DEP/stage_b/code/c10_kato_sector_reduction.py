#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name): return json.load(open(ROOT/'upstream'/name))
def result(name): return json.load(open(ROOT/'results'/name))
def main():
    audit=load('TASK2_C10_QUANTUM_LIFT_AUDIT.json')
    kato=load('TASK2_KATO_MAPPED_CHARGED_GAUSS_STRIP_R1_R3.json')
    u1=result('U1_GAUSS_REFERENCE_COORDINATE.json')
    su2=result('SU2_REFERENCE_PATH_LOOP_RELATION.json')
    su3=result('SU3_RELATIVE_CANCELLATION.json')
    checks={
      'c10_thin_string_superposition_rejected':audit.get('status')=='PASS_C10_STRING_LIFT_NOT_EQUAL_TANGENT_DRESSING',
      'kato_gauss_sector_selection_executed_u1':kato.get('status')=='PASS_KATO_MAPPED_CHARGED_GAUSS_STRIP_R1_R3',
      'u1_reference_string_coordinate_invariance':u1.get('status','').startswith('PASS_'),
      'su2_reference_string_closed_loop_relation':su2.get('status','').startswith('PASS_'),
      'su3_relative_cancellation':su3.get('status','').startswith('PASS_')
    }
    assert all(checks.values())
    out={
      'status':'PASS_C10_KATO_GAUSS_SECTOR_REDUCTION',
      'checks':checks,
      'theorem':'C10 fixes an infrared Gauss boundary condition, not a probability amplitude over orthogonal thin-string states. A reference string only chooses affine coordinates inside the charged sector: changing it multiplies by/translated through the closed-loop algebra. The physical compact state is selected by Kato/Floquet continuation in that complete sector. For charged leptons SU(3) cancels identically in the relative correlator, leaving the U(1)xSU(2) charged boundary problem.',
      'executed_evidence':{
        'C10_infinite_unit_resistance':audit['C10_exact_infinite_unit_resistance'],
        'C10_qabs3_tangent_energy':audit['C10_exact_infinite_q2_tangent_energy_sum_E2'],
        'u1_reference_q3_residual':u1['q3_identity_max_abs_residual'],
        'su2_reference_covariance_residual':su2['max_closed_loop_covariance_residual'],
        'su3_relative_cancellation_residual':su3['max_residual_abs'],
        'u1_kato_strip_checks':kato.get('checks',{})
      },
      'remaining_boundary_object':'A compact C10/Kato boundary-to-core tensor (or an exactly equivalent charged-sector Kato state map) that couples the external BCC dressing regulator to the L=2 24-link PEPO. This object is not present in the supplied archives.',
      'why_it_is_required':'The 24-link periodic PEPO describes the microscopic core. C10 is defined on a Dirichlet BCC ball with compensating flux at its outer boundary. The physical charged source cannot be replaced by the path-probability Wilson-string state (Jensen audit) and cannot be represented by a neutral periodic core alone. The exterior must therefore be integrated into a boundary tensor or retained explicitly during Kato continuation.',
      'forbidden_shortcuts':['probabilistic C10 thin-string superposition','zero-length neutralizing string','static tangent electric phase used as the full compact boundary state','post-summed U1 times SU2 amplitudes']
    }
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
