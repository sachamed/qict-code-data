#!/usr/bin/env python3
"""Representation-theoretic Z6 descent criteria for product-cover characters.

This is a mathematical classification of allowed centre charges, not a change
of the frozen QICT magnetic action or an interacting gauge calculation.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def z6_character_residue(abelian_charge:int, su2_twice_spin:int, su3_triality:int)->int:
    vals=(abelian_charge,su2_twice_spin,su3_triality)
    if not all(isinstance(v,int) for v in vals):
        raise ValueError('Integer center charges required')
    if su2_twice_spin<0 or su3_triality not in (0,1,2):
        raise ValueError('Invalid SU2 spin or SU3 triality')
    return (abelian_charge+3*su2_twice_spin+2*su3_triality)%6


def is_quotient_representation(y:int,j2:int,t3:int)->bool:
    return z6_character_residue(y,j2,t3)==0


def mixed_character(theta,u,v):
    """A normalized character of the product representation (Y=1,2,3)."""
    u=np.asarray(u,complex);v=np.asarray(v,complex)
    if u.shape!=(2,2) or v.shape!=(3,3):raise ValueError('Expected defining SU2 and SU3 matrices')
    return np.exp(1j*theta)*np.trace(u)*np.trace(v)/6


def central_step(theta,u,v):
    return theta+np.pi/3, -u, np.exp(2j*np.pi/3)*v


def build_report():
    cases=[('U1_fundamental',1,0,0),('SU2_fundamental',0,1,0),
           ('SU3_fundamental',0,0,1),('mixed_Y1_doublet_triplet',1,1,1),
           ('U1_sixth_power',6,0,0),('SU2_adjoint',0,2,0),
           ('SU3_adjoint',0,0,0),('mixed_Yminus5_doublet_triplet',-5,1,1)]
    entries=[{'name':label,'Y_integer':y,'twice_spin_SU2':j,'triality_SU3':t,
              'central_residue_mod6':z6_character_residue(y,j,t),
              'descends_to_quotient':is_quotient_representation(y,j,t)}
              for label,y,j,t in cases]
    u=np.diag([np.exp(.2j),np.exp(-.2j)])
    v=np.diag([np.exp(.1j),np.exp(.3j),np.exp(-.4j)])
    theta=.27
    original=mixed_character(theta,u,v)
    errors=[]
    for _ in range(6):
        theta,u,v=central_step(theta,u,v)
        errors.append(float(abs(mixed_character(theta,u,v)-original)))
    assert all(e<1e-13 for e in errors)
    assert is_quotient_representation(1,1,1)
    assert not is_quotient_representation(1,0,0)
    output={'classification':'EXACT_Z6_CHARACTER_DESCENT_THEOREM_WITH_NUMERICAL_WITNESS',
       'kernel_generator':'(exp(i pi/3), -I2, exp(2 pi i/3) I3)',
       'representation_center_phase':'exp(i pi*(Y + 3*(2j) + 2*t)/3)',
       'descent_iff':'(Y+3*(2j)+2*t) mod 6 = 0',
       'cases':entries,
       'mixed_character_definition':'exp(i theta) Tr_2(u) Tr_3(v) / 6',
       'six_central_lifts_max_character_error_float64':max(errors),
       'mixed_character_is_action_original_K_B':False,
       'frozen_action_modified':False,
       'full_quotient_gauge_dynamics_constructed':False}
    (ROOT/'results/CENTER_CHARACTER_SELECTION_RULE.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    return output

if __name__=='__main__':
    report=build_report()
    print(report['descent_iff'],report['six_central_lifts_max_character_error_float64'])
