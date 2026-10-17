#!/usr/bin/env python3
"""Exact algebraic audit of underdetermination in the archived QICT reconstruction.

This program proves mathematical limitations of incomplete physical inputs. The
finite unitaries are abstract counterexamples, NOT QICT microscopic dynamics.
No experimental information enters the calculations.
"""
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'COHERENT_IDENTIFIABILITY_EXACT_CERTIFICATE.json'
PHYSICAL = ROOT/'results'/'physical_g2_execution'/'PHYSICAL_G2_CONTRACT_AUDIT.json'

p=sp.Rational(1,7)
a=sp.sqrt(1-p)
b=sp.sqrt(p)
R=sp.Matrix([[a,b],[-b,a]])
S=sp.Matrix([[a,b],[b,-a]])
I2=sp.eye(2)
assert sp.simplify(R.T*R-I2)==sp.zeros(2)
assert sp.simplify(S.T*S-I2)==sp.zeros(2)
assert all(sp.simplify(R[i,j]**2-S[i,j]**2)==0 for i in range(2) for j in range(2))

# Rank-six isometric receiver injection in the same seven-dimensional space:
# the active channel is e1, with five spectator channels e3,...,e7.
UR=sp.diag(R,sp.eye(5))
US=sp.diag(S,sp.eye(5))
V=sp.zeros(7,6)
for j,k in enumerate((0,2,3,4,5,6)):
    V[k,j]=1
assert V.T*V==sp.eye(6)
C=lambda U,n: sp.simplify(V.T*(U**n)*V)
CR1,CS1=C(UR,1),C(US,1)
CR2,CS2=C(UR,2),C(US,2)
assert CR1==CS1
assert CR2!=CS2
assert sp.simplify((CR2-CS2)[0,0]+sp.Rational(2,7))==0
assert sp.simplify((R**2)[0,0]-sp.Rational(5,7))==0
assert sp.simplify((S**2)[0,0]-1)==0

# Exact dimensionless logarithmic Jacobian: (ln tau, ln gamma, ln |epsilon0|)
# mapped to (ln G, ln |Lambda|). Assumes positive tau,gamma and epsilon0!=0.
J=sp.Matrix([[2,-1,0],[-2,-1,1]])
null=sp.Matrix([1,2,4])
assert J.rank()==2
assert J*null==sp.zeros(2,1)
assert J[:,:2].det()==-4
# A rank-six spectral mass epsilon_f adds one independent effective coordinate.
# If epsilon_f is independent, m_f proportional epsilon_f / tau.
Jfull=sp.Matrix([[2,-1,0,0],[-2,-1,1,0],[-1,0,0,1]])
full_null=sp.Matrix([1,2,4,1])
assert Jfull.rank()==3
assert Jfull*full_null==sp.zeros(3,1)

if not PHYSICAL.exists():
    raise RuntimeError('Physical G2 audit missing: execute contract audit first')
audit=json.loads(PHYSICAL.read_text())
checks=audit['physical_contract_checks']
missing=sorted(k for k,v in checks.items() if v is not True)
assert len(checks)==9, 'Expected nine declared canonical physical-input classes'
assert set(checks)==set(audit['continuum_sequence_inputs'])
assert missing==sorted(audit['missing_physical_objects'])
if missing:
    assert audit['physical_G2_promoted'] is False

out={
 'classification':'EXACT_MATHEMATICAL_OBSTRUCTION_AND_STRICT_PHYSICAL_INPUT_AUDIT',
 'scope':'Abstract unitary counterexample + effective identifiability; NOT physical QICT masses or gravity',
 'p':'1/7',
 'two_channel_unitaries':{'R':'[[sqrt(6/7),sqrt(1/7)],[-sqrt(1/7),sqrt(6/7)]]',
                          'S':'[[sqrt(6/7),sqrt(1/7)],[sqrt(1/7),-sqrt(6/7)]]'},
 'same_entrywise_transition_probabilities': True,
 'six_source_injection': 'e1, e3, e4, e5, e6, e7 into C^7',
 'same_rank6_first_moment': True,
 'first_moment_00':'sqrt(6/7)',
 'second_moment_R_00':'5/7',
 'second_moment_S_00':'1',
 'exact_second_moment_difference_00':'-2/7',
 'unitarity_and_receiver_isometry':'SYMpy EXACT PASS',
 'gravity_log_jacobian': [[2,-1,0],[-2,-1,1]],
 'gravity_log_jacobian_rank':2,
 'gravity_null_direction': [1,2,4],
 'gravity_finite_equivalence':'tau -> a tau, gamma_R -> a^2 gamma_R, epsilon0 -> a^4 epsilon0; a>0',
 'gravity_plus_one_mass_log_jacobian': [[2,-1,0,0],[-2,-1,1,0],[-1,0,0,1]],
 'gravity_plus_mass_rank':3,
 'gravity_plus_mass_null_direction': [1,2,4,1],
 'missing_physical_G2_input_classes':missing,
 'complete_physical_G2_input_classes':len(checks)-len(missing),
 'can_issue_physical_mass_predictions': len(missing)==0,
 'can_issue_gravity_predictions_from_tau_alone': False,
 'interpretation':('Moment and gravity relations cannot provide a unique physical spectrum or independent coupling count without the coherent microscopic data and independent observable definitions. This audit does not establish any positive physical model parameter count.'),
 'exact_assertions_passed': 18
}
OUT.write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
print('PASS: exact rank-six unitary obstruction, effective gravity rank, strict G2 data classification.')
print('Physical input classes complete:',len(checks)-len(missing),'/',len(checks))
