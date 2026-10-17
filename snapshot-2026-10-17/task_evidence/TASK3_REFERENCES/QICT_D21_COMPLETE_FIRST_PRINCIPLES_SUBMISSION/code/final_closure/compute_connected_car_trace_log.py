#!/usr/bin/env python3
from pathlib import Path
import json, math, hashlib
ROOT=Path(__file__).resolve().parents[2]
FACTOR=ROOT/'predictions'/'FACTORISED_SCHUR_PRECOMPARISON_LOCK.json'
OUT=ROOT/'predictions'/'CONNECTED_CAR_PRECOMPARISON_LOCK.json'
SHA=ROOT/'predictions'/'CONNECTED_CAR_PRECOMPARISON_LOCK.sha256'
obj0=json.loads(FACTOR.read_text())
alpha0=float(obj0['factorization_rule']['alpha_numeric'])
NF=96
# The connected family insertion acts on the 3-dimensional family block Y_e.
# X has eigenvalues {1,omega,omega^2}, hence det(I-alpha X)=1-alpha^3.
delta=-(1.0/NF)*math.log1p(-(alpha0**3))
alpha=alpha0+delta
P={1:3,3:21,5:55}
r_mu_e=math.exp((P[5]-P[3])*alpha)
r_tau_mu=math.exp((P[3]-P[1])*alpha)
r_tau_e=math.exp((P[5]-P[1])*alpha)
rel=[1.0,r_mu_e,r_tau_e]
Q=sum(rel)/(sum(math.sqrt(x) for x in rel)**2)
out={
 'status':'CONNECTED_CAR_FORMULA_LOCK_BEFORE_FINAL_AUDIT',
 'external_particle_mass_data_read':False,
 'input_factorised_lock_sha256':hashlib.sha256(FACTOR.read_bytes()).hexdigest(),
 'finite_internal_dimension':NF,
 'family_block_dimension':3,
 'connected_loop_theorem':{
   'normalized_trace':'tau_96(A)=Tr(A)/96, the unique unitary-invariant normalized trace on M_96(C)',
   'family_cycle':'X^3=I_3; tr X=tr X^2=0',
   'embedded_support':'the connected insertion acts in the charged-lepton family block Y_e after the chiral Clebsch support is factored',
   'trace_log':'delta_alpha = -tau_96 log(I-alpha_fac X_e) = -(1/96) log(1-alpha_fac^3)',
   'linked_cluster_interpretation':'the logarithm retains connected CAR family windings; only multiples of three survive the C3 trace'
 },
 'alpha_factorised':alpha0,
 'delta_alpha_connected':delta,
 'fractional_connected_correction':delta/alpha0,
 'alpha_connected':alpha,
 'costs':{'electron':55,'muon':21,'tau':3},
 'predictions':{
   'm_mu_over_m_e':r_mu_e,
   'm_tau_over_m_mu':r_tau_mu,
   'm_tau_over_m_e':r_tau_e,
   'koide_Q_from_relative_spectrum':Q
 },
 'claim_boundary':(
   'The C3 trace-log identity is exact for the proposed reduced connected functional. Its additive use in the full Schur exponent is the reduced linked-cluster closure, not a theorem from the complete lattice. '
   'The BCC Gauss return and local SU(2) Peter-Weyl line amplitude are already contained in alpha_fac. This certificate does not claim a brute-force R>=2 non-factorizing Peter-Weyl tensor contraction of the entire electroweak lattice. The checksum prevents retuning after the formula is fixed; it is not a prospective blind-test claim because lepton data were known during the historical model-development process.'
 )
}
OUT.write_text(json.dumps(out,indent=2)+'\n')
h=hashlib.sha256(OUT.read_bytes()).hexdigest()
SHA.write_text(h+'  '+OUT.name+'\n')
(ROOT/'results'/'connected_car'/'CONNECTED_CAR_PREDICTION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
print('SHA256',h)
