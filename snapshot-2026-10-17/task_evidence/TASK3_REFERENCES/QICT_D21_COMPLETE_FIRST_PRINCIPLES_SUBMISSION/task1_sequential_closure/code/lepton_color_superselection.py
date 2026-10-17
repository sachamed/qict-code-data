#!/usr/bin/env python3
import json, hashlib
# Frozen one-family representation data from Supplement E / AL.
reps={
 'Q_L':{'SU3':'3','SU2':'2','Y':'1/6'},
 'u_R':{'SU3':'3','SU2':'1','Y':'2/3'},
 'd_R':{'SU3':'3','SU2':'1','Y':'-1/3'},
 'L_L':{'SU3':'1','SU2':'2','Y':'-1/2'},
 'e_R':{'SU3':'1','SU2':'1','Y':'-1'},
 'nu_R':{'SU3':'1','SU2':'1','Y':'0'},
}
leptons=['L_L','e_R','nu_R']; quarks=['Q_L','u_R','d_R']
assert all(reps[x]['SU3']=='1' for x in leptons)
assert all(reps[x]['SU3']=='3' for x in quarks)
rep={
 'status':'PASS_LEPTON_COLOR_SUPERSELECTION',
 'frozen_representation_table':reps,
 'leptonic_color_representation_trivial':True,
 'proof':[ 
   'For f in {L_L,e_R,nu_R}, R_{3,f}(C)=1 for every C in SU(3).',
   'Therefore the gauge-covariant one-particle stream restricted to the leptonic subspace is independent of SU(3) link variables.',
   'The declared CAR completion preserves particle number, so a CAR-vacuum one-particle leptonic source cannot generate a quark/color matter line.',
   'Thus SU(3) has no open matter defect in purely leptonic C_n^(6); it remains a pure-gauge tensor factor.'
 ],
 'observable_consequence':{
   'normalized_leptonic_mass_ratios':'The same pure-color vacuum factor is common and cancels.',
   'unnormalized_Cn_or_full_Tplus':'The pure SU(3) vacuum transfer scalar must still be retained unless the observable definition explicitly normalizes it away.',
   'quark_observables':'No factorization; full SU(3) matter intertwiners are required.'
 },
 'scope':'Exact for the declared purely leptonic one-particle channel only; not a claim that the full Standard-Model microscopic operator factorizes.'
}
out='/mnt/data/qict_task1_v5/LEPTON_COLOR_SUPERSELECTION_CERTIFICATE.json'
with open(out,'w') as f:json.dump(rep,f,indent=2);f.write('\n')
print(json.dumps(rep,indent=2))
