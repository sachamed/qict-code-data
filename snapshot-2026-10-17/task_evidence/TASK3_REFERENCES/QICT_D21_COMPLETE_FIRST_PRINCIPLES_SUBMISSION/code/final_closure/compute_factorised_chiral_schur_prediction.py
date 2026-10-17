#!/usr/bin/env python3
from pathlib import Path
import json, math, hashlib
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gamma

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'factorised_schur'
OUT.mkdir(parents=True, exist_ok=True)
PRED=ROOT/'predictions'/'FACTORISED_SCHUR_PRECOMPARISON_LOCK.json'
LOCK=ROOT/'predictions'/'FACTORISED_SCHUR_PRECOMPARISON_LOCK.sha256'

# QICT-internal inputs only.
phi=math.pi/63.0
R_bcc=float(gamma(0.25)**4/(32.0*math.pi**3))

def su2_rotor(nmax:int):
    n=np.arange(nmax+1,dtype=float)
    # Primitive SU(2) rotor used in the local Peter-Weyl diagnostic:
    # K_E |n> = n(n+2)|n>, K_B = 32 I - 16 M_{chi_1/2}.
    diag=n*(n+2.0)+32.0
    off=np.full(nmax,-16.0)
    vals,vecs=eigh_tridiagonal(diag,off,select='i',select_range=(0,0))
    v=vecs[:,0]
    half_char=float(np.sum(v[:-1]*v[1:])) # <chi_{1/2}/2>
    tail=float(v[-1]**2 + (v[-2]**2 if nmax>=1 else 0.0))
    return float(vals[0]),half_char,tail

rot=[]
for nmax in (20,40,60,80):
    E,w,tail=su2_rotor(nmax)
    rot.append({'nmax':nmax,'ground_energy':E,'half_fundamental_character':w,'tail_probability_last_two':tail})

w=rot[-1]['half_fundamental_character']
alpha=R_bcc*math.sqrt(w)*math.cos(phi)

def Pplus(E:int)->int:
    return 2*E*E+E

# Ordered family channels are predictions by increasing suppression: largest route defect is lightest.
defects=[5,3,1]
costs=[Pplus(x) for x in defects] # 55,21,3
relative=[math.exp(-alpha*(p-costs[0])) for p in costs] # electron normalized to 1
mu_over_e=relative[1]/relative[0]
tau_over_mu=relative[2]/relative[1]
tau_over_e=relative[2]/relative[0]
koide=sum(relative)/(sum(math.sqrt(x) for x in relative)**2)

obj={
  'status':'PRECOMPARISON_PARAMETER_FREE_FACTORISED_SCHUR_PREDICTION',
  'external_particle_mass_data_read':False,
  'logical_scope':(
    'Leading one-cell/cluster-factorized approximation to the rank-six chiral Schur kernel. '
    'It is not the converged non-factorized 3D electroweak contraction.'
  ),
  'inputs':{
    'phi_star':'pi/63',
    'bcc_unit_resistance_exact':'Gamma(1/4)^4/(32*pi^3)',
    'bcc_unit_resistance_numeric':R_bcc,
    'su2_rotor_coefficients':{'c2':4,'A2':32},
    'su2_rotor_convergence':rot,
    'route_defects_ordered_light_to_heavy':defects,
    'oriented_electric_costs':costs,
    'electric_polynomial':'P_+(E)=2E^2+E'
  },
  'factorization_rule':{
    'alpha_definition':'R_BCC * sqrt(<chi_1/2/2>_SU2,prim) * cos(phi_*)',
    'alpha_numeric':alpha,
    'mass_shape':'m(Delta) proportional to exp[-alpha P_+(Delta)]',
    'interpretation':(
      'R_BCC is the 3D BCC Gauss return factor, sqrt(W_1/2) is the fundamental-line amplitude '
      'in the primitive SU(2) rotor, and cos(phi_*) is the real primitive collision matrix element. '
      'Their product is the disconnected/factorized leading Schur coefficient.'
    )
  },
  'predictions':{
    'm_mu_over_m_e':mu_over_e,
    'm_tau_over_m_mu':tau_over_mu,
    'm_tau_over_m_e':tau_over_e,
    'koide_Q_from_predicted_ratios':koide,
    'log_spacing_ratio_exact_from_costs':(costs[0]-costs[1])/(costs[1]-costs[2])
  },
  'claim_boundary':(
    'The ratios above are parameter-free outputs of the stated factorized Schur approximation. '
    'They are not promoted to exact QICT masses until the connected 3D electroweak rank-six contraction '
    'is computed and shown to converge to the same limit.'
  )
}
PRED.write_text(json.dumps(obj,indent=2)+'\n')
h=hashlib.sha256(PRED.read_bytes()).hexdigest()
LOCK.write_text(h+'  '+PRED.name+'\n')
(OUT/'FACTORISED_SCHUR_PREDICTION.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj,indent=2))
print('SHA256',h)
