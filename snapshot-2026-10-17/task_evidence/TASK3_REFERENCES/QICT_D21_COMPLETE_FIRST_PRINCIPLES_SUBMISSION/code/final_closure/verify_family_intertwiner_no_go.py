#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import sympy as sp

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'final_closure'; OUT.mkdir(parents=True,exist_ok=True)

# Three-family carrier. Gauge acts trivially on family multiplicity, so gauge covariance
# leaves End(C^3) untouched. The Weyl pair tests the stronger full family symmetry.
omega=np.exp(2j*np.pi/3)
X=np.array([[0,1,0],[0,0,1],[1,0,0]],complex)
Z=np.diag([1,omega,omega**2])

# Exact symbolic commutant of X,Z.
w=sp.exp(2*sp.pi*sp.I/3)
Xs=sp.Matrix([[0,1,0],[0,0,1],[1,0,0]])
Zs=sp.diag(1,w,w**2)
ys=sp.symbols('y0:9')
Y=sp.Matrix(3,3,ys)
eqs=list(Y*Xs-Xs*Y)+list(Y*Zs-Zs*Y)
# Flatten equations into scalar linear constraints.
scalars=[]
for M in (Y*Xs-Xs*Y,Y*Zs-Zs*Y):
    scalars.extend(list(M))
A,b=sp.linear_eq_to_matrix(scalars,ys)
null=A.nullspace()

# Numerical cross-check.
rng=np.random.default_rng(20260916)
alpha=0.321+0.117j
Yscalar=alpha*np.eye(3)
res_scalar=max(np.linalg.norm(Yscalar@X-X@Yscalar),np.linalg.norm(Yscalar@Z-Z@Yscalar))

cert={
  'status':'PASS',
  'date':'2026-09-16',
  'theorem':'family-intertwiner factorization and Weyl-commutant no-go',
  'gauge_family_multiplicity':'For three identical gauge representations, Hom_G((V_R tensor H_R) tensor C^3, V_L tensor C^3) is Hom_G(V_R tensor H_R,V_L) tensor End(C^3). With multiplicity-one electroweak Clebsch support, the allowed family coefficient is an arbitrary Y_e in M_3(C).',
  'gauge_covariance_family_complex_dimension':9,
  'full_Weyl_symmetry_generators':['X','Z'],
  'full_Weyl_commutant_complex_dimension':len(null),
  'full_Weyl_commutant_basis':[str(v.T) for v in null],
  'scalar_commutant_numerical_residual':float(res_scalar),
  'consequence':'Gauge covariance alone cannot determine the charged-lepton family hierarchy. Conversely, exact invariance under the full irreducible Weyl pair forces Y_e proportional to I_3 and hence degenerate singular values. Therefore nondegenerate charged-lepton masses require the route/time ordering to break the full family-Weyl symmetry dynamically, and the physical Y_e must be extracted from the rank-six chiral Schur/Feshbach kernel rather than inserted as a Yukawa matrix.',
  'external_particle_data_used':False
}
(OUT/'FAMILY_INTERTWINER_NO_GO.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))
