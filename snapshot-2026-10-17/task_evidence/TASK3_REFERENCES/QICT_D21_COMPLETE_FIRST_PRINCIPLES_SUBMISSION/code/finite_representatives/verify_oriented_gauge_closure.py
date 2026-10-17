import json, csv, math, cmath
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
CERT = OUT / 'certificates'; CERT.mkdir(exist_ok=True)
phi = math.pi/63.0

def Pp(E): return 2*E*E + E
def Pm(E): return 2*E*E - E

# Integer positivity / unique vacuum certificate over a broad exact integer window.
Es = list(range(-100,101))
assert all(Pp(e) >= 0 for e in Es)
assert [e for e in Es if Pp(e)==0] == [0]
assert all(Pm(e) >= 0 for e in Es)
assert [e for e in Es if Pm(e)==0] == [0]

# Primitive uniqueness among a E^2 + b E with positive orientation b>0,
# nonnegative on Z and unique zero at 0, minimizing a+b for positive integers.
cands=[]
for a in range(1,10):
    for b in range(1,10):
        vals=[a*e*e+b*e for e in range(-30,31)]
        if min(vals) >= 0 and vals[30] == 0 and sum(v==0 for v in vals)==1:
            cands.append((a+b,a,b))
cands.sort()
assert cands[0][1:] == (2,1)

# 126-tick recurrence on U(1) electric spectrum.
rec_p=max(abs(cmath.exp(-1j*126*phi*Pp(e))-1) for e in range(-100,101))
rec_m=max(abs(cmath.exp(-1j*126*phi*Pm(e))-1) for e in range(-100,101))

# Receiver hypercharge flux +/-3 splitting.
pp3=Pp(3); pm3=Pp(-3)
split=(pp3-pm3)*phi
assert pp3==21 and pm3==15
assert abs(split - 2*math.pi/21) < 1e-14

# Casimir integer normalizations.
def su3_3c2(p,q): return p*p+q*q+p*q+3*p+3*q
def su2_4c2(n): return n*(n+2) # n=2j
cas_rows=[]
for p,q in [(0,0),(1,0),(0,1),(1,1),(2,0),(0,2)]:
    cas_rows.append(("SU3",f"({p},{q})",su3_3c2(p,q)))
for n in range(0,7):
    cas_rows.append(("SU2",f"j={n}/2",su2_4c2(n)))
assert all(isinstance(x[2],int) for x in cas_rows)

# Minimal receiver quantum costs including SU2 doublet contribution 4 C2=3.
# These are bridge-flux costs, not masses.
bridge=[]
for name,E in [("H_R",3),("H_R_conjugate",-3)]:
    val=3+Pp(E)
    bridge.append((name,E,val,val*phi))

# Finite U(1) Gauss-law commutator certificate in an electric window, excluding boundaries.
# G = E_right - E_left - Q. K(E) diagonal, hence [K,G]=0 exactly.
# Numerical matrices verify diagonal commutation.
evals=np.arange(-5,6)
E=np.diag(evals.astype(float))
I=np.eye(len(evals))
K=np.kron(np.diag([Pp(int(e)) for e in evals]),I)+np.kron(I,np.diag([Pp(int(e)) for e in evals]))
G=np.kron(E,I)-np.kron(I,E)
comm=np.linalg.norm(K@G-G@K)
assert comm < 1e-14

cert={
  "status":"PASS",
  "theory":"QICT - Quantum Information Copy Time",
  "module":"Primitive oriented U(1)_y gauge closure",
  "phi_star":phi,
  "primitive_polynomial_positive_orientation":"P_+(E)=2E^2+E",
  "primitive_polynomial_reverse_orientation":"P_-(E)=2E^2-E",
  "primitive_uniqueness_minimizer":{"a":cands[0][1],"b":cands[0][2],"complexity":cands[0][0]},
  "unique_zero_flux_vacuum_positive":True,
  "unique_zero_flux_vacuum_reverse":True,
  "recurrence_residual_positive":rec_p,
  "recurrence_residual_reverse":rec_m,
  "receiver_flux_plus3_cost":pp3,
  "receiver_flux_minus3_cost":pm3,
  "orientation_splitting_radians":split,
  "orientation_splitting_exact":"2*pi/21",
  "gauss_commutator_residual":comm,
  "claim_boundary":"This lifts the H_R vs conjugate-H_R flux degeneracy in the oriented gauge sector; it is not by itself a fermion mass prediction."
}
(CERT/'oriented_gauge_closure_certificate.json').write_text(json.dumps(cert,indent=2))
with open(OUT/'data/oriented_flux_table.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['E','P_plus','P_minus','phase_plus','phase_minus'])
    for e in range(-6,7): w.writerow([e,Pp(e),Pm(e),Pp(e)*phi,Pm(e)*phi])
with open(OUT/'data/bridge_flux_costs.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['receiver_carrier','integer_hypercharge_flux','dimensionless_integer_cost_including_SU2','quasienergy_cost'])
    w.writerows(bridge)
with open(OUT/'data/casimir_integer_table.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['group','representation','integer_normalized_casimir']); w.writerows(cas_rows)
print(json.dumps(cert,indent=2))
