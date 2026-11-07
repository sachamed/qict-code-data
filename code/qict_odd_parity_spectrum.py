"""Finite Schwinger-spin receiver parity spectral-gap diagnostic.

Conditional Hamiltonian H/u = 8 K(K-1)(K-2)/(N)_3 +
eta (Jz/N)^2, K=N/2-Jx. Flavor exchange I reverses Jz
and commutes with both terms, yielding even/odd blocks.
The absolute energy scale u and lattice spacing are NOT fixed here.
"""
import json,math
from pathlib import Path
import numpy as np

def blocks(N,eta):
 j=N/2
 m=np.arange(-j,j+1)
 Jx=np.zeros((N+1,N+1),float)
 for k in range(N):
  v=.5*math.sqrt((j-m[k])*(j+m[k]+1))
  Jx[k,k+1]=v;Jx[k+1,k]=v
 I=np.eye(N+1)
 K=j*I-Jx
 H=(8/(N*(N-1)*(N-2)))*K@(K-I)@(K-2*I)+(eta/N**2)*np.diag(m*m)
 Vp=[];Vm=[]
 for k in range((N+1)//2):
  v=np.zeros(N+1);v[k]=v[N-k]=2**-.5;Vp.append(v)
  w=np.zeros(N+1);w[k]=2**-.5;w[N-k]=-2**-.5;Vm.append(w)
 if N%2==0:
  v=np.zeros(N+1);v[N//2]=1;Vp.append(v)
 Vp=np.array(Vp).T;Vm=np.array(Vm).T
 P=np.eye(N+1)[::-1]
 assert np.max(np.abs(P@H-H@P))<1e-10
 assert np.max(np.abs(Vp.T@H@Vm))<1e-10
 ep=np.linalg.eigvalsh(Vp.T@H@Vp)
 em=np.linalg.eigvalsh(Vm.T@H@Vm)
 return {"N":N,"eta":eta,"even_ground_over_u":float(ep[0]),
         "odd_ground_over_u":float(em[0]),
         "odd_minus_even_gap_over_u":float(em[0]-ep[0]),
         "even_excited_gap_over_u":float(ep[1]-ep[0]) if len(ep)>1 else None,
         "parity_conservation_residual":float(np.max(np.abs(P@H-H@P)))}

def main():
 rows=[]
 for N in (10,20,40,100,250):
  for eta in (0,.01,.1,1,10):
   row=blocks(N,eta)
   if eta>0 and row["odd_minus_even_gap_over_u"]<=0:
    raise RuntimeError("Observed no positive odd gap at "+str(row))
   rows.append(row)
 result={
   "status":"passed_finite_matrix_tests",
   "operator":"H/u = 8 K(K-1)(K-2)/(N)_3 + eta (Jz/N)^2",
   "conditional_primitive":"three distinct activated constituents, plus relative-charge inertia",
   "parity":"flavor exchange I:b1<->b2 exactly conserved; odd eigenstate stable if lightest parity-odd and all other sectors even",
   "inertia_note":"eta is an independently specified dimensionless ratio, NOT derived by the copy-time inequality",
   "absolute_mass":"undetermined without u and spacetime normalization",
   "N250_eta1":next(x for x in rows if x["N"]==250 and x["eta"]==1),
   "data":rows,
   "not_claimed":["continuum theorem for any eta","fermionic spin-statistics","absolute relic mass","DM relic abundance","vacuum energy","H0 tension"]
 }
 out=Path("results/qict_odd_parity_spectrum/parity_spectrum.json")
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(result,indent=2)+"\n")
 print("QICT_PARITY_GAP",json.dumps({k:v for k,v in result.items() if k!="data"}),flush=True)

if __name__=="__main__":main()
