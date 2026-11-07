"""Exact receiver inertia, parity-gap and vacuum-hierarchy compatibility test.

Finite-bosonic Schwinger spin (N charge quanta), distinct triplet parent.
This is a structural test, not a derived cosmological constant or DM relic.
"""
import json
import math
from pathlib import Path
import numpy as np

OUT=Path("results/qict_parity_vacuum_hierarchy")
OUT.mkdir(parents=True,exist_ok=True)


def theory(N,eta):
    if int(N)!=N or N<3 or eta<0:raise ValueError("N integer >= 3 and eta >=0")
    A00=N/4
    A11=3*N/4-.5
    A22=5*N/4-2
    A02=math.sqrt(2*N*(N-1))/4
    lam_min=(A00+A22-math.sqrt((A22-A00)**2+4*A02*A02))/2
    c=(A11-lam_min)/N**2
    return {
        "N":N,"eta":eta,"coherent_inertia_over_u":eta/(4*N),
        "expected_distinct_triplet_coherent_potential":"(1-cos(theta))^3+eta/(4N)",
        "perturbation_even_lower_coeff":lam_min/N**2,
        "perturbation_odd_coeff":A11/N**2,
        "perturbation_odd_even_gap_over_u":c*eta,
        "first_order_validity":"eta smaller than unperturbed excited spectral gaps",
        "degenerate_even_basis":"K=0,2; odd zero mode K=1",
        "coefficient":c,
        "independent_field_ranks":{"single_periodic_phase":1,"two_independent_canonical_phases":2}
    }


def matrix_spectrum(N,eta):
    j=N/2
    mm=np.arange(-j,j+1,dtype=float)
    Jx=np.zeros((N+1,N+1),dtype=float)
    for k in range(N):
        v=0.5*math.sqrt((j-mm[k])*(j+mm[k]+1))
        Jx[k,k+1]=v;Jx[k+1,k]=v
    K=j*np.eye(N+1)-Jx
    H=(8/(N*(N-1)*(N-2)))*(K@(K-np.eye(N+1))@(K-2*np.eye(N+1)))
    H+=(eta/N**2)*np.diag(mm**2)
    basis_p=[];basis_m=[]
    for i in range((N+1)//2):
        v=np.zeros(N+1);v[i]=v[N-i]=2**-0.5;basis_p.append(v)
        w=np.zeros(N+1);w[i]=2**-0.5;w[N-i]=-2**-0.5;basis_m.append(w)
    if N%2==0:
        v=np.zeros(N+1);v[N//2]=1;basis_p.append(v)
    P=np.asarray(basis_p).T;M=np.asarray(basis_m).T
    evp=np.linalg.eigvalsh(P.T@H@P)
    evm=np.linalg.eigvalsh(M.T@H@M)
    return {"ground_even_over_u":float(evp[0]),"ground_odd_over_u":float(evm[0]),
            "odd_even_gap_over_u":float(evm[0]-evp[0]),"even_second_gap_over_u":float(evp[1]-evp[0]),
            "parity_mixing_residual":float(np.max(np.abs(P.T@H@M)))}


def binom_expectation(N,theta,eta):
    p=(1-math.cos(theta))/2
    K=np.arange(N+1,dtype=int)
    # Second moment of Jz in balanced N-boson phase coherent state is exactly N/4.
    triplet=math.fsum(math.comb(N,int(k))*p**int(k)*(1-p)**(N-int(k))
                  * (8*k*(k-1)*(k-2)/(N*(N-1)*(N-2)))
                  for k in K)
    return triplet+eta/(4*N)


def main():
    checks=0
    for N in (3,4,7,10,20,40):
      for eta in (0,.01,1.):
        data=theory(N,eta)
        for theta in (0,.2,1.1,2.5,math.pi):
            target=(1-math.cos(theta))**3+eta/(4*N)
            independent=binom_expectation(N,theta,eta)
            assert abs(target-independent)<3e-12*max(1,target),(N,eta,theta)
            checks+=1
        if eta>0:
            spec=matrix_spectrum(N,eta)
            assert spec["odd_even_gap_over_u"]>0
            assert spec["parity_mixing_residual"]<1e-10
            checks+=2
    N=250
    eta_hier=4/N**3
    approximation=theory(N,eta_hier)
    # Full high-precision double diagonalization resolves at 1e-9 for N=250.
    nums=[]
    for eta in (0.01,1.0,eta_hier):
        d=matrix_spectrum(N,eta)
        nums.append({"eta":eta,**d})
        assert d["odd_even_gap_over_u"]>0, d
        checks+=1
    inferred=next(d for d in nums if d["eta"]==eta_hier)
    predicted=approximation["perturbation_odd_even_gap_over_u"]
    assert abs(inferred["odd_even_gap_over_u"]-predicted)<1e-10,(predicted,inferred)
    checks+=1
    # A 3-harmonic 1-variable potential is not a 3-field system.
    for th in (.1,1.,2.5,3.):
        x=1-math.cos(th)
        harmonic=2.5-3.75*math.cos(th)+1.5*math.cos(2*th)-.25*math.cos(3*th)
        assert abs(x**3-harmonic)<3e-14
        checks+=1
    result={
     "status":"passed_exact_and_finite_spectrum",
     "tests":checks,"N":N,
     "theorem":"distinct-triplet coherent expectation plus positive inertia = x^3+eta/(4N) exactly",
     "caution":"the constant is a coherent-state expectation, not the exact many-body ground-state energy nor a physical renormalized cosmological constant",
     "charge_eta_N4_ansatz":{
         "status":"TUNED_HIERARCHY_NOT_DERIVED_FROM_RECEIVER",
         "goal":"eta/(4N)=N^-4",
         "eta_required":eta_hier,
         "coherent_inertia_ratio":eta_hier/(4*N),
         "linearized_gap_predicted_over_u":predicted,
         "finite_matrix_gap_over_u":inferred["odd_even_gap_over_u"],
         "physical_mass":"undetermined without u, lattice spacing and kinetic normalization"
     },
     "eta1":nums[1],
     "spectra":nums,
     "internal_parity_is_not_spin_statistics":"Schwinger bosonic flavor-exchange odd state does not become Dirac fermion from Z2 parity",
     "three_harmonics_are_not_three_dynamical_fields":True,
     "potential_harmonics":"(1-cos theta)^3=5/2-15/4*cos theta+3/2*cos(2theta)-1/4*cos(3theta)",
     "required_for_true_two_field_EDE":"independent second canonical phase and kinetic term, derived from at least two commuting receiver modes",
     "excluded_claims":["quantum vacuum hierarchy proved","relic density","odd particle mass in eV",
                        "two-field Boltzmann fit","Hubble-tension resolution"]
    }
    (OUT/"parity_vacuum_compatibility.json").write_text(json.dumps(result,indent=2)+"\n")
    print("QICT_PARITY_VACUUM_HIERARCHY",json.dumps({k:v for k,v in result.items() if k!="spectra"}),flush=True)


if __name__=="__main__":main()
