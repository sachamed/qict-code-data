#!/usr/bin/env python3
"""Two non-identifiability theorems for QICT dark mass and vacuum.

1. F, V0 and v_R=1 do not fix the microscopic lattice scale ell.
2. Adding a symmetry-invariant c*Identity changes vacuum energy without
   changing parity sectors, excitation gaps or scalar gradients.

No attempt is made to infer a particle's physical mass from a local
spectral gap before deriving a relativistic dispersion relation.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

def microscopic_family(F,V0,N,ell):
    if min(F,V0,ell)<=0 or N<3:
        raise ValueError("F,V0,ell > 0 and N>=3")
    Delta=1/(4*F**2*ell**3)
    hopping=2*F**2*ell/N
    u=V0*ell**3
    output={
      "F":F,"V0":V0,"N":N,"ell":ell,
      "Delta":Delta,"hopping_t":hopping,"local_energy_u":u,
      "F2_reconstructed":1/(4*Delta*ell**3),
      "V0_reconstructed":u/ell**3,
      "relativistic_speed_squared":2*hopping*N*Delta*ell**2
    }
    return output

def spin_hamiltonian(N,eta=1.):
    j=N/2
    m=np.arange(-j,j+1)
    Jx=np.zeros((N+1,N+1))
    for k in range(N):
        v=.5*math.sqrt((j-m[k])*(j+m[k]+1))
        Jx[k,k+1]=Jx[k+1,k]=v
    I=np.eye(N+1)
    K=j*I-Jx
    H=8/(N*(N-1)*(N-2))*K@(K-I)@(K-2*I)+(eta/N**2)*np.diag(m*m)
    P=I[::-1]
    assert np.max(np.abs(P@H-H@P))<1e-9
    return H,P

def even_odd_eigenvalues(H,P):
    N=H.shape[0]-1
    bp=[];bm=[]
    for k in range((N+1)//2):
        a=np.zeros(N+1);a[k]=a[N-k]=2**-.5;bp.append(a)
        a=np.zeros(N+1);a[k]=2**-.5;a[N-k]=-2**-.5;bm.append(a)
    if N%2==0:
        a=np.zeros(N+1);a[N//2]=1;bp.append(a)
    Vp=np.asarray(bp).T;Vm=np.asarray(bm).T
    assert np.max(np.abs(Vp.T@H@Vm))<1e-9
    return (np.linalg.eigvalsh(Vp.T@H@Vp),
            np.linalg.eigvalsh(Vm.T@H@Vm))

def main():
    checks=0
    F,V0,N=.225,201.55504807115258,250
    cases=[microscopic_family(F,V0,N,ell) for ell in (.5,1.,2.,5.)]
    for d in cases:
      assert abs(d["F2_reconstructed"]/F**2-1)<1e-12
      assert abs(d["V0_reconstructed"]/V0-1)<1e-12
      assert abs(d["relativistic_speed_squared"]-1)<1e-12
      checks+=3
    assert abs(cases[2]["local_energy_u"]/cases[1]["local_energy_u"]-8)<1e-12
    checks+=1
    H,P=spin_hamiltonian(20,eta=1.)
    ep,em=even_odd_eigenvalues(H,P)
    gap=float(em[0]-ep[0])
    assert gap>0;checks+=1
    shifted=[]
    for c in (0.,.05,.2,1.):
      Hc=H+c*np.eye(H.shape[0])
      epc,emc=even_odd_eigenvalues(Hc,P)
      assert abs((emc[0]-epc[0])-gap)<1e-12
      assert np.allclose(epc-ep,c,atol=1e-12)
      assert np.allclose(emc-em,c,atol=1e-12)
      assert np.max(np.abs(Hc@P-P@Hc))<1e-10
      checks+=4
      shifted.append({"identity_counterterm_over_u":c,
        "exact_even_ground_over_u":float(epc[0]),
        "odd_even_gap_over_u":float(emc[0]-epc[0]),
        "local_vacuum_density_shift_per_cell_in_u_per_ell3":c})
    output={
      "status":"passed_scale_and_vacuum_non_identifiability_gates",
      "tests":checks,
      "conditional_rotor_dictionary":{
        "F2":"1/(4 Delta ell^3)",
        "V0":"u/ell^3",
        "vR2":"2 t N Delta ell^2",
        "ell_is_undetermined":"yes, a continuous one-parameter family of Delta,t,u yields the same F,V0,vR"},
      "family_at_fixed_cosmological_scales":cases,
      "local_gap_at_N20_eta1_over_u":gap,
      "absolute_odd_gap":"proportional to u and therefore to ell^3 at fixed F,V0; interpreting this as a relativistic particle mass needs its derived dispersion and normalization",
      "identity_counterterm":{
        "Hprime":"H + c u Identity",
        "commutes_with_all_symmetries":"yes",
        "energy_differences_unchanged":"yes",
        "vacuum_energy_density_shift":"c u/ell^3 = c V0",
        "spectra":shifted
      },
      "physical_implication":"A unique late vacuum density and absolute dark mass cannot follow from the tested local spectra and conditional rotor dictionary alone",
      "new_independent_data_needed":[
         "microscopic QICT relation fixing ell or Delta (e.g., a derived copy-time-to-length map)",
         "relativistic odd-sector dispersion and normalized one-particle state",
         "relic production and freeze-out/nonthermal collision integral",
         "gravitational vacuum renormalization/selection rule for identity operator"
      ],
      "scope":"Exact within stated finite-site model; no universal no-go for more complete QICT primitives",
      "not_claimed":["no dark matter exists","the cosmological constant cannot be derived in a UV theory",
                     "the Hubble tension is mathematically impossible"]
    }
    p=Path("results/qict_absolute_scale_no_go/scale_and_vacuum_gate.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(output,indent=2)+"\n")
    print("QICT_ABSOLUTE_SCALE_VACUUM_NO_GO",json.dumps({k:v for k,v in output.items() if k!="family_at_fixed_cosmological_scales"}),flush=True)

if __name__=="__main__":
    main()
