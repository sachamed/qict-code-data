"""Conditional Z3 two-phase homogeneous Einstein--scalar background.

Three receiver-link phases with one cyclic closure constraint leave two
relative phases. This is a standalone *homogeneous* EFT integration,
NOT a CLASS/CAMB perturbation, a likelihood, or a derivation from QICT.
"""
import json,math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

N=250
H0=67.4/299792.458
OM,OR=.315,9.3e-5
OL=1-OM-OR
A=np.array([[2.,-1.],[-1.,2.]])
LINK=np.array([[1.,0.],[-1.,1.],[0.,-1.]])
a,b,c=4/N**2,6*(N-1)/N**2,(N-1)*(N-2)/N**2

def background(q,p,u,V0,G):
    x=1-np.cos(LINK@q)
    P=a*x+b*x*x+c*x**3
    V=V0*np.sum(P)
    Hbg2=H0**2*(OR*np.exp(-4*u)+OM*np.exp(-3*u)+OL)
    R=float(p@G@p)
    if R>=6:raise ValueError("supercritical kinetic fraction")
    H2=(Hbg2+V/3)/(1-R/6)
    f=(H2*R/2+V)/(3*H2)
    return H2,V,f

def integrate(q0,F0=.2,V0=80.):
    G=F0**2*A/3
    Gi=np.linalg.inv(G)
    def rhs(u,y):
      q=y[:2];p=y[2:]
      phases=LINK@q;x=1-np.cos(phases)
      grad=LINK.T@(V0*(a+2*b*x+3*c*x*x)*np.sin(phases))
      H2,V,f=background(q,p,u,V0,G)
      dlnH=-(3*OM*np.exp(-3*u)+4*OR*np.exp(-4*u))*H0**2/(2*H2)-p@G@p/2
      acc=-(3+dlnH)*p-Gi@grad/H2
      return [p[0],p[1],acc[0],acc[1]]
    uu=np.linspace(math.log(1e-7),math.log(1/101),1700)
    sol=solve_ivp(rhs,(uu[0],uu[-1]),[*q0,0.,0.],t_eval=uu,method="DOP853",rtol=1e-9,atol=1e-11)
    if not sol.success:raise RuntimeError(sol.message)
    z=np.exp(-sol.t)-1
    pts=[background(sol.y[:2,i],sol.y[2:,i],u,V0,G) for i,u in enumerate(uu)]
    H2=np.asarray([p[0] for p in pts])
    f=np.asarray([p[2] for p in pts])
    kinetic=3*H2*np.einsum("ij,ji->i",sol.y[2:,:].T@G,sol.y[2:,:])
    drho=np.gradient(3*H2*f,uu)
    normal=np.maximum(np.abs(drho)+np.abs(kinetic),1e-14)
    res=np.abs(drho+kinetic)/normal
    window=(z>500)&(z<10000)
    ii=np.where(window)[0]
    j=ii[np.argmax(f[window])]
    peaks,_=find_peaks(f[window],prominence=.005,distance=30)
    result={
      "initial_phases":list(q0),"F0_over_Mpl":F0,
      "V0_over_Mpl2_in_Mpc_minus2":V0,
      "peak_redshift":float(z[j]),"peak_fraction":float(f[j]),
      "f_scalar_z1100":float(np.interp(1100,z[::-1],f[::-1])),
      "post_peak_oscillatory_local_maxima":[{"z":float(z[ii[k]]),"f":float(f[ii[k]])} for k in peaks],
      "positive_energy":bool(np.min(f)>=0),
      "sampled_Ward_p95":float(np.quantile(res[window],.95)),
      "sampled_Ward_max":float(np.max(res[window])),
      "sampling_grid":len(uu)
    }
    if not result["positive_energy"] or result["sampled_Ward_p95"]>.02:
        raise RuntimeError("Positive energy or sampled conservation check failed")
    return result

def main():
    phases=([2.5,0],[2.5,1.2],[1.7,2.5],[2.7,-.6],[1.5,-1.5])
    rows=[integrate(list(q)) for q in phases]
    result={
        "status":"passed_homogeneous_background",
        "model":"conditional Z3 two-relative-phase canonical system with finite-N link potential",
        "assumptions":"equal channel stiffness, initial velocities zero, background standard matter+radiation+Lambda, no scalar-dark-matter exchange",
        "not_computed":"full two-field linear perturbations, CMB, BAO, posterior, dark matter relic, observed H0",
        "results":rows
    }
    out=Path("results/qict_z3_two_phase_background/trajectories.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    for row in rows:print("QICT_Z3_TWO_PHASE_BACKGROUND",json.dumps(row),flush=True)

if __name__=="__main__":main()
