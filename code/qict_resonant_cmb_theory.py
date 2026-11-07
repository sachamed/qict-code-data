"""Causal QICT resonance: Einstein-Friedmann background and Planck-ready scalar fluid.

The resonance spectral mapping f(H)=A*g^2*x^2/[(1-x^2)^2+g^2*x^2],
x=H/Hc, is a conditional QICT constitutive hypothesis, NOT a theorem
from the receiver-optimised copy-time bound in Quantum Reports (2026).

The full homogeneous density is rho_phi = rho_floor+3 Mpl^2 H^2 f(H).
A canonical scalar has c_s,rest^2=1 and zero intrinsic shear.
A reconstructed potential may be negative when w_phi>1. The fluid
realisation does not prove the proposed microphysical stress tensor.
"""
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.laguerre import laggauss
from cobaya.theory import Theory

HC_FID=100.0*np.sqrt(.143*1701.**3+4.18e-5*1701.**4)

def response(E,H0,A=.30,g=1.60,Hc=HC_FID):
    t=np.asarray(E)**2*(H0/Hc)**2
    D=(1-t)**2+g*g*t
    return A*g*g*t/D, 2*A*g*g*t*(1-t*t)/(D*D)

def history(n=1600,H0=69.28,ombh2=.02237,omch2=.119,
            nnu=3.044,mnu=.06,A=.30,g=1.60,Hc=HC_FID):
    h=H0/100;om=(ombh2+omch2)/h**2;og=2.469e-5/h**2
    a=np.geomspace(1e-8,1,n)
    Tnu=2.7255*(4/11)**(1/3)*8.617333262145e-5
    q,w0=laggauss(96);weights=w0/(1+np.exp(-q))
    y=a[:,None]*(mnu/Tnu);en=np.sqrt(q[None,:]**2+y**2)
    I=np.sum(weights[None,:]*q[None,:]**2*en,axis=1)
    J=np.sum(weights[None,:]*q[None,:]**4/en,axis=1)
    I0=7*np.pi**4/120
    fac=og*a**-4*(15/np.pi**4)*(4/11)**(4/3);nm=nnu/3
    rho_nu=fac*((nnu-nm)*I0+nm*I)
    p_nu=fac*((nnu-nm)*I0/3+nm*J/3)
    rad=og*a**-4
    floor=1-om-og-rho_nu[-1]-response(1,H0,A,g,Hc)[0]
    if not 0<=A<.5 or floor<=0 or g<np.sqrt(2):
        raise ValueError("Fails sufficient monotonic Friedmann condition")
    rhs=om*a**-3+rad+rho_nu+floor
    drhs=-3*om*a**-3-4*rad-3*(rho_nu+p_nu)
    E=np.zeros(n); w=np.zeros(n);fr=np.zeros(n);rs=np.zeros(n)
    for j,R in enumerate(rhs):
        low=np.sqrt(R)*(1-1e-10)
        high=np.sqrt(R)/np.sqrt(1-A)*(1+1e-10)
        E[j]=brentq(lambda e:e*e*(1-response(e,H0,A,g,Hc)[0])-R,
                    low,high,rtol=1e-13)
        f,df=response(E[j],H0,A,g,Hc)
        dlnEdN=drhs[j]/(E[j]**2*(2*(1-f)-df))
        rx=floor+E[j]**2*f
        w[j]=-1-E[j]**2*(2*f+df)*dlnEdN/(3*rx)
        fr[j]=f
        rs[j]=(E[j]**2*(1-f)-R)/E[j]**2
    if np.min(w)<-1-1e-9:
        raise ValueError("Physical phantom reconstructed branch")
    # CAMB's tabulated-fluid constructor rejects *any* entry w<-1.
    # Floating-point cancellation gives O(1e-16) undershoots at frozen epochs;
    # restoring their mathematical value -1 changes no physical prediction.
    w=np.maximum(w,-1.0)
    return a,w,E,fr,float(np.max(np.abs(rs))),float(floor)

class ResonantQICTTheory(Theory):
    params=dict(H0=None,ombh2=None,omch2=None,tau=None,logA=None,
                ns=None,A_res=None,mnu=None)
    lmax=4200
    damping_ratio=1.6
    Hc=HC_FID

    def initialize(self):self._lmax=int(self.lmax)
    def must_provide(self,**requirements):
        if "Cl" in requirements and isinstance(requirements["Cl"],dict):
            self._lmax=max(self._lmax,*[int(x) for x in requirements["Cl"].values()])
        return {}
    def calculate(self,state,want_derived=True,**p):
        import camb
        A=float(p["A_res"]);h=float(p["H0"])
        cp=camb.CAMBparams()
        cp.set_cosmology(H0=h,ombh2=float(p["ombh2"]),
              omch2=float(p["omch2"]),tau=float(p["tau"]),mnu=float(p["mnu"]),
              nnu=3.044,num_massive_neutrinos=int(float(p["mnu"])>0))
        cp.InitPower.set_params(As=float(np.exp(p["logA"])*1e-10),ns=float(p["ns"]))
        a,w,E,fr,err,floor=history(H0=h,ombh2=float(p["ombh2"]),
               omch2=float(p["omch2"]),mnu=float(p["mnu"]),A=A,
               g=float(self.damping_ratio),Hc=float(self.Hc))
        if A<=0:cp.set_dark_energy(w=-1,cs2=1,dark_energy_model="fluid")
        else:cp.set_dark_energy_w_a(a,w,dark_energy_model="fluid")
        cp.set_for_lmax(self._lmax,lens_potential_accuracy=4,
                        lens_output_margin=1250,nonlinear=True)
        cp.NonLinearModel.set_params(halofit_version="mead2016")
        cp.NonLinear=camb.model.NonLinear_both
        r=camb.get_results(cp)
        cls=r.get_cmb_power_spectra(cp,lmax=self._lmax,
                        CMB_unit="muK",raw_cl=True)["total"]
        pp=r.get_lens_potential_cls(lmax=self._lmax,raw_cl=True)[:,0]
        state["Cl"]={"ell":np.arange(len(pp)),"tt":cls[:,0],
                 "ee":cls[:,1],"bb":cls[:,2],"te":cls[:,3],"pp":pp}
        state["derived"]={}
        state["QICT_stats"]={"w_max":float(np.max(w)),"w_min":float(np.min(w)),
              "receiver_at_z1090":float(np.interp(1/1091.,a,fr)),
              "receiver_at_z3400":float(np.interp(1/3401.,a,fr)),
              "friedmann_residual":err,"floor":floor,
              "rdrag":float(r.get_derived_params().get("rdrag",np.nan))}
        return True
    def get_Cl(self,ell_factor=False,units="FIRASmuK2"):
        if units not in ("FIRASmuK2","muK2"):raise ValueError(units)
        x=self.current_state["Cl"]
        if not ell_factor:return x
        ell=x["ell"].astype(float);factor=ell*(ell+1)/(2*np.pi)
        o={"ell":x["ell"]}
        for key in ("tt","ee","bb","te"):o[key]=factor*x[key]
        o["pp"]=(ell*(ell+1))**2/(2*np.pi)*x["pp"]
        return o
