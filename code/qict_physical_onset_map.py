#!/usr/bin/env python3
"""Analytical QICT finite-N onset reconstruction with explicit validity gates.

Uses an instantaneous constant-H tachyonic growth approximation. Gives
initial guesses for numerical shooting, never an observed CMB fit.
"""
from __future__ import annotations
import json,math
from pathlib import Path

C=299792.458  # km/s
N=250
THETA=2.5
OMEGA_BH2=.02237
OMEGA_CH2=.12
N_EFF=3.044
OMEGA_GAMMA_H2=2.469e-5
OMEGA_RH2=OMEGA_GAMMA_H2*(1+0.22710731766*N_EFF)


def micro(theta,N):
    if N<3 or int(N)!=N:raise ValueError("integer charge N>=3 required")
    a=4/N**2;b=6*(N-1)/N**2;c=(N-1)*(N-2)/N**2
    x=1-math.cos(theta)
    P=a*x+b*x*x+c*x**3
    P1=a+2*b*x+3*c*x*x
    P2=2*b+6*c*x
    Q=P2*math.sin(theta)**2+P1*math.cos(theta)
    return {"theta":theta,"x":x,"P":P,"Ptheta2":Q,"coefficients":[a,b,c]}


def recon(zc,f,theta=THETA,N=N,h0=67.4,omega_lambda=.685,alpha=4.):
    if not (0<f<1) or not zc>0 or not alpha>0:
        raise ValueError("physical inputs require 0<f<1, zc>0, alpha>0")
    m=micro(theta,N)
    if not m["Ptheta2"]<0:
        raise ValueError("initial field must be on tachyonically curved side")
    h=h0/100.
    bg=((OMEGA_BH2+OMEGA_CH2)/h**2*(1+zc)**3+
        OMEGA_RH2/h**2*(1+zc)**4+omega_lambda)
    H2=(h0/C)**2*bg/(1-f)
    F2=3*f*abs(m["Ptheta2"])/(alpha*m["P"])
    V0=3*f*H2/m["P"]
    F=math.sqrt(F2)
    return {
       "N":N,"theta_ini":theta,"target_zc":zc,"target_frozen_fraction":f,
       "onset_growth_threshold_mu2_over_H2":alpha,
       "onset_growth_rate_over_H_if_H_constant":(-3+math.sqrt(9+4*alpha))/2,
       "F_over_Mpl_approx":F,"phi_ini_over_Mpl_approx":F*theta,
       "V0_over_Mpl2_in_Mpc_minus2_approx":V0,
       "Hzc_Mpc_inverse_approx":math.sqrt(H2),
       "local_mu2_over_H2":V0*abs(m["Ptheta2"])/(F2*H2),
       "bare_potential_over_total_at_onset":V0*m["P"]/(3*H2),
       "status":"ANALYTICAL_INITIAL_GUESS_NOT_NATIVE_CLASS_CALIBRATION",
       "assumptions":["slow initial field, negligible scalar kinetic energy",
                      "instantaneous constant-H growth-rate proxy",
                      "background omega(lambda) fixed",
                      "zc is rolling-onset proxy, not measured energy-peak redshift",
                      "Einstein gravity and canonical scalar normalization assumed"]
    }


def inflection(N):
    lo,hi=1.8,math.pi
    assert micro(lo,N)["Ptheta2"]>0 and micro(hi,N)["Ptheta2"]<0
    for i in range(100):
        mid=.5*(lo+hi)
        if micro(mid,N)["Ptheta2"]>0:lo=mid
        else:hi=mid
    return .5*(lo+hi)


def main():
    theta_c=inflection(N)
    rows=[recon(zc,f,THETA,N) for zc in (3000,3500,4000)
                              for f in (.05,.075,.10,.125)]
    m=micro(THETA,N)
    old={"V0":201.55504807115258,"F":.225,"phi_initial":.5625}
    h=67.4/100.; z=3500
    bg=((OMEGA_BH2+OMEGA_CH2)/h**2*(1+z)**3+
        OMEGA_RH2/h**2*(1+z)**4+.685)
    H2bg=(67.4/C)**2*bg
    old_fproxy=(old["V0"]*m["P"]/3)/(H2bg+old["V0"]*m["P"]/3)
    old_alpha=(old["V0"]/old["F"]**2*abs(m["Ptheta2"]))/(H2bg/(1-old_fproxy))
    result={"status":"passed","N":N,"theta_inflection_rad":theta_c,
      "target_theta":THETA,"potential_P":m["P"],"curvature_P_theta_theta":m["Ptheta2"],
      "growth_proxy_definition":"r/H=(-3+sqrt(9+4 mu²/H²))/2; r/H=1 -> mu²/H²=4",
      "selected_reconstruction":recon(3500,.1,THETA,N),
      "previous_native_input_comparison":{
         **old,"nominal_frozen_energy_fraction_at_z3500":old_fproxy,
         "mu2_over_H2_at_nominal_z3500_under_frozen_proxy":old_alpha,
         "caution":"frozen proxy is NOT a CLASS evolution or physical zpeak"},
      "grid":rows,
      "interpretation":"This is a two-dimensional analytic seed for a full nonlinear shooting (zc,f_peak). No new cosmological observations are computed."}
    assert abs(result["selected_reconstruction"]["local_mu2_over_H2"]-4)<1e-12
    assert abs(result["selected_reconstruction"]["bare_potential_over_total_at_onset"]-.1)<1e-12
    assert theta_c<THETA<math.pi and 0<old_fproxy<1
    path=Path("results/qict_physical_onset_map/onset_seed.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print("QICT_ONSET_SEED",json.dumps({k:v for k,v in result.items() if k!="grid"}),flush=True)


if __name__=="__main__":main()
