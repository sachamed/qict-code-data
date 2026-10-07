#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

B = {'axis':3/32,'face':7/32,'body':25/96}

def sky_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra=math.radians(ra_deg); dec=math.radians(dec_deg)
    return np.array([math.cos(dec)*math.cos(ra),math.cos(dec)*math.sin(ra),math.sin(dec)])

def rot_zyx(alpha,beta,gamma):
    a,b,g=map(math.radians,(alpha,beta,gamma))
    Rz=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1.]])
    Ry=np.array([[math.cos(b),0,math.sin(b)],[0,1.,0],[-math.sin(b),0,math.cos(b)]])
    Rx=np.array([[1.,0,0],[0,math.cos(g),-math.sin(g)],[0,math.sin(g),math.cos(g)]])
    return Rz@Ry@Rx

def b_of(n):
    n=np.asarray(n,float); n=n/np.linalg.norm(n)
    return float((11-8*np.sum(n**4))/32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--references',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); ref=json.loads(Path(args.references).read_text())
    hbar=ref['constants']['hbar_GeV_s']; Epl=ref['constants']['planck_energy_GeV']
    rows=[]
    for key,d in ref['photon_time_of_flight'].items():
        EQ=float(d['E_QG2_lower_GeV_95cl']); E=float(d['representative_energy_GeV'])
        # Standard quadratic LIV convention: v/c = 1-(3/2)(E/E_QG,2)^2.
        # Matching b(n)/E_tau^2 to (3/2)/E_QG,2^2 gives E_tau/E_QG,2=sqrt(2b/3).
        ratios={name:math.sqrt(2*b/3) for name,b in B.items()}
        Etau_weak=ratios['axis']*EQ          # weakest orientation-specific lower bound = E_QG/4
        Etau_uniform=ratios['body']*EQ       # uniform-in-direction lower bound = 5 E_QG/12
        rows.append({
          'analysis':key,'source':d['source'],'E_QG2_lower_GeV_95cl':EQ,
          'E_tau_over_E_QG2_by_high_symmetry':ratios,
          'orientation_envelope_E_tau_lower_GeV':[Etau_weak,Etau_uniform],
          'weakest_orientation_E_tau_lower_GeV':Etau_weak,
          'uniform_all_directions_E_tau_lower_GeV':Etau_uniform,
          'weakest_orientation_tau_star_upper_s':hbar/Etau_weak,
          'uniform_all_directions_tau_star_upper_s':hbar/Etau_uniform,
          'representative_energy_GeV':E,
          'speed_defect_range_at_weakest_orientation_bound':[B['axis']*(E/Etau_weak)**2,B['body']*(E/Etau_weak)**2]
        })
    # strongest lower bound among the time-of-flight analyses
    strongest=max(rows,key=lambda x:x['E_QG2_lower_GeV_95cl'])
    test_energies=[30.,99.3,7000.,13000.,300000.]
    benchmarks={}
    for name,Etau in [('Planck_clock_benchmark',Epl),('strongest_time_of_flight_weak_orientation_bound',strongest['weakest_orientation_E_tau_lower_GeV']),('strongest_time_of_flight_uniform_bound',strongest['uniform_all_directions_E_tau_lower_GeV']),('1_PeV',1e6),('1_TeV',1e3)]:
        vals=[]
        for E in test_energies:
            x=E/Etau
            vals.append({'E_GeV':E,'E_over_Etau':x,'expansion_parameter_squared':x*x,
                         'axis_defect':B['axis']*x*x,'face_defect':B['face']*x*x,'body_defect':B['body']*x*x,
                         'quadratic_regime': ('asymptotic_quadratic' if abs(x)<0.1 else 'finite_energy_extrapolation')})
        benchmarks[name]={'E_tau_GeV':Etau,'tau_star_s':hbar/Etau,'values':vals}
    # Directional demonstrator: GRB direction for identity BCC-to-J2000 rotation; orientation is a fit parameter in data analysis.
    grb=ref['grb221009a']; n=sky_unit(grb['ra_deg_J2000'],grb['dec_deg_J2000'])
    demo={'ra_deg':grb['ra_deg_J2000'],'dec_deg':grb['dec_deg_J2000'],'identity_frame_b':b_of(n)}
    # Verify rotational range on a deterministic dense sphere sample.
    vals=[]
    for z in np.linspace(-1,1,401):
        r=math.sqrt(max(0,1-z*z))
        for phi in np.linspace(0,2*math.pi,720,endpoint=False):
            vals.append(b_of([r*math.cos(phi),r*math.sin(phi),z]))
    out={'classification':'PHOTON_OBSERVATIONAL_TRANSLATION','b_coefficients':B,
         'time_of_flight_translations':rows,'strongest_cited_time_of_flight_row':strongest,
         'exact_LIV_matching':{'standard_quadratic_velocity':'v/c = 1 - (3/2)(E/E_QG,2)^2',
           'QICT_velocity':'v/c = 1 - b(n)(E/E_tau)^2 + O(E^4)',
           'matching':'E_tau/E_QG,2 = sqrt(2 b(n)/3)',
           'axis_ratio_exact':'1/4','body_ratio_exact':'5/12','orientation_envelope_exact':'[1/4,5/12]'},
         'benchmark_speed_defects':benchmarks,
         'directional_test':{
           'model':'b(R n) with one common R in SO(3) across all sources',
           'grb221009a_identity_frame_demonstrator':demo,
           'sampled_b_min':min(vals),'sampled_b_max':max(vals),
           'exact_b_min':B['axis'],'exact_b_max':B['body'],
           'fit_parameters':'three orientation angles plus E_tau, with source-intrinsic lag nuisance parameters',
           'interpretation':'the identity-frame value is a coordinate demonstrator; a physical sky test jointly fits the common celestial-to-BCC rotation'
         }}
    Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'classification':out['classification'],'strongest':strongest,'demo':demo},indent=2))
if __name__=='__main__': main()
