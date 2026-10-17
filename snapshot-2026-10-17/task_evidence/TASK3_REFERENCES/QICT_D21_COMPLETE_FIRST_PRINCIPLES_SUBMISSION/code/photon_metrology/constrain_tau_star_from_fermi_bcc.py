#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; CERT=ROOT/'certificates'
IN=DATA/'FERMI_LAT_GRB_SOURCE_CORRECTED_2013.csv'; BR=CERT/'BCC_GAUGE_PHOTON_BRIDGE.json'
OUT=CERT/'FERMI_BCC_TAU_STAR_CONSTRAINT.json'; OUTCSV=DATA/'FERMI_BCC_TAU_STAR_CONSTRAINTS.csv'
HBAR=6.582119569e-25 # GeV s

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    bridge=json.loads(BR.read_text()); bmin=bridge['directional_bounds']['b_min_axis']; bmax=bridge['directional_bounds']['b_max_body_diagonal']
    out=[]
    with IN.open() as f:
      for r in csv.DictReader(f):
        L=float(r['EQG2_subluminal_95_GeV'])
        # Fermi convention for n=2: v/c=1-(3/2)(E/E_QG2)^2.
        # QICT BCC tangent: v/c=1-b(n)(E/E_tau)^2.
        # Therefore E_tau > L sqrt(2 b/3).
        e_lo_min=L*math.sqrt(2*bmin/3); e_lo_max=L*math.sqrt(2*bmax/3)
        item={'grb':r['grb'],'theta_deg':float(r['theta_deg']),'phi_deg':float(r['phi_deg']),
              'EQG2_subluminal_95_GeV':L,
              'E_tau_lower_any_BCC_orientation_GeV':e_lo_min,'tau_star_upper_any_BCC_orientation_s':HBAR/e_lo_min,
              'E_tau_lower_body_diagonal_GeV':e_lo_max,'tau_star_upper_body_diagonal_s':HBAR/e_lo_max,
              'SME_d6_interval_contains_zero':float(r['SME_d6_lower_1e_minus20_GeV_minus2'])<=0<=float(r['SME_d6_upper_1e_minus20_GeV_minus2'])}
        out.append(item)
    strongest=max(out,key=lambda x:x['E_tau_lower_any_BCC_orientation_GeV'])
    with OUTCSV.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
    cert={
      'status':'CALCULATED_RETROSPECTIVE_FERMI_BOUND_UNDER_BCC_GAUGE_TANGENT',
      'taxonomy':{'QICT_BCC_photon_tangent':'DEMONSTRATED','Fermi_source_corrected_limits':'PUBLISHED_CALCULATED','tau_star':'BOUNDED_NOT_POINT_MEASURED','directional_variation_detection':'NULL'},
      'fermi_provenance':'Vasileiou et al., Phys. Rev. D 87, 122001 (2013), source-intrinsic-corrected Table V quadratic subluminal limits and Table VI directional d=6 intervals. Fermi LAT event data are public; this present formulation calculation maps the published calibrated likelihood limits and does not claim a fresh raw-FITS reanalysis.',
      'mapping':{'Fermi':'v/c=1-(3/2)(E/E_QG2)^2','QICT':'v/c=1-b(n)(E/E_tau)^2','relation':'E_tau > E_QG2_limit sqrt(2 b/3); tau_star < hbar/E_tau_lower'},
      'b_range':{'min':bmin,'max':bmax},'per_GRB':out,'strongest_orientation_independent_95pct':strongest,
      'headline':{'E_tau_lower_GeV':strongest['E_tau_lower_any_BCC_orientation_GeV'],
                  'E_tau_lower_eV':strongest['E_tau_lower_any_BCC_orientation_GeV']*1e9,
                  'tau_star_95pct_interval_s':[0.0,strongest['tau_star_upper_any_BCC_orientation_s']],
                  'tau_star_upper_s':strongest['tau_star_upper_any_BCC_orientation_s']},
      'directional_result':'All four published source-corrected d=6 directional intervals contain zero. No directional variation of photon speed is detected; present formulation therefore obtains an upper limit on tau_star, not a finite central measurement.',
      'comparison_status':'RETROSPECTIVE_MODEL_TRANSLATION: the Fermi-LAT timing interval constrains tau_* under the declared photon tangent mapping. The sealed angular fingerprint is evaluated only in a separate held-out comparison.',
      'input_sha256':{'fermi_table':sha(IN),'bridge':sha(BR),'script':sha(__file__)}
    }
    assert strongest['grb']=='090510'
    assert abs(strongest['E_tau_lower_any_BCC_orientation_GeV']-1.0e10)<1e-3
    assert abs(strongest['tau_star_upper_any_BCC_orientation_s']-6.582119569e-35)<1e-47
    assert all(x['SME_d6_interval_contains_zero'] for x in out)
    OUT.write_text(json.dumps(cert,indent=2)+'\n')
    print('CALCULATED_RETROSPECTIVE_FERMI_BOUND_UNDER_BCC_GAUGE_TANGENT')
    print(json.dumps(cert,indent=2))
if __name__=='__main__': main()
