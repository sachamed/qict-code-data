#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def angle_set(B, Jabs):
    B=np.asarray(B,float)
    s13=math.sqrt(max(0.0,float(B[0,2]))); c13=math.sqrt(max(0.0,1-s13*s13))
    s12=math.sqrt(max(0.0,float(B[0,1])))/c13
    s23=math.sqrt(max(0.0,float(B[1,2])))/c13
    s12=min(1,max(0,s12)); s23=min(1,max(0,s23))
    c12=math.sqrt(max(0,1-s12*s12)); c23=math.sqrt(max(0,1-s23*s23))
    den=c12*c23*c13*c13*s12*s23*s13
    sind=min(1,max(-1,Jabs/den)) if den else 0.0
    cosd=(float(B[1,0])-s12*s12*c23*c23-c12*c12*s23*s23*s13*s13)/(2*s12*c12*s23*c23*s13) if s12*c12*s23*c23*s13 else 1.0
    cosd=min(1,max(-1,cosd))
    delta=math.atan2(abs(sind),cosd)
    return dict(theta12_deg=math.degrees(math.asin(s12)), theta13_deg=math.degrees(math.asin(s13)),
                theta23_deg=math.degrees(math.asin(s23)), delta_principal_deg=math.degrees(delta),
                delta_principal_rad=delta, delta_conjugate_deg=360-math.degrees(delta), J_abs=float(Jabs))

def anchor_map(svals, refs, names):
    s=np.asarray(svals,float)
    out={}
    for k,anchor in enumerate(names):
        scale=refs[anchor]/s[k]
        pred={names[i]:float(scale*s[i]) for i in range(3)}
        ratio={n:pred[n]/refs[n] for n in names}
        logerr=[math.log(pred[n]/refs[n]) for n in names]
        out[f'{anchor}_anchor']={
            'scale_per_dimensionless_unit_GeV':float(scale),
            'masses_GeV':pred,
            'prediction_to_reference_ratio':ratio,
            'log_rms_all_three':float(math.sqrt(np.mean(np.square(logerr))))
        }
    # best common multiplicative scale in log-space
    logs=[math.log(refs[n]/s[i]) for i,n in enumerate(names)]
    best_scale=math.exp(sum(logs)/3)
    pred={names[i]:float(best_scale*s[i]) for i in range(3)}
    out['best_log_scale']={
        'scale_per_dimensionless_unit_GeV':best_scale,
        'masses_GeV':pred,
        'prediction_to_reference_ratio':{n:pred[n]/refs[n] for n in names},
        'log_rms':float(math.sqrt(np.mean([(math.log(pred[n]/refs[n]))**2 for n in names])))
    }
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmarks',required=True); ap.add_argument('--references',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); bench=json.loads(Path(a.benchmarks).read_text()); ref=json.loads(Path(a.references).read_text())
    qr=ref['quark_masses']
    qgev={'u':qr['u']['value_MeV']/1000,'d':qr['d']['value_MeV']/1000,'s':qr['s']['value_MeV']/1000,
          'c':qr['c']['value_GeV'],'b':qr['b']['value_GeV'],'t':qr['t']['value_GeV']}
    su=bench['receiver_singular_values']['u']; sd=bench['receiver_singular_values']['d']
    up=anchor_map(su,{k:qgev[k] for k in ['u','c','t']},['u','c','t'])
    down=anchor_map(sd,{k:qgev[k] for k in ['d','s','b']},['d','s','b'])

    # Direct ratio comparisons. d/s share the same quoted 2 GeV MSbar convention and are particularly clean.
    quark_ratios={
      'finite_u_c':su[0]/su[1], 'reference_u_c_mixed_scale':qgev['u']/qgev['c'],
      'finite_c_t':su[1]/su[2], 'reference_c_t_mixed_scale':qgev['c']/qgev['t'],
      'finite_d_s':sd[0]/sd[1], 'reference_d_s_2GeV_MSbar':qgev['d']/qgev['s'],
      'finite_s_b':sd[1]/sd[2], 'reference_s_b_mixed_scale':qgev['s']/qgev['b'],
    }
    quark_ratios['d_s_ratio_factor']=quark_ratios['finite_d_s']/quark_ratios['reference_d_s_2GeV_MSbar']

    # Neutral finite shape: two independent one-anchor tests.
    s=np.asarray(bench['neutral_bare_takagi_singular_values'],float)
    ds21=float(s[1]**2-s[0]**2); ds31=float(s[2]**2-s[0]**2)
    pmns=ref['pmns_normal_ordering']
    dm21=float(pmns['delta_m21_eV2']); dm31=float(pmns['delta_m3l_eV2'])
    Benu=np.asarray(bench['B_enu'],float)
    def neutral_from_scale(scale):
        m=s*scale
        terms=Benu[0,:]*m
        mbeta=math.sqrt(float(np.sum(Benu[0,:]*m*m)))
        mbbmax=float(np.sum(terms)); mbbmin=max(0.0,float(2*np.max(terms)-np.sum(terms)))
        return {'masses_eV':m.tolist(),'sum_masses_eV':float(m.sum()),
                'delta_m21_eV2':float(m[1]**2-m[0]**2),'delta_m31_eV2':float(m[2]**2-m[0]**2),
                'm_beta_eV':mbeta,'m_bb_phase_interval_eV':[mbbmin,mbbmax]}
    atm=neutral_from_scale(math.sqrt(dm31/ds31)); sol=neutral_from_scale(math.sqrt(dm21/ds21))
    neutral={
      'dimensionless_takagi_singular_values':s.tolist(),
      'dimensionless_split_ratio':ds21/ds31,
      'reference_split_ratio':dm21/dm31,
      'shape_ratio_factor':(ds21/ds31)/(dm21/dm31),
      'atmospheric_anchor':atm,
      'solar_anchor':sol,
      'atmospheric_anchor_solar_split_ratio_to_reference':atm['delta_m21_eV2']/dm21,
      'solar_anchor_atmospheric_split_ratio_to_reference':sol['delta_m31_eV2']/dm31,
      'reference_constraints':ref['absolute_neutrino_constraints']
    }

    ckm_diag=angle_set(bench['B_ud'],float(bench['abs_J_ud']))
    pmns_diag=angle_set(bench['B_enu'],float(bench['abs_J_enu']))
    ckmref=ref['ckm']; ckmangles={'theta12_deg':math.degrees(math.asin(ckmref['sin_theta12'])),
        'theta13_deg':math.degrees(math.asin(ckmref['sin_theta13'])),'theta23_deg':math.degrees(math.asin(ckmref['sin_theta23'])),
        'delta_deg':math.degrees(ckmref['delta_rad']), 'J_abs':ckmref['J']}
    pmnsref={k:pmns[k] for k in ['theta12_deg','theta13_deg','theta23_deg','delta_deg','delta_m21_eV2','delta_m3l_eV2']}
    out={
      'classification':'FINITE_FLAVOR_COORDINATE_COMPARISON',
      'interpretation':{
        'finite_receiver':'The serialized finite receiver matrices define source-space mass and mixing coordinates whose shape is compared directly with reference observables.',
        'physical_map':'Quark masses, neutrino masses, CKM and PMNS are defined by the common-regulator interacting G2/G3 observables. The comparison quantifies the transformation from finite receiver coordinates to those interacting observables.'
      },
      'quark_direct_identification_test':{
        'reference_GeV':qgev,'up_sector':up,'down_sector':down,'ratio_tests':quark_ratios,
        'finite_receiver_role':'source-space hierarchy coordinate',
        'selected_physical_map':'common color-resolved renormalized inverse propagator; d/s comparison uses the quoted 2 GeV MSbar convention'
      },
      'neutral_direct_identification_test':{
        **neutral,'finite_receiver_role':'source-space Takagi coordinate',
        'selected_physical_map':'common interacting neutral Nambu kernel constrained jointly by the two oscillation splittings'
      },
      'ckm_finite_receiver_test':{'finite':ckm_diag,'reference':ckmangles,
        'angle_differences_deg':{k:ckm_diag[k]-ckmangles[k] for k in ['theta12_deg','theta13_deg','theta23_deg']},
        'finite_receiver_role':'source-space mixing coordinate'},
      'pmns_finite_receiver_test':{'finite':pmns_diag,'reference':pmnsref,
        'angle_differences_deg':{k:pmns_diag[k]-pmnsref[k] for k in ['theta12_deg','theta13_deg','theta23_deg']},
        'finite_receiver_role':'source-space mixing coordinate'},
      'scientific_consequence':'The finite carrier fixes structural ancestry and source coordinates, while the common interacting G2/G3 contraction supplies renormalized flavor observables in the regulator limit.'
    }
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'classification':out['classification'],'d_over_s_factor':quark_ratios['d_s_ratio_factor'],
                      'neutral_shape_factor':neutral['shape_ratio_factor'],
                      'ckm_angles':ckm_diag,'pmns_angles':pmns_diag},indent=2))
if __name__=='__main__': main()
