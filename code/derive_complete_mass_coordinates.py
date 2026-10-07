#!/usr/bin/env python3
from pathlib import Path
import json, csv, numpy as np, math
ROOT=Path(__file__).resolve().parents[1]
bench=json.loads((ROOT/'data/FINITE_FLAVOR_BENCHMARKS.json').read_text())
red=json.loads((ROOT/'results/REDUCED_LEPTON_INVARIANTS.json').read_text())
refs=json.loads((ROOT/'data/EXPERIMENTAL_REFERENCE_VALUES_2026.json').read_text())
u=np.array(bench['receiver_singular_values']['u'],float)
d=np.array(bench['receiver_singular_values']['d'],float)
nu=np.array(bench['neutral_bare_takagi_singular_values'],float)
# Weighted best common-scale projection of the finite neutral shape onto the two oscillation splittings.
y=np.array([refs['pmns_normal_ordering']['delta_m21_eV2'], refs['pmns_normal_ordering']['delta_m3l_eV2']],float)
sig=np.array([0.19e-5, 0.024e-3],float)
A=np.array([nu[1]**2-nu[0]**2,nu[2]**2-nu[0]**2])
w=1/sig**2
x=float(np.sum(w*A*y)/np.sum(w*A*A)); scale=math.sqrt(x)
nu_fit=scale*nu; split_fit=x*A; chi=((split_fit-y)/sig)**2
Benu=np.array(bench['B_enu'],float); eweights=Benu[0]
m_beta=math.sqrt(float(np.sum(eweights*nu_fit**2)))
terms=eweights*nu_fit; m_bb_max=float(np.sum(terms)); m_bb_min=float(max(2*np.max(terms)-np.sum(terms),0.0))
out={
 'classification':'MASS_COORDINATE_REGISTRY',
 'charged_lepton_reduced_coordinates':{
   'normalization':'electron reduced coordinate = 1',
   'electron':1.0,
   'muon':float(red['mu_over_e_reduced']),
   'tau':float(red['tau_over_e_reduced']),
   'mu_over_e':float(red['mu_over_e_reduced']),
   'tau_over_mu':float(red['tau_over_mu_reduced']),
   'tau_over_e':float(red['tau_over_e_reduced']),
   'koide_Q':float(red['koide_reduced'])},
 'quark_finite_receiver_coordinates':{
   'up_raw':u.tolist(),'up_normalized_to_heaviest':(u/u[-1]).tolist(),
   'down_raw':d.tolist(),'down_normalized_to_heaviest':(d/d[-1]).tolist(),
   'same_scale_down_ratio_d_over_s':float(d[0]/d[1]),
   'reference_same_scale_down_ratio_d_over_s':float((refs['quark_masses']['d']['value_MeV'])/(refs['quark_masses']['s']['value_MeV']))},
 'neutrino_finite_takagi_coordinates':{
   'raw':nu.tolist(),'normalized_to_heaviest':(nu/nu[-1]).tolist(),
   'dimensionless_split_ratio':float(A[0]/A[1]),
   'weighted_common_scale_projection_eV_per_coordinate':scale,
   'weighted_projected_masses_eV':nu_fit.tolist(),
   'weighted_projected_splittings_eV2':{'delta_m21':float(split_fit[0]),'delta_m31':float(split_fit[1])},
   'weighted_projected_sum_masses_eV':float(np.sum(nu_fit)),
   'weighted_projected_m_beta_eV':float(m_beta),
   'weighted_projected_m_bb_phase_interval_eV':[m_bb_min,m_bb_max],
   'weighted_split_chi2_contributions':chi.tolist(),'weighted_split_chi2_total':float(np.sum(chi))},
 'physical_conversion_laws':{
   'charged_stable_or_threshold_channel':'m_f=(hbar/(c^2 tau_*))*epsilon_f',
   'quarks':'mbar_f(mu) from the scalar chirality-changing part of the common color-resolved renormalized inverse propagator at declared mu',
   'neutrinos':'m_nu_i=(hbar/(c^2 tau_*))*sigma_i(M_N), Delta m_ij^2=(hbar/(c^2 tau_*))^2*(sigma_i^2-sigma_j^2)'},
 'interpretation':'The table collects the mass coordinates numerically determined by the distributed finite data. Source-space and reduced coordinates are reported together with the common interacting conversion laws that define dimensional observables.'
}
(ROOT/'results/COMPLETE_MASS_COORDINATES.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
rows=[]
rows += [
 {'sector':'charged_lepton_reduced','flavor':'e','coordinate':1.0,'normalization':'m_e^red=1','physical_map':'common G2 spectral coordinate + tau_*'},
 {'sector':'charged_lepton_reduced','flavor':'mu','coordinate':float(red['mu_over_e_reduced']),'normalization':'m_e^red=1','physical_map':'common G2 spectral coordinate + tau_*'},
 {'sector':'charged_lepton_reduced','flavor':'tau','coordinate':float(red['tau_over_e_reduced']),'normalization':'m_e^red=1','physical_map':'common G2 spectral coordinate + tau_*'}]
for name,val in zip(['u','c','t'],u/u[-1]): rows.append({'sector':'up_quark_finite','flavor':name,'coordinate':float(val),'normalization':'heaviest finite coordinate=1','physical_map':'common color-resolved renormalized kernel'})
for name,val in zip(['d','s','b'],d/d[-1]): rows.append({'sector':'down_quark_finite','flavor':name,'coordinate':float(val),'normalization':'heaviest finite coordinate=1','physical_map':'common color-resolved renormalized kernel'})
for name,val in zip(['nu1','nu2','nu3'],nu/nu[-1]): rows.append({'sector':'neutrino_finite_takagi','flavor':name,'coordinate':float(val),'normalization':'heaviest finite Takagi coordinate=1','physical_map':'common interacting neutral Nambu Takagi kernel + tau_*'})
with open(ROOT/'results/COMPLETE_MASS_COORDINATES.csv','w',newline='') as f:
 wtr=csv.DictWriter(f,fieldnames=rows[0].keys()); wtr.writeheader(); wtr.writerows(rows)
print(json.dumps(out,indent=2,sort_keys=True))
