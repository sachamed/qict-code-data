#!/usr/bin/env python3
"""Reproduce the QICT calculations from the distributed scientific inputs."""
from pathlib import Path
import sys,os,argparse,subprocess,json
sys.path.insert(0,str(Path(__file__).resolve().parent))

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'reproducibility/code'
sys.path.insert(0,str(CODE))
from prepare_inputs import prepare as prepare_inputs
from bootstrap import prepare as bootstrap
def prepare():
    return {**bootstrap(), **prepare_inputs()}

GROUPS={
 'analytic_revision':['verify_revision2_analytic_kernels.py'],
 'quantitative':['certify_tensor_massless_weight.py','certify_reduced_mass_and_tensor_bcc.py','quantitative_comparison.py','test_reduced_hierarchy.py','check_quark_cost_identification.py','compute_quark_cost_identification.py'],
 'baseline':['run_all.py'],
 'coherent':[
  'certify_flux_kato_obstruction.py','compute_common_primitive_kato.py',
  'certify_c10_primitive_gauss.py','nonabelian_common_car_kato.py','coherent_boundary_provider.py',
  'compute_global_primitive_kato_seed.py','compute_coupled_plaquette_boundary.py',
  'certify_chiral_cone_domain.py','certify_spatial_spectator_state.py',
  'certify_exact_resonance_reduction.py','compute_weyl_bloch_moments.py',
  'compute_common_cyclic_car_kato.py','certify_coherent_charge_decoupling.py',
  'solve_global_modular_charge_reduction.py','compute_coupled_su2_spin_network.py',
  'compute_charged_su2_theta.py','certify_global_primitive_heat_kernel_gap.py',
  'certify_physical_selection_compatibility.py','compute_common_coupled_electroweak_gns.py'],
 'state':[
  'compute_physical_two_time_kernel.py','compute_finite_C9_common_state.py',
  'verify_formal_closure_framework.py','verify_state_spectral_criteria.py'],
 'static_sources':['complete_static_wilson_sources.py','certify_static_kernel_arithmetic.py','compute_coherent_source_memory.py'],
 'independent':[
  'audit_common_carrier.py','verify_nonabelian_common_car.py','verify_global_boundary_results.py',
  'verify_exact_charge_reductions.py','verify_coupled_su2_vertices.py',
  'verify_global_and_electroweak_closure.py','verify_physical_state_and_framework.py',
  'verify_published_crosschecks.py','verify_static_wilson_sources.py','verify_coherent_source_memory.py',
  'verify_scaling_and_orientable_geometry.py','verify_continuum_source_controls.py', 'test_structural_corrections.py', 'audit_quantitative_stability.py','verify_arc_and_nested_audit.py'],
 'figures':[
  'make_common_carrier_figure.py','make_flux_kato_figure.py','make_nonabelian_car_figures.py',
  'make_global_boundary_figures.py','make_coupled_electroweak_figure.py','make_physical_state_figure.py','make_wilson_occupation_figure.py','regenerate_electric_reference_figure.py','regenerate_reference_figures.py'],
}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--prepare',action='store_true');p.add_argument('--verify',action='store_true')
 p.add_argument('--build',action='store_true')
 p.add_argument('--list',action='store_true');p.add_argument('--group',choices=[*GROUPS,'all'])
 a=p.parse_args()
 if a.list:print(json.dumps(GROUPS,indent=2));return
 if a.prepare or a.group:print(json.dumps(prepare(),indent=2))
 env=os.environ.copy();env.setdefault('OPENBLAS_NUM_THREADS','1');env['PYTHONDONTWRITEBYTECODE']='1'
 if a.verify or not any([a.prepare,a.group,a.build]):
  subprocess.run([sys.executable,str(ROOT/'validate_bundle.py')],cwd=ROOT,env=env,check=True)
 if a.group:
  groups=list(GROUPS) if a.group=='all' else [a.group]
  for group in groups:
   for script in GROUPS[group]:subprocess.run([sys.executable,str(CODE/script)],cwd=ROOT,env=env,check=True)
 if a.build:
  command=[sys.executable,str(ROOT/'build_documents.py')]
  # Refresh the three journal-neutral documents and their cross-references.
  subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
