"""One-command regeneration of every figure, numerical table and model result."""
from pathlib import Path
import argparse, os, subprocess, sys

ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--calculations-only',action='store_true',help='Skip the PDF build; TeX is then optional.')
    args=ap.parse_args()
    commands=[['compute_cosmology.py'],['compute_closure_tests.py'],['compute_operator_closure.py'],
              ['certify_moment_bounds.py'],['compute_portal_thermal.py'],['compute_response_extensions.py'],
              ['structure_metric_and_galaxies.py'],['run_pm_convergence.py'],
              ['validate_results.py'],['validate_closure.py'],['test_operator_closure.py'],
              ['validate_extended.py'],['convergence_checks.py'],['validate_scientific_domains.py'],
              ['derive_extended_results.py'],['reconstruct_c9_moments.py'],
              ['certify_operator_spectra.py'],['validate_operator_certificate.py'],
              ['verify_physical_identities.py'],
              ['generate_tables.py']]
    if not args.calculations_only:commands.append(['build_manuscript.py'])
    for command in commands:
        print('RUN',*command,flush=True)
        subprocess.run([sys.executable,str(ROOT/command[0]),*command[1:]],cwd=ROOT,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    print('Complete: stated finite/effective-model results, figures and tables regenerated.')

if __name__=='__main__':main()
