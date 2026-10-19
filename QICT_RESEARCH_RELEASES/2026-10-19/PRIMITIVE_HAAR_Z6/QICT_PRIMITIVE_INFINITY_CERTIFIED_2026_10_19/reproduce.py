#!/usr/bin/env python3
"""Execute the published finite mathematical and numerical checks."""
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SCRIPTS=(
    'mathematical_checks.py',
    'primitive_infinite_volume_certificate.py',
    'central_character_rule.py',
    'primitive_pole_triple_method.py',
    'primitive_splitstep_diagnostic.py',
    'primitive_floquet_shift_no_go.py',
    'electric_phase_obstruction.py',
    'quantitative_stability.py',
    'verify_finite_evidence.py',
    'contract_inspection.py',
    'independent_analytical_verification.py',
    'generate_status.py',
)

def main():
    env=dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE']='1'
    for name in SCRIPTS:
        subprocess.run([sys.executable,str(ROOT/'src'/name)],cwd=ROOT,env=env,check=True,
                       stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                   cwd=ROOT,env=env,check=True)
    print('PASS: finite mathematical calculations, source provenance and unit tests')
    print('PHYSICAL: BLOCKED_MISSING_INPUT (full interacting gate and physical state not supplied)')

if __name__=='__main__':main()
