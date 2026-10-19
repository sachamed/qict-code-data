#!/usr/bin/env python3
"""Fail-closed HPC production gate. File signatures/physics require human independent approval."""
from pathlib import Path
import argparse, hashlib, json, sys
REQUIRED=('T28_43_ENDPOINT_STATE_SELECTION','FULL_INTERACTING_GATES','Z6_ACTION_CONVENTION',
 'FOUR_PHYSICAL_KATO_STATES','PHYSICAL_C10_MAPS','SIX_PHYSICAL_CHIRAL_SOURCES',
 'PHYSICAL_CN6_MOMENTS','LOCALIZED_RESIDUE_UNIFORM_LOWER_BOUND','PHYSICAL_POLE_THREE_METHODS',
 'PHYSICAL_ODD_SECTOR','INDEPENDENT_TAU_CLOCK','INDEPENDENT_BLIND_REPLICATION')

def validate(directory):
    directory=Path(directory)
    errors=[]
    for key in REQUIRED:
        path=directory/(key+'.json')
        if not path.is_file():
            errors.append('missing '+path.name);continue
        try:
            obj=json.loads(path.read_text())
            if obj.get('status')!='PASS' or obj.get('physical_evidence_verified') is not True:
                errors.append('uncertified '+path.name);continue
            if not isinstance(obj.get('sha256_inputs'),dict) or not obj['sha256_inputs']:
                errors.append('no evidence hashes '+path.name);continue
            for relative,sha in obj['sha256_inputs'].items():
                full=(directory/relative).resolve()
                if not full.is_relative_to(directory.resolve()) or not full.is_file() or hashlib.sha256(full.read_bytes()).hexdigest()!=sha:
                    errors.append('hash mismatch '+path.name+':'+relative)
            if not obj.get('independent_reviewer_signature'):
                errors.append('no reviewer identifier '+path.name)
            # A string identifier is not a cryptographically verified signature.
        except Exception:
            errors.append('malformed '+path.name)
    return {'status':'BLOCKED_MISSING_INPUT' if errors else 'DIAGNOSTIC_ONLY',
        'cryptographic_independent_attestation_verified':False,
        'HPC_PRODUCTION_AUTHORIZED':False,
        'reason':'Automated checks are not a proof of physical correctness; a manual institutional review must authorize production.',
        'errors':errors}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--certificates',type=Path,required=True)
    args=p.parse_args();report=validate(args.certificates)
    print(json.dumps(report,indent=2))
    raise SystemExit(2)
