#!/usr/bin/env python3
"""Independent recomputation of finite primitive and static GNS evidence.

The reported metrics are floating-point diagnostics; the original rational
bounds remain statements of the separately provided analytic certificates.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
RESULTS = ROOT / 'results'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def primitive_checks():
    cert=json.loads((EVIDENCE/'PRIMITIVE_A7_GROUND_CERTIFICATE.json').read_text())
    rows=[]
    if cert.get('physical_full_interacting_Floquet_Kato') is not False:
        raise ValueError('Invalid scope in primitive Kato certificate')
    for x in cert['cutoffs']:
        source=EVIDENCE/x['file']
        if digest(source)!=x['sha256']:
            raise ValueError('Primitive source hash mismatch: '+x['file'])
        with np.load(source,allow_pickle=False) as d:
            flux=np.asarray(d['flux'],int)
            F=np.asarray(d['rho_gram_factor'],complex)
            rho=np.asarray(d['rho'],complex)
            if F.shape!=(len(flux),1):
                raise ValueError('Primitive source factor dimensions invalid')
            psi=F[:,0]
            gram=psi[:,None]*psi[None,:].conj()
            r=flux.astype(float)
            diag=8*r*r+4
            H=np.diag(diag).astype(complex)
            H+=np.diag(np.full(len(r)-1,-2.),1)
            H+=np.diag(np.full(len(r)-1,-2.),-1)
            mu=float((psi.conj()@H@psi).real)
            # The full operator adds two exterior basis components -2 psi(+-N).
            residual=float(np.sqrt(np.linalg.norm(H@psi-mu*psi)**2+
                          4*(abs(psi[0])**2+abs(psi[-1])**2)))
            row={'cutoff':int(x['N']),'source_sha256':x['sha256'],
                 'trace_abs_error':float(abs(np.trace(rho)-1)),
                 'gram_factor_fro_error':float(np.linalg.norm(rho-gram)),
                 'gram_psd_by_construction':True,
                 'Rayleigh_quotient':mu,
                 'full_Hamiltonian_residual_float64':residual,
                 'endpoint_gap_minmax_lower':float(8-mu),
                 'classification':'FINITE_PRIMITIVE_SINGLE_PLAQUETTE'}
            if not (row['trace_abs_error']<1e-12 and row['gram_factor_fro_error']<1e-12
                    and residual<1e-10 and 8-mu>4):
                raise ValueError('Primitive source numerical screening failed')
            rows.append(row)
    return rows


def static_gns_checks():
    cert=json.loads((EVIDENCE/'PRIMITIVE_STATIC_GNS_AUDIT.json').read_text())
    if cert.get('full_interacting_C10_executed') is not False or cert.get('physical_charged_CN6_executed') is not False:
        raise ValueError('Invalid scope of static GNS evidence')
    rows=[]
    for entry in cert['results']:
        source=EVIDENCE/entry['file']
        if digest(source)!=entry['sha256']:
            raise ValueError('GNS source hash mismatch: '+entry['file'])
        with np.load(source,allow_pickle=False) as d:
            C=np.asarray(d['moments'],complex)
            n=len(C)
            if C.shape!=(n,6,6):raise ValueError('GNS six-source matrix dimensions invalid')
            T=np.zeros((6*n,6*n),complex)
            for i in range(n):
                for j in range(n):
                    k=i-j
                    T[6*i:6*(i+1),6*j:6*(j+1)]=C[k] if k>=0 else C[-k].conj().T
            mn=float(np.linalg.eigvalsh((T+T.conj().T)/2)[0])
            err=float(np.linalg.norm(C[0]-np.eye(6)))
            if err>1e-12 or mn<-1e-12: raise ValueError('GNS C0 or Toeplitz numerical screening failed')
            rows.append({'source':entry['file'],'source_sha256':entry['sha256'],
                'cutoff':int(entry['cutoff']),'external_charge':int(entry['q_static_external']),
                'moment_depth':n-1,'C0_frobenius_error_float64':err,
                'toeplitz_smallest_eigenvalue_float64':mn,
                'toeplitz_certified_positive':False,
                'classification':'STATIC_EXTERNAL_CHARGE_GNS_ONLY'})
    return rows


def run():
    row1=primitive_checks();row2=static_gns_checks()
    result={'scope':'INDEPENDENT_FINITE_DATA_RECALCULATION',
            'source_action_sha256':'c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7',
            'cutoff_states':row1, 'static_charge_GNS':row2,
            'full_interacting_QICT_provider_validated':False,
            'physical_dark_mass_certified':False,
            'numerical_method':'complex128 dense small-block verification; analytic proofs stated separately'}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'FINITE_EVIDENCE_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result

if __name__=='__main__':
    report=run()
    print('FINITE_SOURCE_VERIFICATION_PASS:',len(report['cutoff_states']),'primitive cutoffs and',len(report['static_charge_GNS']),'static GNS cases')
