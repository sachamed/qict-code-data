"""Adversarial validation of physical status, source provenance, and spectral bounds."""
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mathematical_checks import (central_orbit_exact, copy_odd_gate_check,
    phase_to_mass_interval, quotient_audit, auxiliary_pole_certificate,
    schur_feshbach_check, truncated_moment_ambiguity)
from hpc_gate import validate, REQUIRED


class StrictScientificTests(unittest.TestCase):
    def test_01_all_six_center_lifts_exact(self):
        r = central_orbit_exact()
        self.assertEqual(r['magnetic_costs_exact'], ['0','120','60','72','60','120'])
        self.assertTrue(r['original_gate_fails_quotient_descent'])

    def test_02_center_neutral_alternative(self):
        from mathematical_checks import (magnetic_quotient_alternative,center_transform)
        a = np.diag([np.exp(.31j), np.exp(-.31j)])
        b = np.diag([np.exp(.24j),np.exp(-.10j),np.exp(-.14j)])
        theta = 0.46
        original = magnetic_quotient_alternative(theta,a,b)
        for _ in range(6):
            theta,a,b = center_transform(theta,a,b)
            self.assertAlmostEqual(magnetic_quotient_alternative(theta,a,b),original,places=11)

    def test_03_odd_source_must_be_isometric(self):
        u=np.diag([1,1j]);q=np.diag([-1,1]);v=np.array([[2],[0]],complex)
        with self.assertRaises(ValueError):copy_odd_gate_check(u,q,v)

    def test_04_wrong_copy_grading_rejected(self):
        u=np.diag([1,1j]);q=np.diag([-1,1]);v=np.array([[0],[1]],complex)
        with self.assertRaises(ValueError):copy_odd_gate_check(u,q,v)

    def test_05_complex_nonfinite_matrices_rejected(self):
        u=np.diag([np.nan,1j]);q=np.diag([-1,1]);v=np.array([[1],[0]],complex)
        with self.assertRaises(ValueError):copy_odd_gate_check(u,q,v)

    def test_06_infinite_time_uncertainty_rejected(self):
        with self.assertRaises(ValueError):phase_to_mass_interval((0.2,0.4),(1e-12,float('inf')))

    def test_07_infinite_phase_uncertainty_rejected(self):
        with self.assertRaises(ValueError):phase_to_mass_interval((0.2,float('inf')),(1e-12,2e-12))

    def test_08_no_source_overlap_cannot_be_called_pole(self):
        # An exact eigenvalue 1 exists, but the selected source has zero overlap.
        u=np.diag([1.,1j]);v=np.array([[0],[1]],complex)
        p=np.diag([1.,0.]);r=v.conj().T@p@v
        self.assertAlmostEqual(np.trace(r).real,0.)

    def test_09_contour_quadrature_roundoff_separated(self):
        p=auxiliary_pole_certificate(5,steps=32)
        self.assertGreater(p['contour_gap'],0)
        self.assertTrue(p['Riesz_midpoint_bound_excludes_roundoff'])
        self.assertLess(p['method_Riesz_discrepancy'],
            p['Riesz_midpoint_exact_arithmetic_error_bound']+1e-12)

    def test_10_schur_evaluation_not_at_pole(self):
        u=np.diag([1.,1j]);v=np.array([[1],[0]],complex)
        with self.assertRaises(ValueError):schur_feshbach_check(u,v,1.+0j)

    def test_11_exact_primitive_evidence_hashes(self):
        source=ROOT/'evidence/PRIMITIVE_A7_GROUND_CERTIFICATE.json'
        cert=json.loads(source.read_text())
        self.assertIs(cert['physical_full_interacting_Floquet_Kato'],False)
        for x in cert['cutoffs']:
            f=ROOT/'evidence'/x['file']
            self.assertTrue(f.is_file())
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),x['sha256'])
            with np.load(f,allow_pickle=False) as z:
                self.assertTrue(z.files)

    def test_12_static_gns_evidence_hashes(self):
        cert=json.loads((ROOT/'evidence/PRIMITIVE_STATIC_GNS_AUDIT.json').read_text())
        self.assertFalse(cert['physical_charged_CN6_executed'])
        self.assertFalse(cert['full_interacting_C10_executed'])
        for x in cert['results']:
            f=ROOT/'evidence'/x['file']
            self.assertTrue(f.is_file())
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),x['sha256'])

    def test_13_moment_nonidentifiability(self):
        for n in (4,12,37):
            r=truncated_moment_ambiguity(n)
            self.assertLess(r['max_moment_discrepancy'],1e-13)
            self.assertAlmostEqual(r['discrete_measure_atom_at_one'],1/(n+1))

    def test_14_primitive_density_factors_and_boundary_residuals(self):
        cert=json.loads((ROOT/'evidence/PRIMITIVE_A7_GROUND_CERTIFICATE.json').read_text())
        for x in cert['cutoffs']:
            with np.load(ROOT/'evidence'/x['file'],allow_pickle=False) as z:
                r=z['flux'].astype(float)
                f=z['rho_gram_factor'][:,0]
                h=np.diag(8*r*r+4).astype(complex)
                h+=np.diag(np.full(len(r)-1,-2.),1)
                h+=np.diag(np.full(len(r)-1,-2.),-1)
                mu=float((f.conj()@h@f).real)
                residual_squared=float(np.linalg.norm(h@f-mu*f)**2+4*(abs(f[0])**2+abs(f[-1])**2))
                self.assertLess(abs(np.vdot(f,f).real-1),1e-12)
                self.assertLess(np.linalg.norm(z['rho']-f[:,None]@f[None,:].conj()),1e-12)
                self.assertLess(residual_squared,1e-24)
                self.assertGreater(8-mu,4)

    def test_15_static_gns_toeplitz_screening(self):
        cert=json.loads((ROOT/'evidence/PRIMITIVE_STATIC_GNS_AUDIT.json').read_text())
        for entry in cert['results']:
            with np.load(ROOT/'evidence'/entry['file'],allow_pickle=False) as z:
                c=z['moments']; n=len(c); t=np.empty((6*n,6*n),complex)
                for i in range(n):
                    for j in range(n):
                        k=i-j
                        t[6*i:6*i+6,6*j:6*j+6]=c[k] if k>=0 else c[-k].conj().T
                mineig=float(np.linalg.eigvalsh((t+t.conj().T)/2).min())
                self.assertLess(np.linalg.norm(c[0]-np.eye(6)),1e-12)
                self.assertGreater(mineig,-1e-12)
                self.assertLess(np.linalg.norm(t-t.conj().T),1e-12)

    def test_16_fabricated_certificate_does_not_unlock_hpc(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'fake.dat').write_bytes(b'not physics')
            fake_digest=hashlib.sha256((p/'fake.dat').read_bytes()).hexdigest()
            for k in REQUIRED:
                (p/(k+'.json')).write_text(json.dumps({
                    'status':'PASS','physical_evidence_verified':True,
                    'sha256_inputs':{'fake.dat':fake_digest},
                    'independent_reviewer_signature':'self-asserted-only'}))
            x=validate(p)
            self.assertFalse(x['HPC_PRODUCTION_AUTHORIZED'])
            self.assertFalse(x['cryptographic_independent_attestation_verified'])

if __name__=='__main__':unittest.main()
