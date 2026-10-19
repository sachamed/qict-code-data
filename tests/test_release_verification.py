"""Positive and negative controls for original scientific release preservation."""
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_scientific_release import audit


def sha_for(content):
    return (hashlib.sha256(content).hexdigest(),
            hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest())


class ReleaseVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.release = self.root / 'QICT_RESEARCH_RELEASES/2026-10-19'
        folder = self.release / 'DARK_SECTOR'
        folder.mkdir(parents=True)
        self.file = folder / 'test.txt'
        self.file.write_bytes(b'exact original payload')
        sha, blob = sha_for(self.file.read_bytes())
        self.manifest = {
            'schema': 'qict-github-release-integrity-v1', 'total_files': 1,
            'original_archives': {
                'PRIMITIVE_HAAR_Z6': {'file_count': 0},
                'DARK_SECTOR': {'file_count': 1}},
            'files': [{'path': 'DARK_SECTOR/test.txt', 'size_bytes': self.file.stat().st_size,
                       'sha256': sha, 'git_blob_sha': blob}],
        }
        self.write_manifest()

    def write_manifest(self):
        (self.release / 'UPLOAD_VERIFICATION_MANIFEST.json').write_text(json.dumps(self.manifest))

    def test_original_bytes_pass(self):
        self.assertEqual('PASS', audit(self.root)['status'])

    def test_tampering_rejected(self):
        self.file.write_bytes(b'exact original payload?')
        self.assertEqual('FAIL', audit(self.root)['status'])

    def test_added_payload_rejected(self):
        (self.release / 'DARK_SECTOR/unknown.py').write_text('pass')
        self.assertEqual('FAIL', audit(self.root)['status'])

    def test_generated_python_bytecode_is_ignored(self):
        pycache = self.release / 'DARK_SECTOR/__pycache__'
        pycache.mkdir()
        (pycache / 'module.cpython-313.pyc').write_bytes(b'generated')
        self.assertEqual('PASS', audit(self.root)['status'])

    def test_missing_file_rejected(self):
        self.file.unlink()
        self.assertEqual('FAIL', audit(self.root)['status'])

    def test_duplicate_entry_rejected(self):
        self.manifest['files'].append(self.manifest['files'][0].copy())
        self.manifest['total_files'] = 2
        self.write_manifest()
        self.assertEqual('FAIL', audit(self.root)['status'])

    def test_parent_traversal_rejected(self):
        self.manifest['files'][0]['path'] = 'DARK_SECTOR/../README.md'
        self.write_manifest()
        self.assertEqual('FAIL', audit(self.root)['status'])

    def test_symlink_rejected(self):
        target = self.root / 'external'
        target.write_bytes(b'exact original payload')
        self.file.unlink()
        self.file.symlink_to(target)
        self.assertEqual('FAIL', audit(self.root)['status'])

    def test_invalid_hash_rejected(self):
        self.manifest['files'][0]['sha256'] = 'z' * 64
        self.write_manifest()
        self.assertEqual('FAIL', audit(self.root)['status'])


if __name__ == '__main__':
    unittest.main()