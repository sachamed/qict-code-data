"""Reject metadata-only providers and synthetic receipts."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from provider_contract import ACTION_SHA256,validate_provider

class Fake:
    def __init__(self,path): self.__file__=str(path)
    def metadata(self):
        return dict(physical_payload=True,action_sha256=ACTION_SHA256,L=24,PiCAR=1,
                    source_order=["L1","L2","L3","R1","R2","R3"],charged_U_cone_complete=True,
                    lambda1_riesz_kato_selected=True,KATO_BUFFER_count=4,
                    buffer_tail_certified=True,C10_shell_CP_maps_present=True,
                    boundary_to_path_present=True,rank6_source_carrier_certified=True,
                    operator_or_state_error_bound=0.0)
    def dimension(self): return 6
    def matmat(self,x): return x
    def source_matrix(self): return None

class ProviderTests(unittest.TestCase):
    def test_metadata_only_not_physical(self):
        with tempfile.TemporaryDirectory() as temp:
            source=pathlib.Path(temp)/"provider.py";source.write_text("pass\n")
            with self.assertRaisesRegex(ValueError,"independent provider certificate missing"):
                validate_provider(Fake(source),24)
    def test_synthetic_receipt_not_physical(self):
        with tempfile.TemporaryDirectory() as temp:
            source=pathlib.Path(temp)/"provider.py";source.write_text("pass\n")
            path=source.with_name(source.name+".independent.json")
            path.write_text(json.dumps({"status":"PASS","action_sha256":ACTION_SHA256,
                "synthetic_data_used":True,"independent_validation":True,
                "evidence_hashes":{"file":"fake"}}))
            with self.assertRaisesRegex(ValueError,"not independently certified"):
                validate_provider(Fake(source),24)

if __name__=="__main__":unittest.main()
