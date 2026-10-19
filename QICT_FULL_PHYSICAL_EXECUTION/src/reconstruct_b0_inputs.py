#!/usr/bin/env python3
"""Reconstruct verified B0 factorized Kato diagnostics from authentic Git data.
This does NOT produce the missing full lambda=1 KATO_BUFFER_0..3 states.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ACTION_SHA="c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7"
EVIDENCE=ROOT/"data/physical_g2_inputs/QICT_REQUESTED_C9_G2_PHYSICAL_ELEMENTS_2026-10-06/03_REQUESTED_ELEMENTS/KATO_BUFFER"
SETS=[
("J31","FACTORIZED_B0_KATO_RIESZ_BUFFER.npz","FACTORIZED_B0_KATO_RIESZ_BUFFER_CERT.json","psi_size{}_E16_J31"),
("J60","FACTORIZED_B0_KATO_RIESZ_BUFFER_J60.npz","FACTORIZED_B0_KATO_RIESZ_BUFFER_J60_CERT.json","psi_size{}_E16_J60"),
("J80","FACTORIZED_B0_KATO_RIESZ_BUFFER_J80_CLUSTER.npz","FACTORIZED_B0_KATO_RIESZ_BUFFER_J80_CLUSTER_CERT.json","psi_size{}_E16_J80"),
]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def reconstruct(out, evidence=EVIDENCE):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    action=ROOT/"QICT_FULL_PHYSICAL_EXECUTION/ACTION/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json"
    if digest(action)!=ACTION_SHA: raise ValueError("Frozen canonical action SHA-256 mismatch")
    entries=[]
    for cutoff,name,cert_name,fmt in SETS:
        path=evidence/name; cert=json.loads((evidence/cert_name).read_text())
        if digest(path)!=cert["npz_sha256"]:raise ValueError("Certified source mismatch: "+name)
        with np.load(path,allow_pickle=False) as z:
            for size,record in zip((1,4),cert["components"]):
                psi=np.asarray(z[fmt.format(size)],dtype=np.complex128)
                norm=float(np.vdot(psi,psi).real)
                gap=float(record["certified_contour_sigma_min_lower_bound_diagonalizable"])
                resid=float(record["selected_product_residual"])
                if not(np.isfinite(psi).all() and abs(norm-1)<1e-11 and
                       len(psi)==int(record["product_dimension"]) and gap>0 and
                       0<=resid<1e-9):
                    raise ValueError("Unsupported B0 state or invalid certificate: "+fmt.format(size))
                F=(psi/np.sqrt(norm))[:,None]
                name_out=f"B0_FACTORIZED_{cutoff}_SIZE{size}_DIAGNOSTIC.npz"
                np.savez_compressed(out/name_out,rho_gram_factor=F,buffer=np.array(0),
                    original_action_sha256=np.array(ACTION_SHA),
                    contour_gap_bound_from_original_certificate=np.array(gap),
                    selected_eigenvector_residual=np.array(resid),
                    classification=np.array("B0_FACTORIZED_DIAGNOSTIC_ONLY"))
                entries.append({"file":name_out,"sha256":digest(out/name_out),
                    "cutoff":cutoff,"component_size":size,"factor_dim":len(psi),
                    "input_sha256":digest(path),"status":"DIAGNOSTIC_ONLY",
                    "is_full_lambda1_interacting_state":False})
    report={"status":"BLOCKED_MISSING_INPUT","action_sha256":ACTION_SHA,
        "original_action_verified":True,"factorized_b0_artifacts":entries,
        "physical_KATO_BUFFER_0_to_3_created":False,
        "reason":"The original factorized b=0 data are not four nested lambda=1 interacting physical buffers.",
        "mass_GeV":None}
    (out/"SOURCE_BACKED_B0_RECONSTRUCTION.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--evidence",type=Path,default=EVIDENCE)
    a=p.parse_args()
    result=reconstruct(a.out,a.evidence)
    print(json.dumps({"status":result["status"],"b0_diagnostics":len(result["factorized_b0_artifacts"])},indent=2))
