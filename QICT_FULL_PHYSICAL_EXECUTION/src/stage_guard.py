#!/usr/bin/env python3
"""Strict preflight guard. Never promotes synthetic controls or partial files to physics."""
import argparse, datetime, hashlib, json, os
from pathlib import Path

BASE=Path(__file__).resolve().parent.parent
ACTION_SHA="c22ff80fcb94204984a65debf7e5636ef1d4e6caad89cf8fd6934dfdb8f684b7"
REQUIRED=["KATO_BUFFER_0.npz","KATO_BUFFER_1.npz","KATO_BUFFER_2.npz","KATO_BUFFER_3.npz","BUFFER_CONTRACTION_CERTIFICATE.json","C10_SHELL_MAPS_0.npz","C10_SHELL_MAPS_1.npz","C10_SHELL_MAPS_2.npz","C10_SHELL_MAPS_3.npz","BOUNDARY_TO_PATH_0.npz","BOUNDARY_TO_PATH_1.npz","BOUNDARY_TO_PATH_2.npz","BOUNDARY_TO_PATH_3.npz","CORE_RECOUPLING_L24.npz","RANK6_EMBEDDING_L24.npz","GLOBAL_CHARGED_UCONE_PROVIDER.py","GLOBAL_CHARGED_UCONE_PASS_CERTIFICATE.json","CN6_MOMENTS.npz","CN6_SHAPE_AND_C0_CERTIFICATE.json"]
STAGES={"kato":5,"c10":9,"boundary":13,"core":14,"rank6":15,"ucone":17,"cn6":19,"pole":19,"multivolume":19,"mass":19}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda: stream.read(1 << 20),b""): h.update(block)
    return h.hexdigest()

def guard(stage, root=BASE):
    root=Path(root).resolve()
    reasons=[]
    action=root/"ACTION/QICT_LATTICE_ACTION_FREEZE_V3.canonical.json"
    if not action.is_file() or action.is_symlink() or sha256(action)!=ACTION_SHA:
        reasons.append("The certified canonical action is missing, symlinked or modified")
    frozen=root/"ACTION/frozen_action.json"
    try:
        metadata=json.loads(frozen.read_text())
        if metadata.get("author_git_commit") is None:
            reasons.append("No author-approved immutable Git commit")
        if metadata.get("full_ucone_operator_hash") is None:
            reasons.append("No certified complete interacting operator hash")
    except (OSError,ValueError):
        reasons.append("Missing or invalid frozen action metadata")
    for filename in REQUIRED[:STAGES[stage]]:
        path=root/"input/verified"/filename
        cert=root/"certificates"/(filename+".independent.json")
        if not path.is_file() or path.is_symlink():
            reasons.append("Missing verified physical input: "+filename)
            continue
        try:
            statement=json.loads(cert.read_text())
            if not (statement.get("status")=="PASS" and
                statement.get("sha256")==sha256(path) and
                statement.get("action_sha256")==ACTION_SHA and
                statement.get("synthetic_data_used") is False):
                reasons.append("Failed independent certificate: "+filename)
        except (OSError,ValueError):
            reasons.append("Missing valid independent certificate: "+filename)
    for additional in ("author_signed_pin.json","end_to_end_operator_validation.json","physics_provenance.json"):
        if not (root/"certificates"/additional).is_file():
            reasons.append("Independent sign-off missing: "+additional)
    result={
        "stage":stage, "status":"BLOCKED_MISSING_INPUT" if reasons else "DIAGNOSTIC_ONLY",
        "accepted_for_physical_execution":False,
        "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reasons":reasons or ["Independent scientific approval and physical validation remain required"]
    }
    output=root/"results/FAIL_CLOSED_AUDIT.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,output)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=STAGES)
    parser.add_argument("--root",default=str(BASE))
    args=parser.parse_args()
    status=guard(args.stage,args.root)
    print(json.dumps(status,indent=2))
    raise SystemExit(2 if not status["accepted_for_physical_execution"] else 0)
