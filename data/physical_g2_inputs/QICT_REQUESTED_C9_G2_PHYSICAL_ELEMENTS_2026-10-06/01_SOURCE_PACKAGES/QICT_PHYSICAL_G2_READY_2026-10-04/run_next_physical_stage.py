#!/usr/bin/env python3
"""Fail-closed launcher for the next QICT physical G2 stage.
It never fabricates missing L24 dynamics. It validates the local certified cluster,
then checks for the external physical tensors required by the submitted provider API.
"""
from pathlib import Path
import argparse,glob,json,subprocess,sys

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent); ap.add_argument('--physical-dir',type=Path,required=True); a=ap.parse_args()
 subprocess.run([sys.executable,str(a.bundle/'verify_step4_strict.py'),'--root',str(a.bundle)],check=True)
 p=a.physical_dir
 buffers=sorted(p.glob('KATO_BUFFER_*.npz'))
 shell=sorted(p.glob('C10_SHELL_MAPS_*.npz'))
 b2p=sorted(p.glob('BOUNDARY_TO_PATH_*.npz'))
 required=[p/'BUFFER_CONTRACTION_CERTIFICATE.json',p/'CORE_RECOUPLING_L24.npz',p/'RANK6_EMBEDDING_L24.npz']
 missing=[]
 if len(buffers)<4: missing.append('>=4 KATO_BUFFER_<b>.npz')
 if not shell: missing.append('C10_SHELL_MAPS_<b>.npz')
 if not b2p: missing.append('BOUNDARY_TO_PATH_<b>.npz')
 missing += [x.name for x in required if not x.exists()]
 # A complete global charged U_cone provider/operator must also be explicitly supplied.
 if not any((p/n).exists() for n in ['GLOBAL_CHARGED_UCONE_L24.npz','GLOBAL_CHARGED_UCONE_PROVIDER.py']):
  missing.append('GLOBAL_CHARGED_UCONE_L24.npz or GLOBAL_CHARGED_UCONE_PROVIDER.py')
 out={'status':'READY_FOR_PHYSICAL_RIESZ_C10_CN6' if not missing else 'BLOCKED_MISSING_PHYSICAL_INPUTS','physical_dir':str(p),'kato_buffer_count':len(buffers),'shell_map_count':len(shell),'boundary_to_path_count':len(b2p),'missing':missing,'note':'Step-4 local charged cluster is verified but is not a substitute for the selected global Kato/Riesz state.'}
 print(json.dumps(out,indent=2,sort_keys=True))
 if missing: sys.exit(3)
if __name__=='__main__': main()
