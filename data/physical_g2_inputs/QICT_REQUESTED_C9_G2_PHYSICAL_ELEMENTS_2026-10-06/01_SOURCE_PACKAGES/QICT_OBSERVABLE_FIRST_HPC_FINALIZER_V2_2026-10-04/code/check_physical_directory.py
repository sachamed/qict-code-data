#!/usr/bin/env python3
from pathlib import Path
import argparse,json
REQ=['BUFFER_CONTRACTION_CERTIFICATE.json','CORE_RECOUPLING_L24.npz','RANK6_EMBEDDING_L24.npz','GLOBAL_CHARGED_UCONE_PROVIDER.py']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dir',type=Path,required=True);a=ap.parse_args();p=a.dir
 miss=[]
 if len(list(p.glob('KATO_BUFFER_*.npz')))<4:miss.append('>=4 KATO_BUFFER_<b>.npz')
 if not list(p.glob('C10_SHELL_MAPS_*.npz')):miss.append('C10_SHELL_MAPS_<b>.npz')
 if not list(p.glob('BOUNDARY_TO_PATH_*.npz')):miss.append('BOUNDARY_TO_PATH_<b>.npz')
 for x in REQ:
  if not (p/x).exists():miss.append(x)
 o={'status':'READY' if not miss else 'BLOCKED','physical_dir':str(p),'missing':miss}
 print(json.dumps(o,indent=2)); raise SystemExit(0 if not miss else 3)
if __name__=='__main__':main()
