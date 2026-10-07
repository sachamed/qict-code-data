#!/usr/bin/env python3
from __future__ import annotations
import argparse,pathlib,subprocess,sys,json

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--providers-dir',type=pathlib.Path,required=True); ap.add_argument('--results-root',type=pathlib.Path,required=True); ap.add_argument('--schedule',type=pathlib.Path,required=True); a=ap.parse_args()
 here=pathlib.Path(__file__).resolve().parent; a.results_root.mkdir(parents=True,exist_ok=True)
 p=subprocess.run([sys.executable,str(here/'run_all_volumes.py'),'--root',str(a.results_root),'--providers-dir',str(a.providers_dir),'--schedule',str(a.schedule)])
 gate=a.results_root/'G2_PHYSICAL_ACCEPTANCE.json'
 q=subprocess.run([sys.executable,str(here/'g2_physical_acceptance_gate.py'),'--results-root',str(a.results_root),'--out',str(gate)])
 out={'status':'PASS_END_TO_END_PHYSICAL_G2' if p.returncode==0 and q.returncode==0 else 'END_TO_END_PHYSICAL_G2_INCOMPLETE','software_chain_returncode':p.returncode,'physical_g2_gate_returncode':q.returncode,'g2_gate_file':str(gate)}
 (a.results_root/'PIPELINE_STATUS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
 raise SystemExit(0 if out['status'].startswith('PASS') else 3)
if __name__=='__main__': main()
