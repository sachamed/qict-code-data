#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pathlib,subprocess,sys,os

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True); ap.add_argument('--providers-dir',type=pathlib.Path,required=True); ap.add_argument('--schedule',type=pathlib.Path,required=True); a=ap.parse_args()
 sched=json.loads(a.schedule.read_text())['volumes']; here=pathlib.Path(__file__).resolve().parent; rows=[]
 env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
 for Ls,s in sched.items():
  L=int(Ls); nmax=int(s['nmax']); provider=a.providers_dir/f'L{L}'/'GLOBAL_CHARGED_UCONE_PROVIDER.py'; rd=a.root/f'L{L}'; rd.mkdir(parents=True,exist_ok=True)
  if not provider.exists(): rows.append({'L':L,'status':'BLOCKED_MISSING_PROVIDER','provider':str(provider)}); continue
  guard_report=rd/f'MEMORY_GUARD_L{L}.json'
  compute=[sys.executable,str(here/'compute_cn6_matrix_free.py'),'--provider',str(provider),'--L',str(L),'--nmax',str(nmax),'--out-dir',str(rd)]
  p=subprocess.run([sys.executable,str(here/'memory_guard.py'),'--limit-mib','4200','--report',str(guard_report),'--']+compute,env=env)
  if p.returncode!=0: rows.append({'L':L,'status':'FAIL_CN6','returncode':p.returncode}); continue
  ec=rd/f'CN6_ERROR_CERTIFICATE_L{L}.json'
  cmd=[sys.executable,str(here/'run_c11_after_cn6.py'),'--L',str(L),'--results-dir',str(rd)]
  if ec.exists(): cmd+=['--error-certificate',str(ec)]
  q=subprocess.run(cmd,env=env); rows.append({'L':L,'status':'PASS_SOFTWARE_CHAIN' if q.returncode==0 else 'FAIL_C11','returncode':q.returncode})
 out={'status':'PASS_ALL_VOLUME_SOFTWARE_CHAIN' if rows and all(r['status']=='PASS_SOFTWARE_CHAIN' for r in rows) else 'INCOMPLETE_PHYSICAL_MULTIVOLUME_CHAIN','rows':rows}
 (a.root/'MULTIVOLUME_SOFTWARE_CHAIN_STATUS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
