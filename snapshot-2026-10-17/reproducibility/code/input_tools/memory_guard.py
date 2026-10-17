#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time, signal, os
import psutil

def tree_rss(proc):
    total=0; pids=[]
    try:
        ps=[proc]+proc.children(recursive=True)
    except psutil.Error:
        ps=[proc]
    for p in ps:
        try:
            total += p.memory_info().rss; pids.append(p.pid)
        except psutil.Error: pass
    return total,pids

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit-mib',type=float,required=True); ap.add_argument('--report',type=pathlib.Path,required=True); ap.add_argument('cmd',nargs=argparse.REMAINDER); a=ap.parse_args()
    cmd=a.cmd[1:] if a.cmd and a.cmd[0]=='--' else a.cmd
    if not cmd: raise SystemExit('missing command')
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    p=subprocess.Popen(cmd,env=env,start_new_session=True); ps=psutil.Process(p.pid)
    lim=int(a.limit_mib*1024**2); peak=0; over=0; killed=False; samples=0; t0=time.time()
    while p.poll() is None:
        rss,pids=tree_rss(ps); peak=max(peak,rss); samples+=1
        if rss>lim: over+=1
        else: over=0
        if over>=3:
            killed=True
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            time.sleep(1.0)
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            break
        time.sleep(0.5)
    rc=p.wait()
    out={'status':'ABORTED_MEMORY_GUARD' if killed else ('PASS' if rc==0 else 'CHILD_FAILED'),'returncode':rc,'limit_mib':a.limit_mib,'peak_tree_rss_mib':peak/1024**2,'samples':samples,'runtime_s':time.time()-t0,'command':cmd,'checkpoint_policy':'Child must use atomic checkpoints; safe to resume after guard abort.'}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
    raise SystemExit(0 if rc==0 and not killed else (75 if killed else rc))
if __name__=='__main__': main()
