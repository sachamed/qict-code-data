#!/usr/bin/env python3
"""QICT reproducibility runner: transactional stages, resumable verified checkpoints.

Scientific scope: reproduces finite calculations in the existing QICT release.
It cannot supply absent coherent physical G2 data or infer uncomputed masses.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest_files(base: Path, patterns: tuple[str,...]) -> dict[str,str]:
    out = {}
    for pat in patterns:
        for p in sorted(base.rglob(pat)):
            if p.is_file() and '__pycache__' not in p.parts:
                out[str(p.relative_to(base))] = digest(p)
    return out


def file_snapshot(base: Path) -> dict[str,tuple[int,int,str]]:
    return {str(p.relative_to(base)):(p.stat().st_size,p.stat().st_mtime_ns,digest(p))
            for p in sorted(base.rglob('*')) if p.is_file() and p.name != 'RUN_MANIFEST.json'}


def stable_hash(data: object) -> str:
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic_json(path: Path, data: object):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(data,stream,sort_keys=True,indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def restore_from_backup(results: Path, backup: Path):
    # This restores exactly the previous files following unsuccessful execution.
    for p in sorted(results.rglob('*'),reverse=True):
        if p.is_file():p.unlink()
    for p in sorted(results.rglob('*'),reverse=True):
        if p.is_dir():p.rmdir()
    shutil.copytree(backup,results,dirs_exist_ok=True)


def main(calls: list[list[str]], root:Path, results:Path):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-steps',type=int,default=0,help='At most N new computations in this invocation; 0=all')
    ap.add_argument('--step-timeout',type=int,default=180,help='Per subprocess seconds; interrupted step can be retried')
    ap.add_argument('--reset-checkpoint',action='store_true',help='Invalidate cache but preserve existing results')
    ap.add_argument('--status',action='store_true',help='Show current task progress and exit')
    args = ap.parse_args()
    checkpoints = root/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    checkpoint_path=checkpoints/'RUN_CHECKPOINT.json'
    old_manifest_path=results/'RUN_MANIFEST.json'
    old_manifest=json.loads(old_manifest_path.read_text()) if old_manifest_path.exists() else {}
    original_owner=old_manifest.get('generated_by_program',{})
    checkpoint={} if args.reset_checkpoint or not checkpoint_path.exists() else json.loads(checkpoint_path.read_text())
    # Crash recovery: a persistent write-ahead journal survives a killed parent
    # process. This can restore the result directory before any new computation.
    staged=checkpoint.get('in_progress')
    if staged and not args.status:
        backup=Path(staged.get('backup_path',str(checkpoints/'stage_backup')))
        if not backup.is_dir():
            raise RuntimeError('An interrupted stage has no recoverable backup; do not silently proceed')
        restore_from_backup(results,backup)
        checkpoint.pop('in_progress',None)
        checkpoint['last_recovery']={'stage':staged,'status':'RESTORED_PRE_STAGE_RESULTS'}
        atomic_json(checkpoint_path,checkpoint)
        shutil.rmtree(backup)
        print('RECOVERED interrupted stage:',staged,flush=True)
    if args.status:
        finished=[int(x)+1 for x,v in checkpoint.get('tasks',{}).items() if v.get('ok')]
        print(json.dumps({'completed_tasks':len(finished),'total_tasks':len(calls),'last_completed_index':max(finished,default=0),
                          'complete_manifest':checkpoint.get('complete_manifest',False),'last_error':checkpoint.get('last_error')},indent=2))
        return
    sources={
      'code':digest_files(root/'code',('*.py','requirements.txt')),
      'data':digest_files(root/'data',('*.json','*.csv','*.txt','*.npz','*.zip')),
      'python':sys.version,
      'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','sympy','mpmath')},
    }
    source_hash=stable_hash(sources)
    if checkpoint.get('source_hash') not in (None,source_hash):
        print('SOURCE CHANGED: invalidating all cached steps, retaining existing scientific results.',flush=True)
        checkpoint={}
    checkpoint.setdefault('tasks',{})
    checkpoint.update({'schema_version':2,'source_hash':source_hash,'complete_manifest':False,
                       'total_tasks':len(calls)})
    atomic_json(checkpoint_path,checkpoint)
    chain=source_hash
    actual_executions=0
    regenerated={}
    started=time.monotonic()
    for i,call in enumerate(calls):
        owner=Path(call[1]).name
        trigger=stable_hash({'chain':chain,'index':i,'args':call,'source':source_hash})
        cached=checkpoint['tasks'].get(str(i))
        if cached and cached.get('ok') and cached.get('trigger') == trigger:
            hashes=cached.get('outputs',{})
            if all((results/p).is_file() and digest(results/p)==expected for p,expected in hashes.items()):
                regenerated.update({p:owner for p in hashes})
                chain=stable_hash({'trigger':trigger,'outputs':hashes})
                print(f'[{i+1:02d}/{len(calls)}] CACHED {owner}',flush=True)
                continue
        if args.max_steps and actual_executions >= args.max_steps:
            checkpoint['last_error']=None
            atomic_json(checkpoint_path,checkpoint)
            print(f'PAUSED: {i}/{len(calls)} stages reached, rerun the identical command to resume.',flush=True)
            return
        print(f'[{i+1:02d}/{len(calls)}] RUN {owner}',flush=True)
        before=file_snapshot(results)
        log_path=checkpoints/'logs'/f'{i+1:02d}_{owner}.log'
        log_path.parent.mkdir(parents=True,exist_ok=True)
        backup=Path(tempfile.mkdtemp(prefix='qict-stage-backup-'))/'results'
        shutil.copytree(results,backup)
        checkpoint['in_progress']={'index':i+1,'program':owner,'backup_path':str(backup)}
        atomic_json(checkpoint_path,checkpoint)
        child_env=os.environ.copy()
        for thread_var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
            child_env[thread_var]=os.environ.get('QICT_BLAS_THREADS','1')
        try:
            with log_path.open('w') as stream:
                stream.write('COMMAND: '+ ' '.join(call)+'\n');stream.flush()
                subprocess.run(call,check=True,timeout=args.step_timeout,stdout=stream,stderr=subprocess.STDOUT,cwd=root,env=child_env)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            restore_from_backup(results,backup)
            checkpoint.pop('in_progress',None)
            checkpoint['last_error']={'index':i+1,'program':owner,'error':repr(exc),
                                     'log':str(log_path.relative_to(root))}
            atomic_json(checkpoint_path,checkpoint)
            shutil.rmtree(backup)
            print('STOPPED AND ROLLED BACK:',checkpoint['last_error'],flush=True)
            sys.exit(2)
        after=file_snapshot(results)
        updated={p for p,a in after.items() if p not in before or before[p][:2] != a[:2] or before[p][2] != a[2]}
        expected={p for p,who in original_owner.items() if who==owner}
        old_record=checkpoint['tasks'].get(str(i),{})
        expected.update(old_record.get('outputs',{}))
        missing=expected-set(after)
        if missing:
            checkpoint['last_error']={'index':i+1,'program':owner,
                                      'error':'Expected output missing: '+repr(sorted(missing))}
            atomic_json(checkpoint_path,checkpoint)
            print('OUTPUT ERROR:',checkpoint['last_error'],flush=True)
            sys.exit(3)
        # Before counting an output as current, require that its owner ran and
        # its file was refreshed, even if deterministic bytes are identical.
        stale=expected-updated
        if stale:
            checkpoint['last_error']={'index':i+1,'program':owner,
                                      'error':'Producer succeeded but did not rewrite: '+repr(sorted(stale))}
            atomic_json(checkpoint_path,checkpoint)
            print('FRESHNESS ERROR:',checkpoint['last_error'],flush=True)
            sys.exit(3)
        owned=(updated|expected)
        outputs={p:after[p][2] for p in sorted(owned)}
        checkpoint['tasks'][str(i)]={'ok':True,'name':owner,'trigger':trigger,'outputs':outputs,
                                      'seconds_since_start':round(time.monotonic()-started,3)}
        checkpoint['last_error']=None
        checkpoint.pop('in_progress',None)
        atomic_json(checkpoint_path,checkpoint)
        shutil.rmtree(backup)
        regenerated.update({p:owner for p in outputs})
        chain=stable_hash({'trigger':trigger,'outputs':outputs})
        actual_executions+=1
        print(f'[{i+1:02d}/{len(calls)}] PASS {owner} ({len(outputs)} output(s))',flush=True)
    current=file_snapshot(results)
    retained=set(current)-set(regenerated)
    # Existing outputs from other declared groups remain explicitly retained.
    # A successful baseline run does not certify that these were recomputed.
    manifest={
      'classification':'REPRODUCIBLE_CALCULATION_MANIFEST',
      'execution_status':'DECLARED_STAGES_COMPLETED_WITH_RETAINED_RESULTS' if retained else 'ALL_STAGES_COMPLETED',
      'execution_mode':'transactional_resumable',
      'python':platform.python_version(),
      'package_versions':sources['packages'],
      'invoked_programs':[Path(c[1]).name for c in calls],
      'generated_by_program':dict(sorted(regenerated.items())),
      'retained_results':sorted(retained),
      'code_sha256':{'code/'+p:h for p,h in sources['code'].items()},
      'primary_data_sha256':{'data/'+p:h for p,h in sources['data'].items()},
      'result_sha256':{'results/'+p:v[2] for p,v in current.items()},
      'pre_finalization_checkpoint_sha256':digest(checkpoint_path),
      'scientific_scope':'finite numerical/symbolic reproductions; does not imply completed physical G2 inputs',
    }
    atomic_json(old_manifest_path,manifest)
    checkpoint.update({'complete_manifest':True,'last_error':None,'manifest_sha256':digest(old_manifest_path)})
    atomic_json(checkpoint_path,checkpoint)
    print(f'COMPLETE: {len(calls)} programs, {len(current)} current artifacts; physical G2 inputs are separately audited.',flush=True)


if __name__=='__main__':
    print('Run through run_all.py to construct the canonical QICT execution list.')
