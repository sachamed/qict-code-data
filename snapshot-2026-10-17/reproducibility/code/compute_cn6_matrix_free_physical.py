#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, hashlib, os, pathlib, sys, time, resource, gc
import numpy as np
from provider_contract import validate_provider, SOURCE_ORDER

def sha256(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path:pathlib.Path,obj):
    path=pathlib.Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    os.replace(tmp,path)

def atomic_npz(path:pathlib.Path,**kw):
    path=pathlib.Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'wb') as f: np.savez_compressed(f,**kw)
    os.replace(tmp,path)

def block_gram(A,B=None,block_rows=131072):
    if B is None: B=A
    if A.shape[0]!=B.shape[0] or A.shape[1]!=6 or B.shape[1]!=6: raise ValueError('block_gram expects (D,6)')
    G=np.zeros((6,6),np.complex128)
    for s in range(0,A.shape[0],block_rows):
        e=min(A.shape[0],s+block_rows)
        aa=np.asarray(A[s:e]); bb=np.asarray(B[s:e])
        G += aa.conj().T @ bb
    return G

def fro2(A,block_rows=131072):
    z=0.0
    for s in range(0,A.shape[0],block_rows):
        e=min(A.shape[0],s+block_rows); x=np.asarray(A[s:e])
        z += float(np.vdot(x,x).real)
    return z

def init_source(mod,D,path,block_rows):
    tmp=path.with_suffix('.tmp.npy')
    if tmp.exists(): tmp.unlink()
    out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.complex128,shape=(D,6))
    if hasattr(mod,'source_matrix_into'):
        ret=mod.source_matrix_into(out)
        if ret is not None and ret is not out:
            arr=np.asarray(ret,np.complex128)
            if arr.shape!=(D,6): raise ValueError('source_matrix_into returned bad shape')
            out[:]=arr
    else:
        arr=np.asarray(mod.source_matrix(),np.complex128)
        if arr.shape!=(D,6): raise ValueError(f'source_matrix shape {arr.shape} != {(D,6)}')
        for s in range(0,D,block_rows): out[s:s+block_rows]=arr[s:s+block_rows]
        del arr
    out.flush(); del out; gc.collect(); os.replace(tmp,path)
    return np.load(path,mmap_mode='r')

def apply_provider(mod,W,out_path,D):
    tmp=out_path.with_suffix('.tmp.npy')
    if tmp.exists(): tmp.unlink()
    Y=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.complex128,shape=(D,6))
    if hasattr(mod,'matmat_into'):
        ret=mod.matmat_into(W,Y)
        if ret is not None and ret is not Y:
            arr=np.asarray(ret,np.complex128)
            if arr.shape!=(D,6): raise ValueError('matmat_into returned bad shape')
            Y[:]=arr
    else:
        arr=np.asarray(mod.matmat(W),np.complex128)
        if arr.shape!=(D,6): raise ValueError(f'matmat returned {arr.shape}, expected {(D,6)}')
        if not np.all(np.isfinite(arr)): raise ValueError('nonfinite matmat output')
        Y[:]=arr; del arr
    Y.flush(); del Y; gc.collect(); os.replace(tmp,out_path)
    return np.load(out_path,mmap_mode='r')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--provider',type=pathlib.Path,required=True)
    ap.add_argument('--L',type=int,required=True)
    ap.add_argument('--nmax',type=int,required=True)
    ap.add_argument('--out-dir',type=pathlib.Path,required=True)
    ap.add_argument('--block-rows',type=int,default=131072)
    ap.add_argument('--keep-states',type=int,default=2,help='Keep this many latest W state files')
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location(f'physical_ucone_provider_L{a.L}',a.provider)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    meta,D=validate_provider(mod,a.L)
    Vpath=a.out_dir/f'SOURCE_V_L{a.L}.npy'
    if Vpath.exists(): V=np.load(Vpath,mmap_mode='r')
    else: V=init_source(mod,D,Vpath,a.block_rows)
    if V.shape!=(D,6): raise ValueError('stored source shape mismatch')
    gram=block_gram(V,block_rows=a.block_rows); gres=float(np.linalg.norm(gram-np.eye(6)))
    if gres>1e-10: raise ValueError(f'source columns not isometric: {gres}')
    latest_path=a.out_dir/'LATEST.json'
    if latest_path.exists():
        latest=json.loads(latest_path.read_text()); n0=int(latest['n']); Wpath=a.out_dir/latest['state_file']
        if sha256(Wpath)!=latest['state_sha256']: raise ValueError('latest state hash mismatch')
        W=np.load(Wpath,mmap_mode='r')
        mz=np.load(a.out_dir/f'CN6_MOMENTS_L{a.L}.npz',allow_pickle=False); C=np.asarray(mz['moments'],np.complex128)
        if C.shape[0]!=n0+1: raise ValueError('moments/latest depth mismatch')
    else:
        n0=0; Wpath=a.out_dir/f'W_L{a.L}_n00.npy'
        if not Wpath.exists():
            tmp=Wpath.with_suffix('.tmp.npy'); out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.complex128,shape=(D,6))
            for s in range(0,D,a.block_rows): out[s:s+a.block_rows]=V[s:s+a.block_rows]
            out.flush(); del out; os.replace(tmp,Wpath)
        W=np.load(Wpath,mmap_mode='r'); C=np.empty((1,6,6),np.complex128); C[0]=gram
        atomic_npz(a.out_dir/f'CN6_MOMENTS_L{a.L}.npz',C=C,moments=C,direct_moments=C,source_gram=gram)
        atomic_json(latest_path,{'n':0,'state_file':Wpath.name,'state_sha256':sha256(Wpath),'moments_file':f'CN6_MOMENTS_L{a.L}.npz'})
    rows=[]; t0=time.time()
    for n in range(n0+1,a.nmax+1):
        ts=time.time(); Ypath=a.out_dir/f'W_L{a.L}_n{n:02d}.npy'; Y=apply_provider(mod,W,Ypath,D)
        nrm=fro2(Y,a.block_rows); drift=abs(nrm-6.0)
        Cn=block_gram(V,Y,a.block_rows)
        C=np.concatenate([C,Cn[None,:,:]],axis=0)
        atomic_npz(a.out_dir/f'CN6_MOMENTS_L{a.L}.npz',C=C,moments=C,direct_moments=C,source_gram=gram)
        state_hash=sha256(Ypath)
        atomic_json(latest_path,{'n':n,'state_file':Ypath.name,'state_sha256':state_hash,'moments_file':f'CN6_MOMENTS_L{a.L}.npz'})
        rows.append({'n':n,'frobenius_norm2_six_columns':nrm,'drift_from_6':drift,'seconds':time.time()-ts,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'state_sha256':state_hash})
        del W; W=Y; gc.collect()
        # Keep only a small rollback window; moments are tiny and retained in full.
        old=n-a.keep_states
        if old>=0:
            op=a.out_dir/f'W_L{a.L}_n{old:02d}.npy'
            if op.exists() and op!=Vpath:
                try: op.unlink()
                except OSError: pass
    out={'status':'PASS_MATRIX_FREE_CN6_STREAM' if all(r['drift_from_6']<1e-7 for r in rows) else 'FAIL_UNITARITY_DRIFT',
         'provider':str(a.provider),'provider_sha256':sha256(a.provider),'provider_metadata':meta,'dimension':D,'L':a.L,'nmax':a.nmax,'source_order':list(SOURCE_ORDER),'source_isometry_residual':gres,'rows':rows,'runtime_s':time.time()-t0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'moments_file':f'CN6_MOMENTS_L{a.L}.npz','moments_sha256':sha256(a.out_dir/f'CN6_MOMENTS_L{a.L}.npz'),
         'scope':'Physical only if provider_contract validation passed; computes C_n^(6)=V^dagger U^n V using disk-backed six-column block propagation and no dense global operator.'}
    atomic_json(a.out_dir/f'CN6_MATRIX_FREE_CERTIFICATE_L{a.L}.json',out)
    print(json.dumps(out,indent=2,sort_keys=True))
    if out['status'].startswith('FAIL'): raise SystemExit(4)
if __name__=='__main__': main()
