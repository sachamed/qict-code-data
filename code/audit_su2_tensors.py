#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, io, json, pickle, pickletools, zipfile
import numpy as np
from pathlib import Path

class SafeTensorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'builtins' and name == 'complex':
            return complex
        raise pickle.UnpicklingError(f'disallowed global {module}.{name}')
    def persistent_load(self, pid):
        raise pickle.UnpicklingError('persistent IDs are not accepted')

def safe_load(data: bytes):
    return SafeTensorUnpickler(io.BytesIO(data)).load()

def sha(data: bytes): return hashlib.sha256(data).hexdigest()

def validate_manifest(z: zipfile.ZipFile):
    man=json.loads(z.read('DATA_MANIFEST.json'))
    checked=0
    for item in man['files']:
        b=z.read(item['path'])
        if len(b)!=item['bytes'] or sha(b)!=item['sha256']:
            raise RuntimeError(f"manifest mismatch: {item['path']}")
        checked += 1
    return man,checked

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sequential',required=True)
    ap.add_argument('--cluster',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args(); result={}

    with zipfile.ZipFile(a.sequential) as z:
        man,nchecked=validate_manifest(z)
        core_names=sorted(n for n in z.namelist() if n.startswith('cores/core_') and n.endswith('.pkl'))
        bond_names=sorted(n for n in z.namelist() if n.startswith('bonds/bond_') and n.endswith('.pkl'))
        core_counts={}; max_left=max_right=-1; global_names=set()
        for n in core_names:
            b=z.read(n)
            count=0
            for op,arg,pos in pickletools.genops(b):
                if op.name=='TUPLE3': count += 1
                elif op.name=='GLOBAL': global_names.add(str(arg))
            obj=safe_load(b)
            if not isinstance(obj,dict): raise TypeError(n)
            if len(obj)!=count: raise RuntimeError(f'static/dynamic entry mismatch {n}: {count} vs {len(obj)}')
            for k in obj:
                if not (isinstance(k,tuple) and len(k)==3): raise TypeError(f'bad tensor key in {n}')
                l,phys,r=k; max_left=max(max_left,int(l)); max_right=max(max_right,int(r))
            core_counts[Path(n).name]=len(obj)
            del obj
        geom_name='geometry/N9_RELATIVE_CONE_IR.npz'
        geom_bytes=z.read(geom_name)
        geom_sha=sha(geom_bytes)
        with np.load(io.BytesIO(geom_bytes), allow_pickle=False) as g:
            required={'links','plaquettes','plaquette_edge_indices','plaquette_intervals','shell_plaquette_mask','shell_link_mask','n','max_open','max_open_cut'}
            missing=required.difference(g.files)
            if missing:
                raise RuntimeError(f'missing geometry arrays: {sorted(missing)}')
            geometry_summary={
                'link_records':int(g['links'].shape[0]),
                'plaquette_records':int(g['plaquettes'].shape[0]),
                'plaquette_edge_records':int(g['plaquette_edge_indices'].shape[0]),
                'plaquette_interval_records':int(g['plaquette_intervals'].shape[0]),
                'shell_plaquette_records':int(np.asarray(g['shell_plaquette_mask'], dtype=bool).sum()),
                'shell_link_records':int(np.asarray(g['shell_link_mask'], dtype=bool).sum()),
                'moment_depth':int(np.asarray(g['n']).reshape(-1)[0]),
                'max_open':int(np.asarray(g['max_open']).reshape(-1)[0]),
                'max_open_cut':int(np.asarray(g['max_open_cut']).reshape(-1)[0]),
            }
        result['sequential_SU2']={
            'manifest_files_verified':nchecked,
            'core_tensor_files':len(core_names),
            'bond_sector_files':len(bond_names),
            'total_core_nonzero_coefficients':int(sum(core_counts.values())),
            'core_nonzero_counts':core_counts,
            'largest_observed_core_bond_index_plus_one':int(max(max_left,max_right)+1),
            'geometry_ir_sha256':geom_sha,
            'geometry':geometry_summary,
            'pickle_global_policy':'Primitive containers and builtins.complex are parsed; executable constructors are disabled.'
        }

    with zipfile.ZipFile(a.cluster) as z:
        man,nchecked=validate_manifest(z)
        shard_names=sorted(n for n in z.namelist() if n.startswith('shards/bucket_') and n.endswith('.pkl'))
        support=0; norm2=0.0; keys=set(); duplicate=0
        for n in shard_names:
            obj=safe_load(z.read(n))
            if not isinstance(obj,dict) or 'partial' not in obj or not isinstance(obj['partial'],dict): raise TypeError(n)
            part=obj['partial']; support += len(part)
            norm2 += sum((abs(complex(v))**2 for v in part.values()))
            for k in part:
                if k in keys: duplicate += 1
                else: keys.add(k)
            del obj
        md=man['metadata']
        result['four_plaquette_SU2']={
            'manifest_files_verified':nchecked,
            'shard_count':len(shard_names),
            'support_direct':int(support),
            'unique_keys_direct':int(len(keys)),
            'cross_shard_duplicate_keys':int(duplicate),
            'norm2_direct':float(norm2),
            'plaquette_sequence':md['plaquette_sequence'],
            'shared_link':md['shared_link'],
            'pickle_global_policy':'Primitive containers and builtins.complex are parsed; executable constructors are disabled.'
        }
    result['classification']='PRIMARY_SU2_TENSOR_VALIDATION'
    Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
