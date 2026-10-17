#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

def sha256_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--archive',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    zpath=Path(a.archive)
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist(); roots=sorted({n.split('/')[0] for n in names if '/' in n})
        if len(roots)!=1: raise SystemExit('archive must contain one primary-data root')
        root=roots[0]+'/'
        def read(rel): return z.read(root+rel)
        manifest={}
        for line in read('MANIFEST_SHA256.txt').decode().splitlines():
            if line.strip():
                h,rel=line.split(None,1); manifest[rel.strip()]=h
        manifest_consistency=all(sha256_bytes(read(rel))==h for rel,h in manifest.items())
        if not manifest_consistency: raise SystemExit('primary-data manifest mismatch')
        desc=json.loads(read('DATA_DESCRIPTION.json'))
        payload=np.load(io.BytesIO(read('L24_N9_RANK6_PRECONTRACTION.npz')),allow_pickle=False)
        required_shapes={
          'source_order':(6,), 'family_pairs':(3,2), 'links':(21136,4), 'plaquettes':(18252,5),
          'su2_connected_pairs':(261346,2), 'c10_nodes':(3925,3), 'c10_edge_src':(17576,),
          'shell10_links':(16008,4), 'shell11_links':(19554,4), 'shell12_links':(23456,4)}
        shape_consistency=all(tuple(payload[k].shape)==v for k,v in required_shapes.items())
        if not shape_consistency: raise SystemExit('primary-data array shape mismatch')
        source_order=[str(x) for x in payload['source_order'].tolist()]
        stats={
          'L':int(payload['L']),'n':int(payload['n']),'source_order':source_order,
          'family_pairs_zero_based':payload['family_pairs'].astype(int).tolist(),
          'links':int(payload['links'].shape[0]),'plaquettes':int(payload['plaquettes'].shape[0]),
          'su2_connected_pairs':int(payload['su2_connected_pairs'].shape[0]),
          'su2_translation_orientation_classes':int(payload['su2_class_signature'].shape[0]),
          'c10_nodes':int(payload['c10_nodes'].shape[0]),'c10_edges':int(payload['c10_edge_src'].shape[0]),
          'shell10_links':int(payload['shell10_links'].shape[0]),'shell10_plaquettes':int(payload['shell10_plaquettes'].shape[0]),
          'shell11_links':int(payload['shell11_links'].shape[0]),'shell11_plaquettes':int(payload['shell11_plaquettes'].shape[0]),
          'shell12_links':int(payload['shell12_links'].shape[0]),'shell12_plaquettes':int(payload['shell12_plaquettes'].shape[0])}
        if stats['L']!=desc['L'] or stats['n']!=desc['moment_depth'] or source_order!=desc['source_order']:
            raise SystemExit('description/payload consistency mismatch')
        out={
          'classification':'EXACT_L24_N9_RANK6_PRECONTRACTION',
          'archive_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),
          'manifest_consistency':'verified','array_shape_consistency':'verified','description_consistency':'verified',
          'statistics':stats,
          'primary_members':sorted(manifest),
          'common_space_continuation':{
            'operator_layer':'coherence-resolved C10 map + boundary-to-path insertion + six-column embedding + charged causal-cone action + selected Riesz state',
            'spectral_output':'C_n^(6) followed by the matrix spectral measure G_2 on one frozen action',
            'interpretation':'The archive supplies the finite L=24, n=9 rank-six coordinates used by the common-space spectral construction.'}}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(out['classification']); print(json.dumps(stats,sort_keys=True))
if __name__=='__main__': main()
