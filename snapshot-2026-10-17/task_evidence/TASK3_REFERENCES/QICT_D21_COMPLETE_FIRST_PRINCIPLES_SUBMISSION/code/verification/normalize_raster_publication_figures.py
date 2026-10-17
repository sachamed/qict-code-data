#!/usr/bin/env python3
"""Deterministically normalize retained legacy raster publication figures to 600 dpi.

No scientific values, labels, geometry, or colormap are changed.  Images below
300 dpi are resampled with Lanczos at fixed physical size and written with
600-dpi metadata.  Vector PDF masters are preferred by TeX wherever available.
"""
from pathlib import Path
from PIL import Image
import json
ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'figures'
TARGET=600.0
NAMES=[
 'electroweak_uncertainty_propagation.png',
 'finite_algebra_exclusion.png',
 'gauge_unimodular_reduction.png',
 'primitive_algebra_exclusion.png',
]
records=[]
for name in NAMES:
    p=FIG/name
    if not p.exists():
        raise FileNotFoundError(p)
    with Image.open(p) as im:
        dpi=im.info.get('dpi',(220.0,220.0))
        dx=float(dpi[0] or 220.0); dy=float(dpi[1] or dx)
        old=im.size
        if min(dx,dy) < 300:
            new=(max(1,round(old[0]*TARGET/dx)), max(1,round(old[1]*TARGET/dy)))
            out=im.convert('RGB').resize(new,Image.Resampling.LANCZOS)
            out.save(p,dpi=(TARGET,TARGET),optimize=True)
        with Image.open(p) as chk:
            ndpi=chk.info.get('dpi',(0,0)); nsize=chk.size
        records.append({'file':name,'old_size':old,'new_size':nsize,'new_dpi':[float(ndpi[0]),float(ndpi[1])]})
        if min(ndpi)<590:
            raise RuntimeError(f'{name}: dpi normalization failed: {ndpi}')
out={'status':'PASS','target_dpi':TARGET,'files':records}
(ROOT/'validation/Raster_Figure_Normalization_2026-09-12.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
