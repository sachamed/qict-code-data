#!/usr/bin/env python3
from pathlib import Path
import json,re,sys
R=Path(__file__).resolve().parent;G=R/'p4_s6_factor_gram';pat=re.compile(r'g(\d+)_(\d+)\.done$');rows=[];bad=[]
for p in G.glob('g*.done'):
 m=pat.match(p.name)
 if not m: continue
 a,b=map(int,m.groups());d=json.load(open(p));sz=sum(x.stat().st_size for x in G.glob(f'b*_g{a:06d}_{b:06d}.bin'))
 if sz!=int(d['bytes']): bad.append((a,b,d['bytes'],sz))
 rows.append((a,b))
rows.sort();contig=(len(rows)==133 and rows[0][0]==0 and rows[-1][1]==187083 and all(rows[i][1]==rows[i+1][0] for i in range(len(rows)-1)))
tmp=list(R.rglob('*.tmp'));c=json.load(open(R/'BALANCED_GRAM_FULL_PASS_CERTIFICATE.json'));ok=contig and not bad and not tmp and c.get('all_checks_pass')
print(json.dumps({'status':'PASS_BALANCED_GRAM_FULL_PASS' if ok else 'FAIL','commits':len(rows),'contiguous':contig,'bad_binary_ranges':bad,'tmp_files':len(tmp)},indent=2));sys.exit(0 if ok else 2)
