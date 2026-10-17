import os,glob,json,re,sys
base=os.path.dirname(os.path.abspath(__file__))
out=os.path.join(base,'p4_s6_factor_gram')
bad=[]; n=0
for mp in sorted(glob.glob(os.path.join(out,'g*.done'))):
 m=re.search(r'g(\d{6})_(\d{6})\.done$',mp)
 if not m: continue
 a,b=m.groups(); meta=json.load(open(mp)); bins=glob.glob(os.path.join(out,f'b??_g{a}_{b}.bin'))
 size=sum(os.path.getsize(p) for p in bins)
 if size != meta.get('bytes'): bad.append((os.path.basename(mp),len(bins),size,meta.get('bytes')))
 n+=1
print('committed_ranges',n); print('bad_ranges',bad); sys.exit(1 if bad else 0)
