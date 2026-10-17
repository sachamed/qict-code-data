import itertools, math, json, functools
import numpy as np, opt_einsum as oe
L=2
sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
si={s:i for i,s in enumerate(sites)}
links=[]
for mu in range(3):
  for s in sites:
    t=list(s); t[mu]^=1; t=tuple(t)
    links.append((mu,s,t))
# incident link ids at each site
inc=[[] for _ in sites]
for e,(mu,s,t) in enumerate(links):
  inc[si[s]].append(e); inc[si[t]].append(e)
assert all(len(x)==6 for x in inc)
@functools.lru_cache(None)
def fuse_mult(spins):
  # invariant multiplicity of SU2 reps, twice-spins
  cur={0:1}
  # better start scalar then tensor each spin
  for s in spins:
    nxt={}
    for a,m in cur.items():
      for b in range(abs(a-s),a+s+1,2): nxt[b]=nxt.get(b,0)+m
    cur=nxt
  return cur.get(0,0)

def count(j2max):
  vals=list(range(j2max+1)); d=len(vals)
  # each vertex tensor over its six edge variables
  Ts=[]; inds=[]
  for v in range(8):
    T=np.zeros((d,)*6,dtype=np.float64)
    for idx in itertools.product(range(d), repeat=6):
      T[idx]=fuse_mult(tuple(vals[i] for i in idx))
    Ts.append(T); inds.append(inc[v])
  args=[]
  for T,ix in zip(Ts,inds): args += [T,ix]
  args += [[]]
  path,info=oe.contract_path(*args,optimize='auto-hq')
  val=oe.contract(*args,optimize=path)
  return int(round(float(val))), str(info)
if __name__=='__main__':
  import argparse
  ap=argparse.ArgumentParser();ap.add_argument('--max',type=int,default=4);a=ap.parse_args()
  out={}
  for j2 in range(1,a.max+1):
    n,info=count(j2);out[str(j2)]={'jmax':j2/2,'gauss_dim':n};print(j2,j2/2,n,flush=True)
  print(json.dumps(out,indent=2))
