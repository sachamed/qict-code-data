#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, time, heapq
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import su2_global_spin_network as gs
ROOT=Path(__file__).resolve().parents[1]
# undirected neighbor pairs, each with two parallel physical links
pairs=defaultdict(list)
for e in range(24):
 a,b=gs.link_endpoints(e); key=tuple(sorted((gs.SID[a],gs.SID[b]))); pairs[key].append(e)
pair_items=sorted(pairs.items())
assert len(pair_items)==12 and all(len(es)==2 for _,es in pair_items)

def is_tree(simple_ids):
 parent=list(range(8))
 def f(x):
  while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
  return x
 for i in simple_ids:
  (a,b),_=pair_items[i];ra,rb=f(a),f(b)
  if ra==rb:return False
  parent[ra]=rb
 return len({f(i) for i in range(8)})==1
simple_trees=[c for c in itertools.combinations(range(12),7) if is_tree(c)]
assert len(simple_trees)==384
P=[set(p['edges']) for p in gs.PLAQ]

def greedy_width(tree):
 rem=[e for e in range(24) if e not in tree]
 supports=[q-set(tree) for q in P]
 # ignore constant plaquettes (none/irrelevant to width)
 proc=set(); order=[]; prof=[]
 while rem:
  best=None
  for e in rem:
   s2=proc|{e}
   w=sum(bool(q&s2) and bool(q-(s2|set(tree))) for q in supports)
   # equivalent: q intersects processed and unprocessed rem
   # tie break by future incidence then id
   future=sum(e in q for q in supports)
   score=(w,-future,e)
   if best is None or score<best[0]:best=(score,e,w)
  _,e,w=best; proc.add(e);rem.remove(e);order.append(e)
  un=set(order)^set(order) # dummy
  # exact open count using remaining vars
  rr=set(x for x in range(24) if x not in tree and x not in proc)
  prof.append(sum(bool(q&proc) and bool(q&rr) for q in supports))
 return max(prof,default=0),order,prof

def exact_pathwidth(tree):
 vars=[e for e in range(24) if e not in tree]; n=len(vars); idx={e:i for i,e in enumerate(vars)}
 masks=[]
 for q in P:
  m=0
  for e in q:
   if e in idx:m|=1<<idx[e]
  if m: masks.append(m)
 full=(1<<n)-1
 # cut width for subset via bit ops
 widths=[0]*(1<<n)
 for s in range(1<<n):
  r=full^s; widths[s]=sum((m&s)!=0 and (m&r)!=0 for m in masks)
 INF=99;dp=[INF]*(1<<n);prev=[-1]*(1<<n);dp[0]=0
 for s in range(1,1<<n):
  w=widths[s];x=s
  best=INF;be=-1
  while x:
   b=x&-x;j=b.bit_length()-1;val=max(dp[s^b],w)
   if val<best:best=val;be=j
   x-=b
  dp[s]=best;prev[s]=be
 # reconstruct backwards
 s=full; rev=[]
 while s:
  j=prev[s];rev.append(vars[j]);s^=1<<j
 order=list(reversed(rev))
 prof=[];pr=set();rr=set(vars)
 for e in order:
  pr.add(e);rr.remove(e);prof.append(sum(bool((q-set(tree))&pr) and bool((q-set(tree))&rr) for q in P))
 return dp[full],order,prof

def main():
 t=time.time();best=[];count=0
 for st in simple_trees:
  for bits in range(1<<7):
   tree=set()
   for a,i in enumerate(st):tree.add(pair_items[i][1][(bits>>a)&1])
   w,o,p=greedy_width(tree);count+=1
   item=(w,tuple(sorted(tree)),o,p)
   if len(best)<40: heapq.heappush(best,(-w,item))
   elif w < -best[0][0]: heapq.heapreplace(best,(-w,item))
 # unique lowest heuristic candidates
 cand=sorted([x[1] for x in best],key=lambda z:z[0]); seen=set(); exact=[]
 for w,tr,o,p in cand:
  if tr in seen:continue
  seen.add(tr); ew,eo,ep=exact_pathwidth(set(tr));exact.append((ew,w,tr,eo,ep));print('candidate heuristic',w,'exact',ew,'tree',tr,flush=True)
 exact.sort(key=lambda x:(x[0],x[1])); B=exact[0]
 out={'status':'PASS_GAUGE_FIXED_SPANNING_TREE_PATHWIDTH_SEARCH','simple_spanning_trees':len(simple_trees),'parallel_tree_variants_examined':count,'best_exact_width':B[0],'best_heuristic_width':B[1],'tree_links':list(B[2]),'remaining_links':[e for e in range(24) if e not in B[2]],'optimal_order_on_remaining':B[3],'profile':B[4],'exact_candidates_checked':len(exact),'seconds':time.time()-t,'scope':'Exact lattice-tree gauge fixing removes seven link variables. Width is counted on the 24 plaquette hyperedges after substitution of tree links by the identity; physical validity for the charged relative correlator additionally requires the gauge-fixed matter/worldline tensors to be transformed consistently.'}
 (ROOT/'results'/'GAUGE_FIXED_TREEWIDTH_SEARCH.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
