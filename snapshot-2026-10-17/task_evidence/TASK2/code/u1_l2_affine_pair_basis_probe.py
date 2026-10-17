import numpy as np, heapq, math, json
from collections import deque
L=2
sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
links=[(mu,x,y,z) for mu in range(3) for x,y,z in sites]; li={t:i for i,t in enumerate(links)}
plaquettes=[]
for mu in range(3):
 for nu in range(mu+1,3):
  for x,y,z in sites:
   a=[x,y,z]; am=a.copy(); am[mu]^=1; an=a.copy(); an[nu]^=1
   v=np.zeros(len(links),dtype=np.int8)
   v[li[(mu,*a)]]+=1; v[li[(nu,*am)]]+=1; v[li[(mu,*an)]]-=1; v[li[(nu,*a)]]-=1
   plaquettes.append(v)
plaquettes=np.asarray(plaquettes,dtype=np.int8)

def pplus(e): return 2*e*e+e

def div(E):
 d={s:0 for s in sites}
 for idx,(mu,x,y,z) in enumerate(links):
  a=(x,y,z); b=[x,y,z]; b[mu]^=1; b=tuple(b)
  d[a]+=int(E[idx]); d[b]-=int(E[idx])
 return d

def minflow(src=(0,0,0),sink=(1,1,1),q=3):
 E=np.zeros(24,dtype=np.int16)
 # successive shortest path with convex marginal pplus(k+1)-pplus(k), forward directed edges only
 for unit in range(q):
  dist={s:1e99 for s in sites}; prev={}; dist[src]=0; pq=[(0,src)]
  while pq:
   du,a=heapq.heappop(pq)
   if du!=dist[a]: continue
   if a==sink: break
   for mu in range(3):
    idx=li[(mu,*a)]; b=list(a); b[mu]^=1; b=tuple(b)
    k=int(E[idx]); w=pplus(k+1)-pplus(k)
    nd=du+w
    if nd<dist[b]-1e-12:
     dist[b]=nd; prev[b]=(a,idx); heapq.heappush(pq,(nd,b))
  a=sink
  while a!=src:
   pa,idx=prev[a]; E[idx]+=1; a=pa
 return E

def basis(M,src=(0,0,0),sink=(1,1,1),q=3):
 E0=minflow(src,sink,q); z=tuple(map(int,E0)); Q=deque([z]); seen={z}
 def KE(t): return sum(pplus(e) for e in t)
 while Q:
  E=np.asarray(Q.popleft(),dtype=np.int16)
  for b in plaquettes:
   for sg in (-1,1):
    F=E+sg*b; t=tuple(map(int,F));
    if t in seen: continue
    if KE(t)<=M:
     seen.add(t); Q.append(t)
 return E0,seen

if __name__=='__main__':
 E0=minflow(); print('E0 KE',sum(pplus(int(e)) for e in E0),'E2',int(E0@E0),'nonzero',[(links[i],int(e)) for i,e in enumerate(E0) if e])
 print('div',div(E0))
 for M in [27,30,33,36,40,45,50]:
  E0,s=basis(M); print('M',M,'dim',len(s),'KEmin',min(sum(pplus(e) for e in t) for t in s))
