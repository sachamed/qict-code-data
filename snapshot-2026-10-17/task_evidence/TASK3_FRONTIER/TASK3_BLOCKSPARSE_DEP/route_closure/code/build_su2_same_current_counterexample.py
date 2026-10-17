import numpy as np, json, math
from collections import defaultdict
coords=[(x,y,z) for x in range(2) for y in range(2) for z in range(2)]
links=[(mu,x,y,z) for mu in range(3) for x,y,z in coords]; li={t:i for i,t in enumerate(links)}
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]],complex); sz=np.array([[1,0],[0,-1]],complex)
SIG=[sx,sy,sz]
def move(s,mu): a=list(s);a[mu]^=1;return tuple(a)
def su2(axis,ang):
 S=SIG[axis]
 return math.cos(ang/2)*np.eye(2)-1j*math.sin(ang/2)*S
U=[]
for e,(mu,x,y,z) in enumerate(links):
 axis=(e+mu)%3; ang=0.17+0.071*(e+1)
 U.append(su2(axis,ang))
# path state: site, current tuple, ordered signed link word
states=[((0,0,0),(0,)*24,())]
for d in range(1,4):
 for mu in (2,1,0):
  out=[]
  for site,J,w in states:
   y=move(site,mu); e=li[(mu,*site)]; jj=list(J);jj[e]+=1;out.append((y,tuple(jj),w+(e+1,)))
   y=move(site,mu); e=li[(mu,*y)]; jj=list(J);jj[e]-=1;out.append((y,tuple(jj),w+(-(e+1),)))
  states=out
G=defaultdict(list)
for site,J,w in states:G[(site,J)].append(w)
def hol(w):
 H=np.eye(2,dtype=complex)
 for g in w:
  M=U[abs(g)-1]; M=M if g>0 else M.conj().T
  H=M@H
 return H
best=None
for (site,J),ws in G.items():
 if len(ws)<2: continue
 vals=[np.trace(hol(w))/2 for w in ws]
 for i in range(len(ws)):
  for j in range(i+1,len(ws)):
   diff=abs(vals[i]-vals[j])
   if best is None or diff>best[0]: best=(diff,site,J,ws[i],ws[j],vals[i],vals[j],hol(ws[i]),hol(ws[j]))
diff,site,J,w1,w2,t1,t2,H1,H2=best
out={'status':'PASS_DETERMINISTIC_COUNTEREXAMPLE_TO_SU2_CURRENT_ONLY_COMPRESSION','d_macro':3,'substeps':9,'same_endpoint':list(site),'same_integer_link_current':list(J),'path_word_1':list(w1),'path_word_2':list(w2),'normalized_SU2_trace_1':[float(t1.real),float(t1.imag)],'normalized_SU2_trace_2':[float(t2.real),float(t2.imag)],'trace_difference_abs':float(diff),'holonomy_matrix_difference_fro':float(np.linalg.norm(H1-H2)),'link_assignment':'U_e=exp[-i theta_e sigma_axis/2], theta_e=0.17+0.071(e+1), axis=(e+mu_e) mod 3','conclusion':'The abelian current vector does not determine the SU(2) ordered holonomy. A common-network contraction must preserve path ordering or an equivalent non-Abelian transfer state.'}
open('/mnt/data/SU2_CURRENT_ORDER_COUNTEREXAMPLE.json','w').write(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
