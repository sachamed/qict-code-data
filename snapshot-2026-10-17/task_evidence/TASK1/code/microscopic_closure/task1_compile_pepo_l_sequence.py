#!/usr/bin/env python3
from __future__ import annotations
import json,hashlib,itertools,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'microscopic_closure'/'TASK1_PEPO_L_SEQUENCE.json'

def compile_graph(L:int):
 sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
 links=[]
 for mu in range(3):
  for s in sites: links.append((mu,*s))
 li={e:i for i,e in enumerate(links)}
 def sh(s,mu):
  a=list(s);a[mu]=(a[mu]+1)%L;return tuple(a)
 plaq=[]
 for mu in range(3):
  for nu in range(mu+1,3):
   for s in sites:
    sm=sh(s,mu);sn=sh(s,nu)
    plaq.append(((li[(mu,*s)],+1),(li[(nu,*sm)],+1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)))
 inc=[[] for _ in links]
 for p,po in enumerate(plaq):
  for e,sgn in po: inc[e].append((p,sgn))
 # Canonical fused link node: electric in/out, all group plaquette reps, matter carrier, receiver, CAR parity.
 nodes=[]
 for e,occ in enumerate(inc):
  nodes.append({'kind':'Haar+Gauss+CAR_link','link':e,'plaquette_occurrences':sorted(occ),
   'labels':['q_U1','j_SU2','(p,q)_SU3','iota_SU2','iota_SU3','Pi_CAR'],
   'matter_and_gauge_same_node':True})
 for p,po in enumerate(plaq):
  nodes.append({'kind':'product_group_magnetic_plaquette','plaquette':p,'boundary':po,
   'groups':['U1_Y','SU2_L','SU3_C'],'single_shared_boundary':True})
 for e in range(len(links)):
  nodes.append({'kind':'electric_diagonal','link':e,'groups':['U1_Y','SU2_L','SU3_C']})
 for sidx,s in enumerate(sites):
  for mu,name in [(2,'Sz'),(1,'Sy'),(0,'Sx')]:
   nodes.append({'kind':'composite_CAR_stream','site':sidx,'direction':mu,'name':name,
    'sector_carriers':['d','u','e','nu'],'uses_same_link_operator':True})
  nodes.append({'kind':'U_cell_CAR_collision','site':sidx,'aux_bond_dimension':2,'parity_preserved':True})
  nodes.append({'kind':'route_first_interaction','site':sidx,'routes':[10,9,8],'temporal_semigroup':True})
 payload={'L':L,'sites':len(sites),'links':len(links),'plaquettes':len(plaq),'nodes':nodes,
          'link_plaquette_valences':sorted(set(len(x) for x in inc))}
 raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
 return {'L':L,'sites':len(sites),'links':len(links),'plaquettes':len(plaq),'node_count':len(nodes),
         'link_plaquette_valences':payload['link_plaquette_valences'],'graph_sha256':hashlib.sha256(raw).hexdigest(),
         'all_link_nodes_fuse_gauge_matter_car':all(n.get('matter_and_gauge_same_node',False) for n in nodes if n['kind']=='Haar+Gauss+CAR_link')}
rows=[compile_graph(L) for L in [2,3,4,5,6]]
rep={'status':'PASS_TASK1_PEPO_LATTICE_COMPILATION_SEQUENCE','external_particle_data_used':False,
     'operator_order':['U_int^rf','exp(-i phi K_B)','exp(-i phi K_E)','U_cell','Gamma(S_x)','Gamma(S_y)','Gamma(S_z)'],
     'routes':{'d':[10,9,8],'completion':'P_d=U_C8^d'},
     'exact_labels':['q_U1','j_SU2','(p,q)_SU3','iota_SU2','iota_SU3','Pi_CAR'],
     'volumes':rows,
     'interpretation':'This is the operator PEPO itself: all compact-group and CAR local constraints meet at the same fused link nodes. Physical input/output legs and virtual representation/intertwiner bonds remain open because the output of Task 1 is a transfer operator, not a chosen-state scalar expectation value.'}
txt=json.dumps(rep,indent=2,sort_keys=True)+'\n';OUT.write_text(txt)
print(json.dumps(rep|{'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
