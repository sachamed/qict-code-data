#!/usr/bin/env python3
"""QICT rank-six full-product-group tensor-network IR, revision 2.

This revision repairs the chiral gauge carrier: V_R,f is never streamed bare
through the Clebsch bridge.  The active right branch is V_R,f tensor R_f, where
R_f is H_R or Htilde_R.  By the exact intertwiner identity it carries the same
U(1)_Y x SU(2)_L x SU(3)_C representation as V_L,f.  The IR therefore preserves
Gauss labels through chirality conversion and includes color explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import argparse,json,math,hashlib
L=2; PHI=math.pi/63; ROUTES=(10,9,8); FAMILIES=('I','X','Z')
SECTORS={
 'd': {'left':(1,1,(1,0)),'right':(-2,0,(1,0)),'receiver':(+3,1,(0,0),'H_R')},
 'u': {'left':(1,1,(1,0)),'right':(+4,0,(1,0)),'receiver':(-3,1,(0,0),'Htilde_R')},
 'e': {'left':(-3,1,(0,0)),'right':(-6,0,(0,0)),'receiver':(+3,1,(0,0),'H_R')},
 'nu':{'left':(-3,1,(0,0)),'right':(0,0,(0,0)),'receiver':(-3,1,(0,0),'Htilde_R')},
}
@dataclass
class Leg: name:str; kind:str; sectors:dict
@dataclass
class Node: name:str; kind:str; legs:list[str]; metadata:dict

def geometry():
 s=[(x,y,z) for x in range(2) for y in range(2) for z in range(2)]
 links=[(mu,*a) for mu in range(3) for a in s]; li={t:i for i,t in enumerate(links)}; ps=[]
 for mu in range(3):
  for nu in range(mu+1,3):
   for a in s:
    a=list(a);am=a.copy();am[mu]^=1;an=a.copy();an[nu]^=1
    ps.append([(li[(mu,*a)],1),(li[(nu,*am)],1),(li[(mu,*an)],-1),(li[(nu,*a)],-1)])
 return s,links,ps

def compile_ir(nmom=10,u1_shift=6,su2_j2=12,su3_pq=15):
 sites,links,ps=geometry();legs=[];nodes=[]
 # rigorous worst-case support, intentionally conservative.
 Emax=nmom*(4*u1_shift+6)
 J2max=nmom*(4*su2_j2+1)
 PQmax=nmom*(4*su3_pq+1)
 for t in range(nmom+1):
  for e in range(len(links)):
   legs.append(Leg(f'g[{t},{e}]','full_gauge_time',{
    'q_U1_E':[-Emax,Emax], 'j_SU2_twice':[0,J2max],
    'pq_SU3':{'p':[0,PQmax],'q':[0,PQmax]},
    'iota_SU2':'exact fusion-channel multiplicity label',
    'iota_SU3':'exact Littlewood-Richardson/Haar multiplicity label',
    'Pi_CAR':[0,1]}))
 for t in range(nmom):
  for e in range(len(links)):
   nodes.append(Node(f'E[{t},{e}]','electric_diagonal',[f'g[{t},{e}]',f'gE[{t},{e}]'],{
    'U1':'exp[-i phi P_+(E)]','SU2':'exp[-i phi K_E(j)]','SU3':'exp[-i phi K_E(p,q)]','unitary':True}))
  for p,po in enumerate(ps):
   nodes.append(Node(f'B[{t},{p}]','product_group_plaquette_character_shift',
    [f'pY[{t},{p}]',f'p2[{t},{p}]',f'p3[{t},{p}]']+[f'gE[{t},{e}]' for e,_ in po]+[f'gB[{t},{e}]' for e,_ in po],{
      'boundary':po,
      'U1':{'k':[-u1_shift,u1_shift],'coefficient':'e^-i4phi i^k J_k(4phi)'},
      'SU2':{'twice_j':[0,su2_j2],'coefficient':'e^-i32phi i^(2j) 2(2j+1)J_(2j+1)(32phi)/(32phi)'},
      'SU3':{'max_p_q':su3_pq,'coefficient':'Weyl-character coefficient of exp[-i(4pi/7)(1-ReTr(U)/3)]'},
      'shared_link_bonds':True}))
  for mu,name in ((2,'Sz'),(1,'Sy'),(0,'Sx')):
   meta={'direction':mu,'spin_projectors':'(I +/- sigma_mu)/2','sector_carriers':{}}
   for f,s in SECTORS.items():
    yL,jL,cL=s['left']; yR,jR,cR=s['right']; yH,jH,cH,hname=s['receiver']
    assert (yR+yH,jH,cR)==(yL,jL,cL)
    meta['sector_carriers'][f]={
      'left':{'q':yL,'j2':jL,'pq':cL},
      'right_composite':{'bare_right':{'q':yR,'j2':jR,'pq':cR},'receiver':{'name':hname,'q':yH,'j2':jH,'pq':cH},'total':{'q':yL,'j2':jL,'pq':cL}},
      'transport_rule':'rho_R(g) tensor rho_receiver(g), Clebsch-equivalent to rho_L(g)'}
   nodes.append(Node(f'{name}[{t}]','C8_covariant_composite_stream',[],meta))
  nodes.append(Node(f'C[{t}]','chiral_Clebsch_intertwiner',[],{
    'maps':['d_R tensor H_R -> Q_L','u_R tensor Htilde_R -> Q_L','e_R tensor H_R -> L_L','nu_R tensor Htilde_R -> L_L'],
    'multiplicity':1,'gauge_label_change':0,'CAR_parity_change':0,'unitary_involution_on_active_pair':True}))
  for e in range(len(links)):
   nodes.append(Node(f'Haar[{t},{e}]','full_product_group_Gauss_Haar_intertwiner',[f'gB[{t},{e}]',f'g[{t+1},{e}]'],{
    'U1':'exact integer Gauss delta','SU2':'sum_iota2 |iota2><iota2|','SU3':'sum_iota3 |iota3><iota3|',
    'CAR':'exact Pi_CAR conservation','four_incident_plaquette_legs':True,'matter_and_receiver_share_same_Haar_node':True}))
 nodes += [Node('V6_in','rank_six_source',[],{'families':3,'chirality':['L','R'],'rank':6}),Node('V6_out','rank_six_sink',[],{'families':3,'chirality':['L','R'],'rank':6})]
 return {'schema_version':'QICT-R6TN-2','geometry':{'L':2,'sites':len(sites),'links':len(links),'plaquettes':len(ps)},
  'moments':{'nmax':nmom,'routes':list(ROUTES),'family_generators':list(FAMILIES)},
  'cutoffs':{'U1_abs_k':u1_shift,'SU2_twice_j_max':su2_j2,'SU3_max_p_q':su3_pq},
  'worst_case_vertical_support':{'U1_abs_E':Emax,'SU2_twice_j':J2max,'SU3_p_q_each':PQmax},
  'exact_labels':['q_U1','j_SU2','(p,q)_SU3','iota_SU2','iota_SU3','Pi_CAR'],
  'legs':[asdict(x) for x in legs],'nodes':[asdict(x) for x in nodes],
  'contraction_order':['character-expand all three plaquette factors','fuse plaquette and composite-matter reps at each common Haar link node','retain q,j,(p,q),iota,Pi blocks exactly','sweep spatial boundary with symmetry-preserving block SVD','apply electric diagonal gate on reached representation support','repeat time slices and read C_n^(6)'],
  'nonfactorization':'Gauge plaquette, electric, matter, receiver and CAR tensors meet on the same labelled link/Haar blocks before compression. Factorization is used only as the exact conditional product-group identity inside a fixed representation/worldline block.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--nmax',type=int,default=10);ap.add_argument('--out',default='');a=ap.parse_args()
 ir=compile_ir(a.nmax); txt=json.dumps(ir,indent=2)+'\n';
 out=Path(a.out) if a.out else Path(__file__).resolve().parents[2]/'results'/'microscopic_closure'/'RANK6_TENSOR_NETWORK_IR_V2.json';out.write_text(txt)
 print(json.dumps({'status':'PASS_FULL_PRODUCT_GROUP_TENSOR_IR_V2','nodes':len(ir['nodes']),'legs':len(ir['legs']),'cutoffs':ir['cutoffs'],'labels':ir['exact_labels'],'sha256':hashlib.sha256(txt.encode()).hexdigest()},indent=2))
if __name__=='__main__':main()
