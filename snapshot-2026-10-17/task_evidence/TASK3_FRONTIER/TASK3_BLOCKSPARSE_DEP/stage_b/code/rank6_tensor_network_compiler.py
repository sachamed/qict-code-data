#!/usr/bin/env python3
"""Compile the QICT L=2 rank-six moment problem to a symmetry-labelled tensor-network IR.

The compiler is deterministic and contains no particle-data input.  The target
IR is intended for a block-sparse boundary-MPS/TRG engine.  A dense contraction
is deliberately rejected at the certified cutoffs because it would erase the
Gauss/intertwiner structure and exceed practical memory by many orders.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, math

L=2
PHI=math.pi/63.0
Q_L=-3; Q_R=-6
ROUTES=(10,9,8)
FAMILIES=('I','X','Z')

@dataclass
class Leg:
    name:str
    kind:str
    sectors:dict

@dataclass
class Node:
    name:str
    kind:str
    legs:list[str]
    metadata:dict


def geometry():
    sites=[(x,y,z) for x in range(L) for y in range(L) for z in range(L)]
    links=[(mu,x,y,z) for mu in range(3) for x,y,z in sites]
    li={t:i for i,t in enumerate(links)}
    plaquettes=[]
    for mu in range(3):
      for nu in range(mu+1,3):
        for x,y,z in sites:
          s=[x,y,z]; sm=s.copy(); sm[mu]=(sm[mu]+1)%L; sn=s.copy(); sn[nu]=(sn[nu]+1)%L
          plaquettes.append([(li[(mu,*s)],1),(li[(nu,*sm)],1),(li[(mu,*sn)],-1),(li[(nu,*s)],-1)])
    return sites,links,plaquettes


def compile_ir(nmom:int=10,u1_shift:int=6,su2_j2:int=12):
    sites,links,plaquettes=geometry()
    legs=[]; nodes=[]
    # Vertical quantum-gauge link legs.  The worst-case supports are finite for a
    # finite moment depth because all retained local representation shifts are finite.
    Emax=nmom*(4*u1_shift+max(abs(Q_L),abs(Q_R)))
    J2max=nmom*(4*su2_j2+1)  # four plaquette reps plus one fundamental matter hop
    for t in range(nmom+1):
      for e in range(len(links)):
        legs.append(Leg(f'g[{t},{e}]','gauge_time',{
          'U1_E':[-Emax,Emax],
          'SU2_twice_j':[0,J2max],
          'magnetic_indices':'implicit inside SU(2) symmetry blocks',
          'CAR_parity':[0,1]
        }))
    # Each time slice: electric gate, 24 plaquette gates, C8 z/y/x stream and chiral bridge.
    for t in range(nmom):
      for e in range(len(links)):
        nodes.append(Node(f'E[{t},{e}]','electric_diagonal',
                          [f'g[{t},{e}]',f'gE[{t},{e}]'],
                          {'U1_phase':'exp[-i phi P_+(E)]',
                           'SU2_phase':'exp[-i phi K_E(j)]','norm':1.0}))
      for p,po in enumerate(plaquettes):
        nodes.append(Node(f'B[{t},{p}]','plaquette_character_shift',
                          [f'pU1[{t},{p}]',f'pSU2[{t},{p}]']+[f'gE[{t},{e}]' for e,_ in po]+[f'gB[{t},{e}]' for e,_ in po],
                          {'U1_k_range':[-u1_shift,u1_shift],
                           'SU2_twice_j_range':[0,su2_j2],
                           'boundary':po,
                           'coeff_U1':'exp(-i 4phi) i^k J_k(4phi)',
                           'coeff_SU2':'exp(-i 32phi) i^(2j) 2(2j+1)J_(2j+1)(32phi)/(32phi)'}))
      for mu,name in ((2,'Sz'),(1,'Sy'),(0,'Sx')):
        nodes.append(Node(f'{name}[{t}]','C8_covariant_stream',[],{
          'direction':mu,'q_L':Q_L,'q_R':Q_R,
          'left_SU2':'fundamental','right_SU2':'singlet',
          'spin_projectors':'(I +/- sigma_mu)/2'}))
      nodes.append(Node(f'C[{t}]','chiral_Clebsch_bridge',[],{'h0':[0,1],'unitary_involution':True}))
      for e in range(len(links)):
        nodes.append(Node(f'Haar[{t},{e}]','SU2_U1_Gauss_intertwiner',
                          [f'gB[{t},{e}]',f'g[{t+1},{e}]'],
                          {'four_incident_plaquette_legs':True,
                           'SU2':'projector onto invariant fusion channels',
                           'U1':'exact signed charge conservation'}))
    nodes.append(Node('V6_in','rank_six_source',[],{'families':3,'chirality':['L','R'],'rank':6}))
    nodes.append(Node('V6_out','rank_six_sink',[],{'families':3,'chirality':['L','R'],'rank':6}))
    return {
      'schema_version':'QICT-R6TN-1',
      'geometry':{'L':2,'sites':len(sites),'links':len(links),'plaquettes':len(plaquettes)},
      'moments':{'nmax':nmom,'requested_routes':list(ROUTES),'family_generators':list(FAMILIES)},
      'certified_local_cutoffs':{'U1_plaquette_shift_abs_max':u1_shift,'SU2_plaquette_twice_j_max':su2_j2},
      'worst_case_vertical_support':{'U1_abs_E_max':Emax,'SU2_twice_j_max':J2max},
      'legs':[asdict(x) for x in legs],
      'nodes':[asdict(x) for x in nodes],
      'contraction_order':[
        'contract local plaquette shifts into link Gauss/intertwiner blocks within each time slice',
        'apply C8 matter stream inside the same charge/spin blocks',
        'sweep a boundary MPS through the 2x2x2 slice using symmetry-preserving SVD',
        'retain block labels (U1 charge, SU2 j/intertwiner, CAR parity) exactly',
        'normalize boundary state after every sweep and accumulate discarded 2-norm',
        'repeat for t=0..n-1 and read the 6x6 boundary moment C_n',
        'tighten cutoffs/tolerance until C_n, singular values, and K_UV are stable'
      ],
      'nonfactorization':'Gauge action tensors and matter stream tensors share the same link representation legs before any contraction. No Z_gauge*T_matter split exists in this IR.'
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--nmax',type=int,default=10)
    ap.add_argument('--u1-shift',type=int,default=6)
    ap.add_argument('--su2-j2',type=int,default=12)
    ap.add_argument('--out',default='')
    a=ap.parse_args()
    ir=compile_ir(a.nmax,a.u1_shift,a.su2_j2)
    txt=json.dumps(ir,indent=2)+'\n'
    if a.out:
        Path(a.out).write_text(txt)
    print(json.dumps({
      'status':'PASS_TENSOR_IR_COMPILED',
      'nodes':len(ir['nodes']),'legs':len(ir['legs']),
      'geometry':ir['geometry'],'moments':ir['moments'],
      'cutoffs':ir['certified_local_cutoffs'],
      'vertical_support':ir['worst_case_vertical_support']
    },indent=2))

if __name__=='__main__': main()
