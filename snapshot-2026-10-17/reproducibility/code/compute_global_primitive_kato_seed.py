#!/usr/bin/env python3
"""Exact full-support A7 gauge Kato coefficients at the electric seed.

No endpoint extrapolation is made. The coefficients use every serialized
plaquette, all three compact groups and the frozen cost normalization.
"""
from fractions import Fraction as F
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PAYLOAD=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'
OUT=ROOT/'reproducibility/results/global_primitive_seed'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(PAYLOAD,allow_pickle=False) as z:
        links=z['links'].astype(int);faces=z['plaquette_edge_indices'].astype(int)
        signs=z['plaquette_edge_sign'].astype(int)
    assert len(set(map(tuple,links)))==len(links)
    keys=[]
    for edges,sgn in zip(faces,signs):
        assert len(set(map(int,edges)))==4 and int(sgn.sum())==0
        key=tuple(sorted(zip(map(int,edges),map(int,sgn))))
        inverse=tuple((e,-s) for e,s in key)
        keys.append(min(key,inverse))
    assert len(set(keys))==len(keys)
    incident=defaultdict(list)
    for p,edges in enumerate(faces):
        for edge in edges:incident[int(edge)].append(p)
    intersections=Counter(pair for face_list in incident.values()
                          for pair in combinations(face_list,2))
    assert max(intersections.values())==1
    starts=links[:,1:];ends=starts.copy();ends[np.arange(len(links)),links[:,0]]+=1
    assert np.all((np.sum(starts,axis=1)-np.sum(ends,axis=1))%2==1)
    # The simple bipartite graph has no one-, two- or three-edge cycles;
    # actual four-edge plaquettes attain its girth. A nontrivial neutral
    # spin network needs at least one cycle or at least four active edges.
    per_group={
      'U1':{'electric_excitation':8,'magnetic_amplitude':-2,'orientations':2,'first_vector_coefficient':F(1,4)},
      'SU2':{'electric_excitation':12,'magnetic_amplitude':-16,'orientations':1,'first_vector_coefficient':F(4,3)},
      'SU3':{'electric_excitation':16,'magnetic_amplitude':-6,'orientations':2,'first_vector_coefficient':F(3,8)}}
    first_norm=sum(v['orientations']*v['first_vector_coefficient']**2 for v in per_group.values())
    second=-sum(F(v['orientations']*v['magnetic_amplitude']**2,v['electric_excitation']) for v in per_group.values())
    # SU3: two paths 3 -> bar3 -> vacuum and bar3 -> 3 -> vacuum.
    third=F(2*(-6)**3,16**2)
    assert first_norm==F(629,288) and second==-F(161,6) and third==-F(27,16)
    count=len(faces);beta=72*count
    fraction=F(1,1000);s=fraction/beta
    remainder=fraction**4/(1-fraction)
    energy=F(beta)*s+count*second*s*s+count*third*s**3
    rows=[]
    for p in range(count):
        # Codes 0,1: U1 orientations; 2: weak fundamental; 3,4: color
        # fundamental/conjugate. These are normalized gauge-singlet loops.
        for code,num,den,electric in [(0,1,4,8),(1,1,4,8),(2,4,3,12),
                                      (3,3,8,16),(4,3,8,16)]:
            rows.append((p,code,num,den,electric))
    npz=OUT/'GLOBAL_A7_FIRST_DERIVATIVE_LOOP_DATA.npz'
    np.savez_compressed(npz,loop_amplitude_records=np.asarray(rows,np.int32),
                        plaquette_edge_indices=faces,plaquette_edge_sign=signs)
    result={'status':'EXACT_GLOBAL_PRIMITIVE_GAUGE_KATO_SEED_COEFFICIENTS',
      'payload_sha256':hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
      'plaquettes_used':count,'gauge_links':len(links),
      'distinct_oriented_plaquette_cycles':len(set(keys)),
      'maximum_shared_edges_between_distinct_plaquettes':max(intersections.values()),
      'graph_simple_and_bipartite':True,'graph_girth':4,
      'neutral_electric_ground_gap':{'U1':8,'SU2':12,'SU3':16,'all_groups':8},
      'homotopy':'K_prim(s)=K_E+s*K_B, with physical endpoint s=1',
      'energy_Taylor_coefficients':{'linear':beta,'quadratic_rational':str(count*second),
                                    'cubic_rational':str(count*third)},
      'energy_per_plaquette':'72*s-(161/6)*s^2-(27/16)*s^3+O(s^4)',
      'ground_vector_first_derivative':'Sum over every plaquette: (Y_++Y_-)/4 + (4/3)*chi_1/2 + (3/8)*(chi_3+chi_bar3).',
      'ground_vector_derivative_norm_squared_rational':str(count*first_norm),
      'normalized_single_loop_records':len(rows),
      'nonconstant_magnetic_operator_norm_upper':beta,
      'analytic_energy_disk_radius_rational':str(F(1,beta)),
      'Cauchy_remainder_after_cubic':'(beta*|s|)^4/(1-beta*|s|), for beta*|s|<1',
      'small_homotopy_point':{'s_rational':str(s),'cubic_energy_rational':str(energy),
                              'remainder_upper_rational':str(remainder),
                              'remainder_upper':float(remainder)},
      'physical_endpoint_ground_computed':False,
      'full_matter_receiver_state_computed':False,
      'spatial_limit_certified':False,
      'scope':'Exact gauge-ground derivatives on the entire isolated serialized gauge support near the electric seed. No extrapolation to s=1 or replacement of the full ordered charged circuit.',
      'data_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
    (OUT/'GLOBAL_A7_KATO_SEED_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'plaquettes':count,'loop_derivative_records':len(rows),
                      'E1':beta,'E2':str(count*second),'E3':str(count*third),
                      'norm_derivative_squared':str(count*first_norm)},indent=2))


if __name__=='__main__':main()
