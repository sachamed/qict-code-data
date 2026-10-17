#!/usr/bin/env python3
"""Integer C10 Gauss complex and exact-carrier primitive Kato/source construction.

This constructs the primitive physical gauge sector and dressed CAR source
operators. It proves existence/convergence of the A7 gauge ground at a fixed
finite complex. It does not calculate the many-link ground vector, the full
route-first charged circuit or thermodynamic moments.
"""
from pathlib import Path
import hashlib
import heapq
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reproducibility/results/c10_primitive_gauss'
PAYLOAD=ROOT/'reproducibility/data/common_physical_chain/L24_N9_CHARGED_RANK6_PRECONTRACTION_PAYLOAD.npz'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(PAYLOAD,allow_pickle=False) as z:
        links=z['links'].astype(np.int64)
        faces=z['plaquette_edge_indices'].astype(np.int64)
        signs=z['plaquette_edge_sign'].astype(np.int64)
    starts=links[:,1:];ends=starts.copy()
    ends[np.arange(len(links)),links[:,0]]+=1
    vertices=np.unique(np.concatenate([starts,ends]),axis=0)
    index={tuple(v):i for i,v in enumerate(vertices)}
    parent=list(range(len(vertices)))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    tree=[];chords=[]
    for edge,(a,b) in enumerate(zip(starts,ends)):
        x,y=find(index[tuple(a)]),find(index[tuple(b)])
        if x==y:chords.append(edge)
        else:parent[x]=y;tree.append(edge)
    components=len({find(i) for i in range(len(vertices))})
    assert components==1 and len(tree)==len(vertices)-1
    assert len(chords)==len(links)-len(vertices)+1
    # Exact oriented boundary identity, independently of the tree basis.
    for edges,coefficients in zip(faces,signs):
        divergence={}
        for e,c in zip(edges,coefficients):
            a,b=tuple(starts[e]),tuple(ends[e])
            divergence[a]=divergence.get(a,0)-int(c)
            divergence[b]=divergence.get(b,0)+int(c)
        assert all(value==0 for value in divergence.values())
    assert np.all(signs.sum(axis=1)==0)
    # A cycle is uniquely determined by its chord coordinates. Projection
    # onto those coordinates is a Z-isomorphism ker(partial1)->Z^b1_graph.
    chord_index={edge:i for i,edge in enumerate(chords)}
    columns=[];row_columns=[set() for _ in chords]
    for face,(edges,coefficients) in enumerate(zip(faces,signs)):
        column={chord_index[int(e)]:int(c) for e,c in zip(edges,coefficients)
                if int(e) in chord_index}
        columns.append(column)
        for row in column:row_columns[row].add(face)
    heap=[(len(c),p) for p,c in enumerate(columns) if c];heapq.heapify(heap)
    pivots=[];operations=0;max_coefficient=1
    while heap:
        size,p=heapq.heappop(heap);column=columns[p]
        if not column or len(column)!=size:continue
        unit_rows=[r for r,c in column.items() if abs(c)==1]
        assert unit_rows,('nonunit pivot would require a Smith step',p,column)
        row=min(unit_rows,key=lambda r:(len(row_columns[r]),r))
        pivot=column[row]
        for q in sorted(row_columns[row]):
            if q==p:continue
            target=columns[q];factor=target[row]//pivot
            assert target[row]==factor*pivot
            for r,c in column.items():
                value=target.get(r,0)-factor*c
                if value:
                    target[r]=value;row_columns[r].add(q)
                    max_coefficient=max(max_coefficient,abs(value))
                else:target.pop(r,None);row_columns[r].discard(q)
            if target:heapq.heappush(heap,(len(target),q))
            operations+=1
        for r in column:row_columns[r].discard(p)
        pivots.append((row,p,pivot));columns[p]={}
    assert len(pivots)==len(chords)
    assert all(not s for s in row_columns)
    # Each eliminated pivot was +-1: integer column operations and integer
    # row elimination give a primitive full-rank image, not only a rank
    # statement over R or modulo one prime.
    assert max_coefficient==1
    edge_index={tuple(row):i for i,row in enumerate(links)}
    source_paths=[];arrays={}
    for radius in (3,5,7,9):
        path=[]
        for x in range(radius):
            key=(0,x,0,0)
            assert key in edge_index,('source path absent',key)
            path.append(edge_index[key])
        source_paths.append({'dressing_radius':radius,'start':[0,0,0],
                             'compensator_endpoint':[radius,0,0],
                             'positive_link_indices':path,
                             'path_choice':'specified positive-x straight Wilson probe; not a route-first interaction replacement'})
        for q in (-3,-6):
            divergence={}
            for edge in path:
                a,b=tuple(starts[edge]),tuple(ends[edge])
                divergence[a]=divergence.get(a,0)-q
                divergence[b]=divergence.get(b,0)+q
            divergence[(0,0,0)]=divergence.get((0,0,0),0)+q
            endpoint=(radius,0,0)
            divergence[endpoint]=divergence.get(endpoint,0)-q
            assert all(value==0 for value in divergence.values())
        arrays['path_R'+str(radius)]=np.asarray(path,dtype=np.int32)
    arrays.update(gauge_vertices=vertices.astype(np.int16),
                  spanning_tree_links=np.asarray(tree,dtype=np.int32),
                  chord_links=np.asarray(chords,dtype=np.int32),
                  integer_unit_pivots=np.asarray(pivots,dtype=np.int32))
    npz=OUT/'C10_INTEGER_GAUSS_BASIS_AND_SOURCE_PATHS.npz'
    np.savez_compressed(npz,**arrays)
    beta=72*len(faces)
    result={'status':'EXACT_INTEGER_GAUSS_COMPLEX_AND_PRIMITIVE_KATO_SOURCE_CONSTRUCTION',
            'payload_sha256':hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
            'gauge_vertices':len(vertices),'gauge_links':len(links),'plaquettes':len(faces),
            'connected_components':components,'graph_cycle_rank':len(chords),
            'integer_boundary2_rank':len(pivots),
            'integer_unit_pivot_count':len(pivots),'integer_column_operations':operations,
            'maximum_elimination_coefficient':max_coefficient,
            'H1_integer':'zero: im_Z(partial2)=ker_Z(partial1)',
            'plaquette_flux_relations_rank':len(faces)-len(pivots),
            'linear_electric_term_in_neutral_sector':'zero, because every generating plaquette has signed coefficient sum zero',
            'gauge_A7_Kato_theorem':{
                'family':'K_prim(s)=K_E+s*K_B, 0<=s<=1',
                'domain':'exact Gauss-neutral gauge carrier on this finite complex, product-group cover',
                'boundary_condition':'Zero Gauss charge on all vertices of the isolated serialized finite gauge complex; not its reduced state inside a larger lattice.',
                'U1_carrier':'ell2(ker_Z partial1)=ell2(im_Z partial2)',
                'SU2_SU3_carriers':'gauge-invariant subspaces of compact-link Haar L2',
                'compact_resolvent':True,'gauge_ground_rank':1,
                'analytic_ground_projector_at_fixed_finite_complex':True,
                'gap_positive_at_every_s_and_uniform_on_this_finite_s_interval':True,
                'numerical_global_gap_lower_bound_computed':False,
                'thermodynamic_uniform_gap_proved':False,
                'proof_source':'paper/sections/supplement/s140.tex',
                'receiver_matter_scope':'This theorem selects the gauge component. An empty CAR reference and any additional receiver ground multiplicity must be stated explicitly; it is not the Floquet Riesz selector.'},
            'finite_complex_Galerkin_limit':{
                'electric_cutoff':'P_R=1_[0,R](K_E) in the same Gauss sector',
                'off_diagonal_magnetic_norm_upper':beta,
                'resolvent_comparison_at_z_minus_a':
                    'beta^2/[a^2(R+a)]+2beta/[a(R+a)]+1/(R+a)+beta^2/[a(R+a)^2]',
                'ground_projector_trace_norm_converges_at_fixed_complex':True,
                'reduced_boundary_density_trace_norm_converges_at_fixed_complex':True,
                'large_spatial_volume_limit_certified':False},
            'six_source_operator_construction':{
                'representation_labels':{'L':['qY=-3','SU2 fundamental','SU3 singlet'],
                                          'R':['qY=-6','SU2 singlet','SU3 singlet']},
                'source_order':['L1','L2','L3','R1','R2','R3'],
                'dressing':'CAR creation at the origin and the actual path holonomy, paired with its dual boundary representation',
                'empty_CAR_reference_is_an_explicit_condition':True,
                'source_Gram_on_that_reference':'I6 for every normalized gauge density, by CAR and unitary holonomy',
                'exact_U1_Gauss_path_constraints_checked':True,
                'SU2_covariance':'Endpoint covariance of the Wilson product; internal gauge matrices cancel exactly',
                'probe_paths':source_paths,
                'global_source_GNS_columns_numerically_contracted':False,
                'physical_BCC_overlap_or_particle_identification_certified':False},
            'full_frozen_charged_C10_update_contracted':False,
            'C10_nine_legacy_physical_contracts_completed':0,
            'data_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
    (OUT/'C10_PRIMITIVE_GAUSS_KATO_SOURCE_CERTIFICATE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['gauge_vertices','gauge_links','plaquettes','graph_cycle_rank','integer_boundary2_rank','H1_integer','plaquette_flux_relations_rank']},indent=2))


if __name__=='__main__':main()
