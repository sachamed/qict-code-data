#!/usr/bin/env python3
"""Evaluate the coherent recursion on the archived L=24 geometric flow.

Identity link callbacks test graph transport only. They are explicitly not the
interacting quantum links. All diagnostics below use binary64 arithmetic.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from coherent_path_dag import CoherentPathDAG,Edge

def evaluate(path):
 with np.load(path,allow_pickle=False) as z:
  nodes=z['c10_nodes'];potential=z['c10_potential']
  src=z['c10_edge_src'];dst=z['c10_edge_dst']
  p=z['c10_edge_probability'];archived=z['c10_edge_flow']
  L=int(z['L']);n=int(z['n'])
 center=np.flatnonzero(np.all(nodes==0,axis=1))
 if len(center)!=1:raise ValueError('Expected one central injection node')
 source=int(center[0]);sink=-1
 order=[int(i) for i in np.argsort(-potential)]+[sink]
 identity=lambda x:x
 edges=[Edge(int(a),int(b),float(w),identity,identity,1.0,1.0) for a,b,w in zip(src,dst,p)]
 dag=CoherentPathDAG(order,sink,edges)
 flow=dag.flow_weights(source)
 identity_defect=float(np.linalg.norm(dag.apply(source,np.ones(1))-1))
 adjoint_defect=float(np.linalg.norm(dag.apply_adjoint(source,np.ones(1))-1))
 discrepancy=float(np.max(abs(flow-archived)))
 balance=np.bincount(src,weights=archived,minlength=len(nodes))-np.bincount(dst[dst>=0],weights=archived[dst>=0],minlength=len(nodes))
 balance[source]-=1
 budget=dag.error_diagnostic(source)
 assert identity_defect<1e-11 and adjoint_defect<1e-11
 assert discrepancy<1e-10 and np.max(abs(balance))<1e-10
 return dict(classification='ARCHIVED_C10_GEOMETRY_FLOATING_POINT_VALIDATION',
  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),L=L,n=n,
  interior_nodes=len(nodes),boundary_sink_nodes=1,directed_edges=len(edges),
  expected_path_length=budget['expected_path_length'],
  uniform_edge_error_amplification=budget['operator_error_bound_excluding_roundoff'],
  outgoing_probability_normalization_residual=dag.normalization_residual,
  maximum_archived_flow_discrepancy=discrepancy,
  maximum_archived_conservation_residual=float(np.max(abs(balance))),
  identity_transport_residual=identity_defect,identity_adjoint_residual=adjoint_defect,
  boundary_absorption_probability=float(flow[dst<0].sum()),
  interpretation='For exact normalized probabilities and true and approximate link contractions with a common certified error epsilon, the analytic bound is expected_path_length times epsilon. This file evaluates the archived geometry only; its decimal values do not enclose floating-point error.',
  physical_interacting_links_supplied=False,physical_C10_certified=False,
  roundoff_certified=False,particle_prediction_claimed=False)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--payload',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
 a=ap.parse_args();r=evaluate(a.payload);a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
