import numpy as np, json, os, hashlib
root=str(__import__('pathlib').Path(__file__).resolve().parents[2]/'data'); outdir=str(__import__('pathlib').Path(__file__).resolve().parents[2]/'results/boundary_readouts'); os.makedirs(outdir,exist_ok=True)
report={'status':'PASS_BOUNDARY_VECTOR_RECONSTRUCTION_AND_READOUT','definition':'Vacuum boundary labels are the minimal admissible active-carrier labels: left (0,0,0), right (0,0,1) in q=-3,2j=1. Retained vectors are ell=U^* e_L and r=V^*? equivalently Vh e_R; h_d=ell^* diag(s)^d r.','results':{}}
for cutoff in ['COARSE','FINE']:
 f=f'{root}/binding/{cutoff}_qm3_j1_SVD.npz'; z=np.load(f)
 L=z['left_labels']; R=z['right_labels']; U=z['U']; s=z['s'].astype(complex); Vh=z['Vh']
 li=np.where(np.all(L==[0,0,0],axis=1))[0]; ri=np.where(np.all(R==[0,0,1],axis=1))[0]
 assert len(li)==1 and len(ri)==1,(li,ri)
 li=int(li[0]); ri=int(ri[0])
 ell=U.conj().T[:,] @ np.eye(U.shape[0],dtype=complex)[:,li] # U^* eL
 rr=Vh @ np.eye(Vh.shape[1],dtype=complex)[:,ri] # Vh eR
 # check d1 against ambient M element without materializing M
 h1=np.vdot(ell,s*rr); direct=np.dot(U[li,:]*s,Vh[:,ri])
 assert abs(h1-direct)<1e-15
 hs={}
 for d in [8,9,10]: hs[str(d)]=complex(np.vdot(ell,(s**d)*rr))
 np.savez_compressed(f'{outdir}/{cutoff}_qm3_j1_BOUNDARY_VECTOR.npz',ell=ell,r=rr,left_index=li,right_index=ri,left_label=L[li],right_label=R[ri])
 report['results'][cutoff]={'left_index':li,'right_index':ri,'left_label':L[li].tolist(),'right_label':R[ri].tolist(),'ell_norm':float(np.linalg.norm(ell)),'r_norm':float(np.linalg.norm(rr)),'h1_direct_residual':float(abs(h1-direct)),'h':{k:[v.real,v.imag] for k,v in hs.items()}}
# convergence differences raw
for d in ['8','9','10']:
 c=complex(*report['results']['COARSE']['h'][d]); f=complex(*report['results']['FINE']['h'][d]); report.setdefault('coarse_fine_abs_difference',{})[d]=abs(f-c)
json.dump(report,open(f'{outdir}/BOUNDARY_VECTOR_READOUT_REPORT.json','w'),indent=2)
print(json.dumps(report,indent=2))
