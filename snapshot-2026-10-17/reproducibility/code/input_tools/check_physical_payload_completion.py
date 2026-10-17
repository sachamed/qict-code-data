#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, hashlib

REQ_KATO={'rho','rho_gram_factor','buffer','uniform_contour_gap_lower_bound','riesz_residual_upper_bound'}
SOURCE_ORDER=['L1','L2','L3','R1','R2','R3']

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 root=Path(__file__).resolve().parents[1]; p=root/'physical/L24'
 checks={}; details={}
 bufs=sorted(p.glob('KATO_BUFFER_*.npz'))
 valid_buf=[]
 for f in bufs:
  try:
   z=np.load(f,allow_pickle=False); miss=REQ_KATO-set(z.files)
   rho=np.asarray(z['rho'],complex) if not miss else None
   gram=np.asarray(z['rho_gram_factor'],complex) if not miss else None
   ok=(not miss and rho.ndim==2 and rho.shape[0]==rho.shape[1] and gram.ndim==2 and gram.shape[0]==rho.shape[0]
       and float(z['uniform_contour_gap_lower_bound'])>0 and float(z['riesz_residual_upper_bound'])>=0)
   if ok: valid_buf.append(f)
   details[f.name]={'valid':bool(ok),'missing_fields':sorted(miss),'sha256':sha(f)}
  except Exception as e: details[f.name]={'valid':False,'error':str(e)}
 checks['four_valid_nested_kato_buffers']=len(valid_buf)>=4
 cert=p/'BUFFER_CONTRACTION_CERTIFICATE.json'; checks['buffer_tail_certificate']=cert.exists()
 shell=sorted(p.glob('C10_SHELL_MAPS_*.npz')); checks['quantum_c10_shell_maps']=len(shell)>0
 ins=sorted(p.glob('BOUNDARY_TO_PATH_*.npz')); checks['boundary_to_path_tensor']=len(ins)>0
 core=p/'CORE_RECOUPLING.npz'; checks['core_recoupling_payload']=core.exists()
 emb=p/'RANK6_EMBEDDING.npz'; checks['rank6_embedding']=emb.exists()
 provider=p/'GLOBAL_CHARGED_UCONE_PROVIDER.py'; marker=p/'GLOBAL_CHARGED_UCONE_CERTIFICATE.json'
 checks['complete_charged_ucone_matvec']=provider.exists() and marker.exists()
 moments=p/'CN6_MOMENTS.npz'; checks['cn6_moments']=moments.exists()
 if moments.exists():
  try:
   z=np.load(moments,allow_pickle=False); C=np.asarray(z['moments'],complex)
   checks['cn6_shape_and_C0']=bool(C.ndim==3 and C.shape[1:]==(6,6) and C.shape[0]>=10 and np.linalg.norm(C[0]-np.eye(6))<=1e-8)
   details['CN6_MOMENTS.npz']={'shape':list(C.shape),'C0_fro_error':float(np.linalg.norm(C[0]-np.eye(6))),'sha256':sha(moments)}
  except Exception as e: checks['cn6_shape_and_C0']=False; details['CN6_MOMENTS.npz']={'error':str(e)}
 else: checks['cn6_shape_and_C0']=False
 ready=all(checks.values())
 out={'status':'PASS_L24_PHYSICAL_RANK6_PAYLOAD_COMPLETE' if ready else 'L24_PHYSICAL_RANK6_PAYLOAD_INCOMPLETE',
      'ready_for_physical_C9_and_G2':ready,'source_order':SOURCE_ORDER,'checks':checks,'details':details,
      'missing':[k for k,v in checks.items() if not v]}
 (root/'L24_PHYSICAL_PAYLOAD_COMPLETION_STATUS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
