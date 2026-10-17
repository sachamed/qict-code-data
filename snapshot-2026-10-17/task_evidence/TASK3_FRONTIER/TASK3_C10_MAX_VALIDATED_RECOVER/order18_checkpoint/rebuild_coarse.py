import sys, json, pickle, time, os
sys.path.insert(0,'/mnt/data/task3_order18')
import recoupled_engine_order18 as m
pids=[0,1,14,23,19,6,7,12,20,16]
state=m.vacuum_mps(); reports=[]
for pid in pids:
 t=time.time()
 state,a=m.apply_gate_span_pruned(state,pid,12,2e-10,verbose=False)
 state,c=m.right_compress(state,2e-9,verbose=False)
 reports.append({'pid':pid,'apply':a,'canon':c,'seconds':time.time()-t})
 print(json.dumps({'pid':pid,'apply_disc':a['discarded_l2'],'canon_disc':c['discarded_l2'],'maxa':a['max_bond'],'maxc':c['max_bond'],'state_err':state.error_bound,'seconds':reports[-1]['seconds']}),flush=True)
pickle.dump({'state':state,'done':pids,'reports':reports},open('/mnt/data/task3_order18/coarse_order18_ck.pkl','wb'),protocol=4)
print(json.dumps({'DONE':True,'error_bound':state.error_bound,'maxbond':max(map(len,state.bond_sectors)),'scale_abs':abs(state.scale)}))
for pid in range(24):
 rel=m.gate_relevant_sites(pid)
 print('SPAN',pid,rel[0],rel[-1],len(rel),rel[-1]-rel[0]+1,rel)
