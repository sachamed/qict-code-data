import sys,json,pickle,time,os
sys.path.insert(0,'/mnt/data/task3_order18'); import recoupled_engine_order18 as m
pids=[1,12,8,15,19,13,23,0,10,14,7,16,20,6,9,5,18,22,11,4,2,3,17,21]
out='/mnt/data/task3_order18/shortfirst_ck.pkl'; report='/mnt/data/task3_order18/shortfirst_report.json'
if os.path.exists(out):
 x=pickle.load(open(out,'rb')); state=x['state']; done=x['done']; reps=x['reports']
else: state=m.vacuum_mps(); done=[]; reps=[]
for pid in pids[len(done):]:
 t=time.time(); state,a=m.apply_gate_span_pruned(state,pid,12,2e-10,verbose=False); state,c=m.right_compress(state,2e-9,verbose=False)
 row={'pid':pid,'apply':a,'canon':c,'seconds':time.time()-t}; reps.append(row);done.append(pid)
 pickle.dump({'state':state,'done':done,'reports':reps},open(out+'.tmp','wb'),protocol=4);os.replace(out+'.tmp',out)
 json.dump({'status':'RUNNING','done':done,'cum_error':state.error_bound,'maxbond':max(map(len,state.bond_sectors)),'reports':reps},open(report+'.tmp','w'));os.replace(report+'.tmp',report)
 print(json.dumps({'n':len(done),'pid':pid,'span':m.gate_relevant_sites(pid)[-1]-m.gate_relevant_sites(pid)[0]+1,'apply_disc':a['discarded_l2'],'canon_disc':c['discarded_l2'],'maxa':a['max_bond'],'maxc':c['max_bond'],'err':state.error_bound,'sec':row['seconds']}),flush=True)
