import pickle,sys,time,os,json
sys.path.insert(0,'.')
import recoupled_engine_order18 as m
src=sys.argv[1]; pid=int(sys.argv[2]); out=sys.argv[3]; report=sys.argv[4]
x=pickle.load(open(src,'rb')); s=x['state']; done=list(x['done']); reps=list(x['reports'])
assert pid not in done
t=time.time(); s,a=m.apply_gate_span_pruned(s,pid,12,2e-10,verbose=True); s,c=m.right_compress(s,2e-9,verbose=False)
row={'pid':pid,'apply':a,'canon':c,'seconds':time.time()-t}; reps.append(row); done.append(pid)
tmp=out+'.tmp'; pickle.dump({'state':s,'done':done,'reports':reps},open(tmp,'wb'),protocol=4); os.replace(tmp,out)
r={'status':'RUNNING_ADAPTIVE_ORDER18','done':done,'cum_error':s.error_bound,'maxbond':max(map(len,s.bond_sectors)),'reports':reps}
json.dump(r,open(report+'.tmp','w')); os.replace(report+'.tmp',report)
print('FINAL',json.dumps({'n':len(done),'pid':pid,'apply_disc':a['discarded_l2'],'canon_disc':c['discarded_l2'],'maxa':a['max_bond'],'maxc':c['max_bond'],'err':s.error_bound,'sec':row['seconds']}),flush=True)
