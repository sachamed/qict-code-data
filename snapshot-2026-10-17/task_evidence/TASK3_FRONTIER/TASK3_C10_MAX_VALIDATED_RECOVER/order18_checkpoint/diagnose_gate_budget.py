import pickle,sys
sys.path.insert(0,'.'); import recoupled_engine_order18 as m
pid=int(sys.argv[1]); bud=float(sys.argv[2])
x=pickle.load(open('shortfirst_ck.pkl','rb')); s=x['state']
print('BEGIN',pid,bud,'err0',s.error_bound,flush=True)
s2,a=m.apply_gate_span_pruned(s,pid,12,bud,verbose=True)
print('DONE',a,flush=True)
