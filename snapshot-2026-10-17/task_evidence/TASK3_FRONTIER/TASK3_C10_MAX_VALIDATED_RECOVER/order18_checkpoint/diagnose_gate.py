import pickle,sys,time
sys.path.insert(0,'.')
import recoupled_engine_order18 as m
pid=int(sys.argv[1])
x=pickle.load(open('shortfirst_ck.pkl','rb')); s=x['state']
print('BEGIN pid',pid,'err',s.error_bound,flush=True)
s2,a=m.apply_gate_span_pruned(s,pid,12,2e-10,verbose=True)
print('APPLY_DONE',a,flush=True)
