#!/usr/bin/env python3
import json, math, hashlib, os
base_path='/mnt/data/qict_task1_v5/SIXBLOCK_STREAM_N4_REFINED.json'
files={
 'N4_refined':base_path,
 'N4_N5':'/mnt/data/qict_task1_v5/N4_N5_INCREMENT_BOUND_FAST.json',
 'N5_N6':'/mnt/data/qict_task1_v5/N5_N6_INCREMENT_BOUND_FAST.json',
 'N6_N7':'/mnt/data/qict_task1_v5/N6_N7_INCREMENT_BOUND_FAST.json',
 'N7_N8':'/mnt/data/qict_task1_v5/N7_N8_INCREMENT_BOUND_FAST.json',
 'N8_N9':'/mnt/data/qict_task1_v5/N8_N9_INCREMENT_BOUND_FAST.json',
 'N9_inf':'/mnt/data/qict_task1_v5/BESSEL_INFINITY_TAIL_CERTIFICATE.json',
}
def load(p):
 with open(p) as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
base=load(files['N4_refined'])
z=complex(*base['Z2_N4']); r=abs(z); L=-math.log(r)
components=[]
components.append(('N4 omitted-sector bound', float(base['omitted_sector_abs_bound'])))
for key in ['N4_N5','N5_N6','N6_N7','N7_N8','N8_N9']:
 d=load(files[key]); components.append((f"{d['N0']}->{d['N1']} validated increment",float(d['validated_global_abs_Z_increment_bound'])))
tail=load(files['N9_inf'])
row9=next(rw for rw in tail['rows'] if rw['N']==9)
components.append(('N9->infinity analytic Bessel/multilinear tail',float(row9['global_Z_tail_upper_multilinear_12F'])))
eps=sum(v for _,v in components)
log_lo=-math.log(r+eps)
log_hi=-math.log(r-eps)
log_rad=max(L-log_lo,log_hi-L)
rep={
 'status':'PASS_FREEZE_U1_TWO_STEP_TEMPORAL_BESSEL_CUTOFF',
 'scope':'L=2 U(1) temporal two-step six-block contraction only; not full Task 1',
 'central_cutoff':4,
 'Z_center':[z.real,z.imag],
 'abs_Z_center':r,
 'minus_log_abs_center':L,
 'absolute_Z_error_bound_to_infinite_Bessel_cutoff':eps,
 'Z_real_interval':[z.real-eps,z.real+eps],
 'Z_imag_interval':[z.imag-eps,z.imag+eps],
 'minus_log_abs_interval':[log_lo,log_hi],
 'minus_log_abs_symmetric_radius':log_rad,
 'bound_components':[{'name':n,'abs_bound':v} for n,v in components],
 'proof_chain':[
   'N=4 center is the symmetry-reduced six-block contraction refined to sector threshold 1e-12.',
   'Its omitted sectors are bounded by a positive L1 sector majorant.',
   'Successive cutoff increments N->N+1 for N=4,...,8 are bounded by a telescoping positive L1 convolution bound with an explicit float64 safety envelope.',
   'The remaining N>=9 tail is bounded analytically using an absolute Bessel-series majorant and 12-factor multilinearity.',
   'Triangle inequality yields the stated absolute Z error bound; monotonicity of -log r on the radial interval yields the logarithmic interval.'
 ],
 'inputs_sha256':{k:sha(p) for k,p in files.items()},
}
out='/mnt/data/qict_task1_v5/U1_TEMPORAL_BESSEL_INFINITY_FREEZE_CERTIFICATE.json'
with open(out,'w') as f:json.dump(rep,f,indent=2);f.write('\n')
print(json.dumps(rep,indent=2))
