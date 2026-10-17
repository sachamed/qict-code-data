# Deliberately nonphysical provider: used only to demonstrate fail-closed metadata checks.
import numpy as np
D=12
U=np.diag(np.exp(1j*np.linspace(0.1,1.2,D)))
V=np.eye(D,6,dtype=complex)
def dimension(): return D
def source_matrix(): return V.copy()
def matmat(X): return U@X
def metadata():
 return {'physical_payload':False,'action_sha256':'synthetic','L':24,'PiCAR':1,'source_order':['L1','L2','L3','R1','R2','R3']}
