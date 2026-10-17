#!/usr/bin/env python3
from __future__ import annotations
import cmath,json,math
from math import gcd
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True); OUT=CERT/'primitive_recurrence_cell_certificate.json'
S=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)
def U(k): return np.diag([1,1,1,cmath.exp(-1j*2*math.pi*k/126)])@S
def main():
    primitive=[k for k in range(1,64) if gcd(k,126)==1]; kstar=min(primitive)
    M=U(kstar); I=np.eye(4,dtype=complex)
    unit=float(np.linalg.norm(M.conj().T@M-I,2)); rec=float(np.linalg.norm(np.linalg.matrix_power(M,126)-I,2))
    diff=abs(cmath.exp(-1j*math.pi/63)-cmath.exp(-2j*math.pi/63))
    cert={'status':'PASS','task':'C8-collision','final_status':'C8 primitive-character theorem','primitive_positive_characters':primitive,'selected_k':kstar,'selected_phase_exact':'pi/63','unitarity_defect':unit,'recurrence_126_defect':rec,'k1_k2_composite_moment_difference':diff,'expected_difference':0.04986138347614575}
    assert kstar==1 and unit<1e-12 and rec<1e-11 and abs(diff-0.04986138347614575)<1e-15
    OUT.write_text(json.dumps(cert,indent=2)+'\n'); print('PASS C8-collision',json.dumps(cert,indent=2))
if __name__=='__main__': main()
