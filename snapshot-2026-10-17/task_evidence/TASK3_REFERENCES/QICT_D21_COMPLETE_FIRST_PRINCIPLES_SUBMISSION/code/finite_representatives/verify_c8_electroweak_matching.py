#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; CERT=ROOT/'certificates'; CERT.mkdir(exist_ok=True)
OUT=CERT/'c8_electroweak_matching_certificate.json'
def main():
    Y={'QL':Fraction(1,6),'uR':Fraction(2,3),'dR':Fraction(-1,3),'LL':Fraction(-1,2),'eR':Fraction(-1,1),'nuR':Fraction(0,1)}
    kY=6*Y['QL']**2+3*Y['uR']**2+3*Y['dR']**2+2*Y['LL']**2+Y['eR']**2
    Tfund=Fraction(1,2)
    k2=3*Tfund+Tfund
    k3=2*Tfund+Tfund+Tfund
    gY2_g22=k2/kY; g32_g22=k2/k3
    sin2=gY2_g22/(1+gY2_g22)
    # C8 normalized nonzero vacuum orbit: choose the gauge-fixed representative h0=(0,1).
    h0=np.array([0.0+0j,1.0+0j])
    vacuum_norm=float(np.vdot(h0,h0).real)
    T3=np.diag([0.5,-0.5]); YH=0.5*np.eye(2)
    QH=T3+YH
    vacuum_stabilizer_defect=float(np.linalg.norm(QH@h0))
    # lower component of H_R: T3=-1/2,Y=+1/2
    Qvac=Fraction(-1,2)+Fraction(1,2)
    Qnu=Fraction(1,2)+Y['LL']; QeL=Fraction(-1,2)+Y['LL']; QeR=Y['eR']
    QuL=Fraction(1,2)+Y['QL']; QdL=Fraction(-1,2)+Y['QL']; QuR=Y['uR']; QdR=Y['dR']
    cert={'status':'PASS','principle':'C8 primitive common gauge normalization',
          'unbroken_generator':'Q=T3+Y','vacuum_charge':str(Qvac),
          'normalized_vacuum_representative':'(0,1)^T','vacuum_norm':vacuum_norm,'vacuum_stabilizer_defect':vacuum_stabilizer_defect,
          'charges':{'nu_L':str(Qnu),'e_L':str(QeL),'e_R':str(QeR),'u_L':str(QuL),'d_L':str(QdL),'u_R':str(QuR),'d_R':str(QdR)},
          'kY':str(kY),'k2':str(k2),'k3':str(k3),'gY2_over_g22':str(gY2_g22),'g32_over_g22':str(g32_g22),'sin2_thetaW':str(sin2),
          'e_over_g2':'sqrt(3/8)','e_over_gY':'sqrt(5/8)','absolute_coupling_fixed':False}
    assert abs(vacuum_norm-1.0)<1e-15 and vacuum_stabilizer_defect<1e-15
    assert Qvac==0 and (Qnu,QeL,QeR)==(0,-1,-1)
    assert (QuL,QdL,QuR,QdR)==(Fraction(2,3),Fraction(-1,3),Fraction(2,3),Fraction(-1,3))
    assert kY==Fraction(10,3) and k2==2 and k3==2 and gY2_g22==Fraction(3,5) and g32_g22==1 and sin2==Fraction(3,8)
    OUT.write_text(json.dumps(cert,indent=2)+'\n'); print('PASS C8-electroweak',json.dumps(cert,indent=2))
if __name__=='__main__': main()
