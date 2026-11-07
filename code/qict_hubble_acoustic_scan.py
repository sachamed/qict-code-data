"""Numerical H0 geometry scan for the QICT finite-N CLASS model.

No posterior, no fitted sound-horizon measurement and no optimized likelihood.
The reference angle is taken from the archived weak-amplitude native CLASS output.
"""
import argparse,json
from pathlib import Path
import numpy as np
from qict_native_early_energy_gate import prepare,update

H0_VALUES=(67.4,69.5,71.5,73.04,75.0,77.0)
REF_WEAK_100THETA_GEOMETRIC=1.0410104064745338
OMEGA_R_PHYSICAL=2.469e-5*(1+0.22710731766*3.044)
OMEGA_M_PHYSICAL=0.02237+0.12
RHO_PHI0_PHYSICAL=0.00038644699225749197*(0.674**2)

def name(h0):
    return f"qict_scan_h{h0:.2f}_"

def build(class_root):
    prepare(class_root)
    base=(class_root/"qict_early_g0.02.ini").read_text()
    for H0 in H0_VALUES:
        h=H0/100.
        lam=1-(OMEGA_M_PHYSICAL+OMEGA_R_PHYSICAL+RHO_PHI0_PHYSICAL)/h**2
        if not 0<lam<1:
            raise ValueError("unphysical lambda predictor")
        s=update(base,"h",f"{h:.12g}")
        s=update(s,"Omega_Lambda",f"{lam:.15g}")
        s=update(s,"root",f"results/{name(H0)}")
        p=class_root/f"qict_scan_h{H0:.2f}.ini"
        p.write_text(s)
        print("H0 geometry scan",H0,"Lambda-closure predictor",lam,flush=True)

def validate(root):
    rows=[]
    for H0 in H0_VALUES:
        p=root/name(H0)
        # CLASS appends 00_ between root and file kind.
        stem=root/(name(H0)+"00_")
        bg=np.loadtxt(str(stem)+"background.dat")
        cl=np.loadtxt(str(stem)+"cl_lensed.dat")
        if not np.isfinite(bg).all() or not np.isfinite(cl).all():
            raise ValueError(f"non-finite outputs at H0={H0}")
        if len(cl)!=3999 or bg.shape[1]<32:
            raise ValueError("missing CLASS high-l data")
        order=np.argsort(bg[:,0]);z=bg[order,0]
        rs=np.interp(1090.,z,bg[order,7])
        chi=np.interp(1090.,z,bg[order,4])
        theta=100*rs/chi
        ell=cl[:,0];m=(ell>=150)&(ell<=320)
        peak=int(ell[m][np.argmax(cl[m,1])])
        frac=bg[:,14]/bg[:,13]
        rows.append({"H0_input":H0,"rs_z1090_Mpc":float(rs),
          "chi_z1090_Mpc":float(chi),"100_theta_geometric_z1090":float(theta),
          "abs_theta_distance_to_weak_benchmark":float(abs(theta-REF_WEAK_100THETA_GEOMETRIC)),
          "TT_first_peak_ell_proxy":peak,
          "f_phi_peak":float(frac.max()),"z_f_phi_peak":float(bg[np.argmax(frac),0]),
          "f_phi_today":float(frac[np.argmin(bg[:,0])])})
    vals=np.array([r["100_theta_geometric_z1090"] for r in rows])
    hs=np.array([r["H0_input"] for r in rows])
    if not (np.diff(vals)>0).all():
        raise ValueError("H0 acoustic monotonicity did not hold")
    best=rows[int(np.argmin(abs(vals-REF_WEAK_100THETA_GEOMETRIC)))]
    bracket=bool(vals.min()<=REF_WEAK_100THETA_GEOMETRIC<=vals.max())
    interpolation=float(np.interp(REF_WEAK_100THETA_GEOMETRIC, vals,hs)) if bracket else None
    doc={"status":"forward_CLASS_geometric_scan_complete",
       "not_a_likelihood_fit":True,
       "cosmology":"massless Neff=3.044, fixed omega_b and omega_c, g=0.02, N=250, F=0.225",
       "theta_reference":"independent archived weak-amplitude CLASS run at z=1090, not Planck posterior",
       "target_100theta_geometric":REF_WEAK_100THETA_GEOMETRIC,
       "bracketed":bracket,"interpolated_H0_if_bracketed":interpolation,
       "closest_grid_point":best,"results":rows}
    (root/"QICT_H0_ACOUSTIC_GEOMETRY_SCAN.json").write_text(json.dumps(doc,indent=2)+"\n")
    print(json.dumps(doc,indent=2),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","validate"))
    ap.add_argument("--class-root",default="/tmp/qict_idm")
    ap.add_argument("--results",default="results/qict_hubble_geometry")
    args=ap.parse_args()
    build(Path(args.class_root)) if args.mode=="prepare" else validate(Path(args.results))
