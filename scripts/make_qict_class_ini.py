"""Create two reproducible low-amplitude QICT CLASS smoke-test inputs.

These initial numbers are computational test points, NOT measured best fits.
Use the fork's own iDM.ini as a reference with all required defaults.
"""
import re,sys,pathlib
root=pathlib.Path(sys.argv[1])
text=(root/"ini/iDM.ini").read_text()
def update(src,key,value):
    pattern=re.compile(r"(?m)^(\s*)"+re.escape(key)+r"(\s*=\s*)([^#\n]*)(.*)$")
    m=pattern.search(src)
    if not m:
        return src+"\n"+key+" = "+str(value)+"\n"
    return src[:m.start()]+m.group(1)+key+m.group(2)+str(value)+m.group(4)+src[m.end():]
for g in (0.0,0.02):
    s=text
    for k,v in {
        "model_cdm":"i",
        "cdm_c":g,
        "scf_potential":"QICT_finite_N",
        "scf_parameters":"1e-8, 0.35, 250., 0., 0., 0., 0., 0., 0., 0., 0.875, 0.",
        "attractor_ic_scf":"no",
        "scf_tuning_index":"0",
        "Omega_scf":"0.0001",
        "gauge":"newtonian",
        "output":"tCl,pCl,lCl,mPk",
        "l_max_scalars":"1250",
        "l_max_lss":"120",
        "nbody_gauge_transfer_functions":"no",
        "root":f"results/qict_class_g{g:.2f}_",
    }.items():
        s=update(s,k,v)
    path=root/f"qict_g{g:.2f}.ini"
    path.write_text(s)
    print(path,flush=True)
