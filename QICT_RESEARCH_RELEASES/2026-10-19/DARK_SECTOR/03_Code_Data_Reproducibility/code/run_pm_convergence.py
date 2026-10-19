"""Reproduce the 200/300-step PM comparison without retaining temporary runs."""
from pathlib import Path
import json, os, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parent
P=ROOT.parent
def main():
    records={}
    with tempfile.TemporaryDirectory(prefix='qict-pm-comparison-') as td:
        for steps in [200,300]:
            env=os.environ.copy();env['QICT_PM_STEPS']=str(steps)
            dest=Path(td) if steps==200 else P
            env['QICT_PM_OUTPUT']=str(dest)
            subprocess.run([sys.executable,str(ROOT/'particle_mesh_3d.py')],env=env,check=True)
            report=json.loads((dest/'data/particle_mesh_results.json').read_text())
            records[str(steps)]=report['snapshots']['a1.00']
    a,b=records['200'],records['300']
    records.update(rms_relative_delta=abs(a['rms_delta']/b['rms_delta']-1),
                   peak_relative_delta=abs(a['peak_delta']/b['peak_delta']-1))
    (P/'data/pm_convergence.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'rms_relative_delta':records['rms_relative_delta'],
                      'peak_relative_delta':records['peak_relative_delta']}))

if __name__=='__main__':main()
