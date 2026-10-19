import pathlib, subprocess, sys, tempfile, unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
CMD=ROOT/"tools/repository_integrity.py"

class RepositoryTests(unittest.TestCase):
    def test_runner_help(self):
        p=subprocess.run([sys.executable,str(ROOT/"REPRODUCE_ALL.py"),"--help"],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_missing_manuscript_audit_is_refused_before_numeric_work(self):
        if (ROOT/"paper/manuscript_final.tex").exists():self.skipTest("manuscript provided")
        p=subprocess.run([sys.executable,str(ROOT/"REPRODUCE_ALL.py"),"--publication-audits"],
                         capture_output=True,text=True,timeout=30)
        self.assertNotEqual(p.returncode,0)
        self.assertIn("Publication source package not available",p.stderr+p.stdout)
    def test_integrity_detects_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            root=pathlib.Path(t);f=root/"a.txt";f.write_text("original")
            out=root/"hashes.txt"
            w=subprocess.run([sys.executable,str(CMD),"--root",t,"--write",str(out)],capture_output=True,text=True)
            self.assertEqual(w.returncode,0,w.stderr)
            good=subprocess.run([sys.executable,str(CMD),"--root",t,"--check",str(out)],capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            f.write_text("changed")
            bad=subprocess.run([sys.executable,str(CMD),"--root",t,"--check",str(out)],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
    def test_physical_mass_remains_blocked(self):
        script=ROOT/"QICT_FULL_PHYSICAL_EXECUTION/src/stage_guard.py"
        p=subprocess.run([sys.executable,str(script),"--stage","mass"],capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,2,p.stderr)
        self.assertIn("BLOCKED_MISSING_INPUT",p.stdout)

if __name__=="__main__":unittest.main()
