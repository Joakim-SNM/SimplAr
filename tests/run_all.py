"""Run every test in this folder, each in its own Python process.

    python tests/run_all.py

Exits with 1 if any test fails.  The GUI tests run off-screen.
"""
import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg',
           SIMPLAR_DATA_DIR=tempfile.mkdtemp(), PYTHONIOENCODING='utf-8')
failed = []
for path in sorted(glob.glob(os.path.join(HERE, 'test_*.py'))):
    name = os.path.basename(path)
    r = subprocess.run([sys.executable, path], cwd=HERE, env=env,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(f"{'passed' if r.returncode == 0 else 'FAILED'}  {name}")
    if r.returncode != 0:
        failed.append(name)
        print(r.stdout[-3000:], r.stderr[-3000:], sep='\n')
print(f"\n{len(failed)} failed" if failed else "\nall tests passed")
sys.exit(1 if failed else 0)
