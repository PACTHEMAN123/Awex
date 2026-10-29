"""Detach the controller from SSH while retaining its PID and log."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base = Path('/tmp/weightbridge-memory-1029')
marker = base / 'launch-formal.json'
if marker.exists():
    raise SystemExit('Launch already recorded; inspect its status before restarting')
with (base / 'suite-formal.log').open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', str(base / 'run_suite.py')],
                               env=dict(os.environ, PYTHONPATH=str(base)),
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=base)
row = dict(pid=process.pid, started=time.time(), log=str(base / 'suite-formal.log'))
marker.write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
