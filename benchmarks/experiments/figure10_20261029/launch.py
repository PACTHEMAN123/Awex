"""Launch the exclusive controller independently of the SSH session."""
import os
from pathlib import Path
import subprocess
import sys
import time

from protocol import BASE, save

assert not (BASE / 'launch.json').exists(), 'Inspect the existing launch before restarting'
with (BASE / 'suite.log').open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('run_suite.py'))],
                               env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent)),
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=BASE)
save(BASE / 'launch.json', dict(pid=process.pid, started=time.time(), log=str(BASE / 'suite.log')))
print(f'Controller PID {process.pid}', flush=True)
