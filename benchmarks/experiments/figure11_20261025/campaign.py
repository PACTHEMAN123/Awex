"""Run qualified Fig 11 first, then remaining Awex RDMA paper points."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/tmp/figure11-1025')
SCRIPT=Path(__file__).with_name('run_matrix.py')

def save(value):
 p=BASE/'campaign-state.json';tmp=p.with_suffix('.partial')
 tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
 stages=[('full-model-qualification',['--backend','awex','--limit','1']),
         ('figure11',[]),('other-paper-awex',['--suite','paper-awex'])]
 for stage,args in stages:
  save(dict(state='running',stage=stage,pid=os.getpid(),started=time.time()))
  print(json.dumps(dict(event='campaign-stage',stage=stage)),flush=True)
  code=subprocess.call([sys.executable,'-u',str(SCRIPT),*args])
  if code:
   save(dict(state='needs-investigation',stage=stage,exit_code=code,finished=time.time()))
   raise SystemExit(code)
  root=BASE/('other-results' if stage=='other-paper-awex' else 'results')
  state=json.loads((root/'active.json').read_text())
  required=1 if stage=='full-model-qualification' else (132 if stage=='figure11' else 24)
  assert state['passed']==required,state
 save(dict(state='complete',figure11_launches=132,other_awex_launches=24,finished=time.time()))

if __name__=='__main__':main()
