"""Run qualified Fig 11 first, then remaining Awex NCCL paper points."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/tmp/figure11-1025')
SCRIPT=Path(__file__).with_name('run_matrix.py')
from plan import matrix, matching_completions
from other_awex_plan import matrix as other_matrix

def save(value):
 p=BASE/'campaign-state.json';tmp=p.with_suffix('.partial')
 tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
 stages=[('full-model-qualification',['--backend','awex','--limit','1']),
         ('figure11',[]),('other-paper-awex',['--suite','paper-awex'])]
 for stage,args in stages:
  if stage=='full-model-qualification':
   row=next(r for r in matrix() if r['case_id']=='inter-r1' and r['backend']=='awex' and r['repeat']==0)
   qualified=matching_completions(BASE/'results',row)
   if qualified:
    assert len(qualified)==1
    summary=json.loads(qualified[0].with_name('summary.json').read_text())
    assert summary['backend']=='awex' and summary['verified_updates']==35 and summary['full_parameters_verified']
    continue
  save(dict(state='running',stage=stage,pid=os.getpid(),started=time.time()))
  print(json.dumps(dict(event='campaign-stage',stage=stage)),flush=True)
  if stage=='other-paper-awex':
   code=subprocess.call([sys.executable,'-u',str(SCRIPT.with_name('verify_other_assets.py'))])
   if code:
    save(dict(state='needs-investigation',stage='other-model-checksums',exit_code=code,finished=time.time()))
    raise SystemExit(code)
  code=subprocess.call([sys.executable,'-u',str(SCRIPT),*args])
  if code:
   save(dict(state='needs-investigation',stage=stage,exit_code=code,finished=time.time()))
   raise SystemExit(code)
  root=BASE/('other-results' if stage=='other-paper-awex' else 'results')
  state=json.loads((root/'active.json').read_text())
  required=1 if stage=='full-model-qualification' else (132 if stage=='figure11' else len(other_matrix()))
  assert state['passed']==required,state
 save(dict(state='complete',figure11_launches=132,other_awex_launches=len(other_matrix()),finished=time.time()))

if __name__=='__main__':main()
