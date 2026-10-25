"""Reconstruct one accepted launch from raw evidence; standard library only."""
import hashlib,json,math,re,statistics,tarfile
from pathlib import Path
HF_BYTES=61064245248

def load(p):return json.loads(p.read_text())
def save(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def json_events(text,token):
 decoder=json.JSONDecoder()
 return [decoder.raw_decode(text[m.end():])[0] for m in re.finditer(re.escape(token),text)]

def analyze(directory):
 directory=Path(directory);case=load(directory/'case.json');outcome=load(directory/'outcome.json')
 assert load(directory/'exit.json')['code']==0 and outcome['passed']
 assert outcome['publication_mechanism']==case['mechanism']
 assert outcome['weight_format']=='bf16' and outcome['model_weight_profile']
 assert outcome['prefix_caching_disabled'] and outcome['gpu_placement']=='nic-spread'
 assert outcome['layout']==case['layout'] and outcome['training_hosts']==case['training_hosts'] and outcome['rollout_hosts']==case['rollout_hosts']
 assert outcome['training']==case.get('training',dict(tp=4,pp=1,cp=2,ep=8,etp=1,world_size=case['training_gpus']))
 for key in ['tp','dp','ep']:assert outcome['rollout'][key]==case[key]
 assert outcome['rollout']['instances']==case['replicas'] and outcome['rollout']['world_size']==case['rollout_gpus']
 assert outcome['num_updates']==35 and outcome['warmup_updates']==5 and outcome['publication_bucket_mb']==256
 assert len(outcome['generation_before'])==len(outcome['generation_after'])==case['replicas']
 assert [r['message'] for r in outcome['generation_before']]==[r['message'] for r in outcome['generation_after']]
 env=outcome['transport_environment']
 assert env['NCCL_BUFFSIZE']=='4194304'
 if case['backend']=='shardstream':
  expected={'SHARDSTREAM_DP_SOURCE_POLICY':'stream','SHARDSTREAM_RING_SOURCE_NODE_SWIZZLE':'1',
   'SHARDSTREAM_REPLICA_ROUTING':'logical','SHARDSTREAM_COMPACT_FIFO':'1','SHARDSTREAM_MAX_CHANNELS':'64',
   'SHARDSTREAM_GIN_FIFO_DEPTH':'16','SHARDSTREAM_GIN_CONNECTIONS':'4'}
  for key,value in expected.items():assert env[key]==value,(key,env)
 else:assert 'SHARDSTREAM_COMPACT_FIFO' not in env
 before=load(directory/'inventory.json');after=load(directory/'final-inventory.json')
 assert len(before)==len(after)==4
 assert all(not r['gpu_processes'].strip() for r in before+after)
 assert {r['prepared']['revision'] for r in before}=={case['revision']}
 assert len({r['prepared']['native_sha256'] for r in before})==1
 if case['backend']=='nixl':
  assert all(r['prepared']['nixl']['version']=='1.5.0' for r in before)
  assert {r['prepared']['nixl']['artifact_sha256'] for r in before}=={'de778242fbd632b9f44ca0c8240b393b746068b56e6856b8bdfb1edaa1580653'}
  assert len({json.dumps(r['prepared']['nixl']['files'],sort_keys=True) for r in before})==1
  assert '/tmp/figure11-1025/nixl-runtime' in case['command'][case['command'].index('--extra-python-path')+1].split(':')
 records=None;affinity={};training_layout=[];native_metrics=[];archives=[];rdma_transports=[];rdma_completions=[];rdma_reuse=[];astate_transports=[];astate_completions=[]
 for item in load(directory/'manifest.json'):
  path=directory/item['file'];assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  archives.append(dict(file=str(path),sha256=item['sha256']))
  with tarfile.open(path) as a:
   for member in a.getmembers():
    if member.name=='training-publications.json':
     assert records is None;publication=json.load(a.extractfile(member));assert publication['passed'];records=publication['records']
    elif member.name.endswith('.log'):
     text=a.extractfile(member).read().decode(errors='replace')
     training_layout.extend(json_events(text,'SHARDSTREAM_TRAIN_LAYOUT '))
     rdma_transports.extend(json_events(text,'AWEX_RDMA_TRANSPORT '))
     rdma_completions.extend(json_events(text,'AWEX_RDMA_COMPLETE '))
     rdma_reuse.extend(json_events(text,'AWEX_RDMA_REUSE '))
     astate_transports.extend(json_events(text,'AWEX_ASTATE_TRANSPORT '))
     astate_completions.extend(json_events(text,'AWEX_ASTATE_COMPLETE '))
     if member.name.startswith('rollout-'):
      rows=json_events(text,'SHARDSTREAM_RANK_AFFINITY ')
      if rows:affinity[member.name]=(item['ip'],rows)
 if case['mechanism']=='awex_native_astate':
  assert not rdma_transports and not rdma_completions, 'mixed Awex transports'
  artifacts=[r['prepared']['astate'] for r in before]
  qualification=(directory/'astate-qualification.json').read_bytes()
  assert hashlib.sha256(qualification).hexdigest()==case['astate_qualification_sha256']
  qualified=json.loads(qualification)
  assert len(qualified)==2 and {q['role'] for q in qualified}=={'send','receive'}
  for q in qualified:
   assert q['code']==0 and not q['timed_out'] and not q['gpu_processes_after']
   assert [e['version'] for e in q['events']]==list(range(-1,3))
   assert all(e['verified'] for e in q['events'])
   assert all(q['artifact']['extension_sha256']==r['extension_sha256'] and q['artifact']['utrans_sha256']==r['utrans_sha256'] for r in artifacts)
  assert len({r['extension_sha256'] for r in artifacts})==1
  assert len({r['utrans_sha256'] for r in artifacts})==1
  assert len({r['source_revision'] for r in artifacts})==1
  assert len({r['source_patch_sha256'] for r in artifacts})==1
  assert all(r['transport']=='native-ucx-rdma' for r in artifacts)
  expected_count=case['training_gpus']+case['rollout_gpus']
  assert len(astate_transports)==len(astate_completions)==expected_count
  assert all(r['protocol']=='rdma' for r in astate_transports+astate_completions)
  assert all(r['staging']=='cpu_pinned' and r['discovery']=='HTTP' and r['max_rdma_devices']=='2' for r in astate_transports)
  assert {r['sha256'] for r in astate_transports}=={artifacts[0]['extension_sha256']}
  assert {r['utrans_sha256'] for r in astate_transports}=={artifacts[0]['utrans_sha256']}
  assert {r['source_revision'] for r in astate_transports}=={artifacts[0]['source_revision']}
  assert {r['source_patch_sha256'] for r in astate_transports}=={artifacts[0]['source_patch_sha256']}
  assert all(len(r['rdma_devices'])==len(set(r['rdma_devices']))==2 for r in astate_transports)
  assert all(r['transport_implementation']=='native-ucx-rdma' and r['ucx_tls']=='rc,self' and r['ucx_gid_index']=='3' for r in astate_transports)
  for rows in [astate_transports,astate_completions]:
   assert sum(r['role']=='send' for r in rows)==case['training_gpus']
   assert sum(r['role']=='receive' for r in rows)==case['rollout_gpus']
   assert {r['rank'] for r in rows}==set(range(expected_count))
 elif case['backend']=='awex':
  assert case['mechanism']=='awex_native_rdma'
  assert not astate_transports and not astate_completions, 'mixed Awex transports'
  qualification=(directory/'mooncake-qualification.json').read_bytes()
  assert hashlib.sha256(qualification).hexdigest()==case['mooncake_qualification_sha256']
  qualified=json.loads(qualification)
  assert len(qualified)==3 and sum(q['role']=='send' for q in qualified)==1
  for q in qualified:
   assert q['code']==0 and not q['timed_out'] and not q['gpu_processes_after'] and q['closed']
   assert [e['version'] for e in q['events']]==list(range(-1,3))
   assert all(e['verified'] and e['register_calls']==2 for e in q['events'])
   assert len(q['before'])==len(q['after'])==4 and set(q['before'])==set(q['after'])
   assert any(q['after'][n][k]>q['before'][n][k] for n in q['before'] for k in ['tx_bytes','rx_bytes'])
   assert len({tuple(e['registered_addresses']) for e in q['events']})==1
   assert all(q['artifact']['awex_transport_sha256']==r['prepared']['awex_transport_sha256']
              and q['artifact']['awex_revision']==r['prepared']['awex_revision'] for r in before)
  assert all(r['prepared']['registration_policy']=='persistent_snapshots' for r in before)
  assert all(r['prepared']['mooncake_version']=='0.3.13.post1' for r in before)
  assert len({r['prepared']['mooncake_sha256'] for r in before})==1
  expected_count=case['training_gpus']+case['rollout_gpus']
  assert len(rdma_transports)==len(rdma_completions)==expected_count
  assert all(r['protocol']=='rdma' for r in rdma_transports+rdma_completions)
  assert all(r['handshake_max_length']=='67108864' and r['registration_policy']=='persistent_snapshots' for r in rdma_transports)
  assert len(rdma_reuse)==35*expected_count
  for rank in range(expected_count):
   rows=sorted([r for r in rdma_reuse if r['rank']==rank],key=lambda r:r['sequence'])
   assert len(rows)==35 and [r['sequence'] for r in rows]==list(range(1,36))
   assert len({(r['pid'],r['role'],r['register_calls'],r['registered_buffers'],r['registered_bytes']) for r in rows})==1
   assert all(r['registration_policy']=='persistent_snapshots' for r in rows)

  assert {r['sha256'] for r in rdma_transports}=={before[0]['prepared']['awex_transport_sha256']}
  assert sum(r['role']=='send' for r in rdma_completions)==case['training_gpus']
  assert sum(r['role']=='receive' for r in rdma_completions)==case['rollout_gpus']
 assert records is not None and len(records)==35
 assert [r['version'] for r in records]==list(range(-1,34))
 assert len(training_layout)==case['training_gpus']
 assert {r['rank'] for r in training_layout}==set(range(case['training_gpus']))
 pointers={};checks=0;network=[]
 for i,r in enumerate(records):
  assert len(r['ranks'])==case['replicas']
  assert sum(len(v) for v in r['ranks'].values())==case['rollout_gpus']
  for engine,workers in r['ranks'].items():
   assert len(workers)==case['tp']*case['dp']
   for rank,w in enumerate(workers):
    assert w['verified'] and w['version']==r['version'] and w['parameter_count']==len(w['pointers'])>0
    assert pointers.setdefault((engine,rank),w['pointers'])==w['pointers'];checks+=1
    if case['backend']=='shardstream':
     metric=w['transfer_metrics'];assert metric['channel_limit']==64 and metric['registered_fifo_depth']==16
     assert metric['gin_connection_count']==4
     if i>=5:assert metric['plan_cache_hit']
     native_metrics.append(dict(version=r['version'],engine=engine,rank=rank,metrics=metric))
  if case['backend'] in ('nccl','nixl'):
   metric=r['backend_metrics'];assert metric['backend']==case['backend'] and metric['payload_bytes']==HF_BYTES
   assert len(metric['receivers'])==case['rollout_gpus']
   for recv in metric['receivers']:
    assert recv['payload_bytes']==HF_BYTES and recv['received_tensors']==metric['tensor_count']>0
  old,new=r['telemetry_before'],r['telemetry_after']
  nodes={f'train-{x["rank"]}':(x,y) for x,y in zip(old['training_ranks'],new['training_ranks']) if x['rdma']}
  nodes.update({host:(x,new['rollout_nodes'][host]) for host,x in old['rollout_nodes'].items()})
  deltas={}
  for node,(a,b) in nodes.items():
   assert len(a['rdma'])==4 and set(a['rdma'])==set(b['rdma'])
   ports={port:{key:b['rdma'][port][key]-v[key] for key in ['tx_bytes','rx_bytes']} for port,v in a['rdma'].items()}
   assert all(v>=0 for port in ports.values() for v in port.values());deltas[node]=ports
  network.append(dict(version=r['version'],nodes=deltas))
 engines=[];counts=[case['replicas']//len(case['rollout_hosts'])+(i<case['replicas']%len(case['rollout_hosts'])) for i in range(len(case['rollout_hosts']))]
 for host,count in zip(case['rollout_hosts'],counts):
  placement=outcome['node_placements'][host];gpus=placement['gpus'];assert len(gpus)==len(set(gpus))==count*case['tp']*case['dp']
  for local_engine in range(count):
   number=len(engines);size=case['tp']*case['dp'];chosen=gpus[local_engine*size:(local_engine+1)*size]
   hcas=[placement['affinity'][gpu]['hca'].lstrip('=') for gpu in chosen]
   if case['tp']==4:
    for offset in range(0,size,4):assert len(set(hcas[offset:offset+4]))==4
   ip,rows=affinity['rollout-'+str(number)+'.log'];assert ip==host
   assert {r['physical_gpu'] for r in rows}==set(chosen) and all(r['before_nccl_init'] for r in rows)
   if case['backend']=='shardstream':
    expected={gpu:hca for gpu,hca in zip(chosen,hcas)}
    assert all(r['hca'].lstrip('=')==expected[r['physical_gpu']] for r in rows)
   else:
    all_hcas={v['hca'].lstrip('=') for v in placement['affinity'].values()}
    assert len(all_hcas)==4
    assert all(set(r['hca'].lstrip('=').split(','))==all_hcas for r in rows)
   engines.append(dict(engine=number,host=host,gpus=chosen,hcas=hcas))
 assert len(engines)==case['replicas']
 samples=[r['end_to_end_update_ms'] for r in records[5:]]
 assert len(samples)==30 and all(math.isfinite(v) and v>0 for v in samples)
 result=dict(backend=case['backend'],mechanism=case['mechanism'],case_id=case['case_id'],series=case['series'],repeat=case['repeat'],revision=case['revision'],
  verified_updates=35,measured_updates=30,worker_checks=checks,p50_ms=statistics.median(samples),p95_ms=sorted(samples)[28],
  cold_update_ms=records[0]['end_to_end_update_ms'],samples_ms=samples,training=outcome['training'],rollout=outcome['rollout'],
  training_hosts=case['training_hosts'],rollout_hosts=case['rollout_hosts'],active_rollout_hosts=len(case['rollout_hosts']),
  native_sha256=before[0]['prepared']['native_sha256'],generation_unchanged=True,full_parameters_verified=True,
  engine_placement=engines,archive_provenance=archives)
 if case['backend']=='awex':
  result['awex_revision']=case['awex_revision']
  if case['mechanism']=='awex_native_rdma':
   result['registration_policy']='persistent_snapshots'
   result['awex_transport_sha256']=before[0]['prepared']['awex_transport_sha256']
   result['registration_counts']=[r for r in rdma_reuse if r['sequence']==1]
 if case['mechanism']=='awex_native_astate':
  result['astate_artifact']=artifacts[0]
 if case.get('memory_profile'):
  assert outcome['communication_memory_baseline']
  memory=[]
  for r in records:
   for engine,workers in r['ranks'].items():
    for rank,w in enumerate(workers):
     m=w['communication_memory'];assert m['sample_count']>0
     assert m['sampled_device_peak_bytes']>=m['sampled_device_before_bytes']
     if case['mechanism']=='awex_native_astate':
      assert m['sampled_host_rss_peak_bytes']>=m['sampled_host_rss_before_bytes']
     memory.append(dict(version=r['version'],engine=engine,rank=rank,**m))
  result['communication_memory']=memory
 return result,network,native_metrics,records
