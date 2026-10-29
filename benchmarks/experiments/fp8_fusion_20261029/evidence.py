import hashlib,json,math,os,re,statistics,tarfile,time,subprocess
from pathlib import Path
import ray
ROOT=Path("/tmp/fp8-fusion-resume-1029/results")

def save(path,value):
 path.write_text(json.dumps(value,indent=2)+"\n")


@ray.remote(num_cpus=0)
def cleanup_shm(inventory):
    # The global experiment lock is held. Only names absent before this exact
    # launch are candidates; never reclaim another experiment's old segments.
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not processes, processes
    candidates={p.name:p for p in Path('/dev/shm').glob('nccl-*')
                if p.name not in inventory['shm_files'] and p.stat().st_uid==os.getuid()}
    if 'allowed_inodes' in inventory:
        candidates={name:file for name,file in candidates.items()
                    if inventory['allowed_inodes'].get(name)==file.stat().st_ino}
    references=set()
    boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    excluded=[]
    unreadable_new=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try:
            references.update(re.findall(r'/dev/shm/(nccl-[A-Za-z0-9]+)',(process/'maps').read_text()))
            for fd in (process/'fd').iterdir():
                try: references.update(re.findall(r'/dev/shm/(nccl-[A-Za-z0-9]+)',os.readlink(fd)))
                except FileNotFoundError: pass
        except (FileNotFoundError,ProcessLookupError): pass
        except PermissionError:
            try:
                fields=(process/'stat').read_text().rsplit(')',1)[1].split()
                started=boot+int(fields[19])/os.sysconf('SC_CLK_TCK')
                if not (fields[0]=='Z' or started<inventory['captured'] or process.stat().st_uid!=os.getuid()):
                    unreadable_new.append(int(process.name))
                excluded.append(int(process.name))
            except (FileNotFoundError,ProcessLookupError): pass
    if unreadable_new:
        return dict(ip=ray.util.get_node_ip_address(),removed=[],
                    deferred=[dict(file=str(p),inode=p.stat().st_ino) for p in candidates.values()],
                    unreadable_new_pids=unreadable_new,
                    free_bytes=os.statvfs('/dev/shm').f_bavail*os.statvfs('/dev/shm').f_frsize)
    removed=[]
    for name,p in sorted(candidates.items()):
        if name in references: continue
        stat=p.stat()
        removed.append(dict(file=str(p),bytes=stat.st_size,allocated_bytes=stat.st_blocks*512))
        p.unlink()
    return dict(ip=ray.util.get_node_ip_address(),removed=removed,
                skipped_referenced=sorted(set(candidates)&references),excluded_pids=excluded,
                free_bytes=os.statvfs('/dev/shm').f_bavail*os.statvfs('/dev/shm').f_frsize)

def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered)-1)*q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(index-lo)

def analyze(directory):
    outcome=json.loads((directory/'outcome.json').read_text())
    case=json.loads((directory/'case.json').read_text())
    publication=None
    profiles=[]
    layouts={}
    routes={}
    plans=[]
    entries=[]
    lowered={}
    decoder=json.JSONDecoder()
    for item in json.loads((directory/'manifest.json').read_text()):
        path=directory/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        with tarfile.open(path) as archive:
            if 'training-publications.json' in archive.getnames():
                assert publication is None
                publication=json.load(archive.extractfile('training-publications.json'))
            for member in archive.getmembers():
                if not member.name.endswith('.log'): continue
                text=archive.extractfile(member).read().decode(errors='replace')
                for marker,target in [('SHARDSTREAM_PROFILE ',profiles),
                                      ('SHARDSTREAM_LOGICAL_REPLICA_PLAN ',plans),
                                      ('SHARDSTREAM_LOGICAL_REPLICA_ENTRIES ',entries)]:
                    for match in re.finditer(marker,text):
                        row,_=decoder.raw_decode(text[match.end():]); target.append(row)
                for marker,target in [('SHARDSTREAM_TRAIN_LAYOUT ',layouts),('SHARDSTREAM_RING_ROUTES ',routes)]:
                    for match in re.finditer(marker,text):
                        row,_=decoder.raw_decode(text[match.end():])
                        assert row['rank'] not in target
                        target[row['rank']]=row
                for match in re.finditer(r'Lowered transport plan rank=(\d+) sender=(True|False) spans=(\d+) payload_bytes=(\d+)',text):
                    lowered[int(match[1])]=dict(sender=match[2]=='True',spans=int(match[3]),bytes=int(match[4]))
    assert outcome['passed'] and publication and publication['passed']
    assert [r['message'] for r in outcome['generation_before']]==[r['message'] for r in outcome['generation_after']]
    tw=outcome['training']['world_size']; iw=outcome['rollout']['world_size']
    assert len(layouts)==tw and len(routes)==tw+iw and len(lowered)==tw+iw
    rows=publication['records']; assert len(rows)==outcome['num_updates']
    assert [r['version'] for r in rows]==list(range(len(rows)))
    bootstrap=publication['checkpoint_bootstrap']
    assert bootstrap['version']==-1 and outcome['weight_format']=='fp8'
    assert outcome['generation_baseline_after_checkpoint_bootstrap']
    assert outcome['generation_paused_for_publications'] and outcome['generation_pause_mode']=='wait'
    assert outcome['generation_pause_results']==[{'status':'paused'}]*outcome['rollout']['instances']
    assert outcome['generation_resume_results']==[{'status':'resumed'}]*outcome['rollout']['instances']
    pointers={}
    for row in [bootstrap,*rows]:
        assert len(row['ranks'])==outcome['rollout']['instances']
        for engine,workers in row['ranks'].items():
            assert len(workers)==outcome['rollout']['tp']*outcome['rollout']['dp']
            for index,worker in enumerate(workers):
                assert worker['verified'] and worker['version']==row['version']
                assert worker['full_checkpoint_reference'] and worker['fp8_bytes_exact']
                assert worker['fp8_elements']>0 and worker['scale_elements']>0
                assert worker['scale_rtol']==5e-7 and worker['scale_atol']==0
                assert worker['parameter_count']==len(worker['pointers'])>0
                key=engine,index
                assert pointers.setdefault(key,worker['pointers'])==worker['pointers']
    measured=rows[outcome['warmup_updates']:]
    samples=[r['end_to_end_update_ms'] for r in measured]
    versions={r['version'] for r in measured}
    steady=[r for r in profiles if r.get('event')=='weight_transfer' and r['step_id'] in versions]
    assert {(r['rank'],r['step_id']) for r in steady}=={(r,v) for r in range(tw+iw) for v in versions}
    for row in steady:
        assert row['plan_cache_hit'] and row['fp8_blockwise']
        assert row['fp8_quantized_span_count']>0
        assert all(row[k]==0 for k in ['build_batch_time_ms','metadata_upload_time_ms','plan_initialization_time_ms','transport_init_time_ms'])
    assert outcome['rollout_gpu_memory_utilization_override']==0.35
    mode=case['mode']
    assert outcome['transport_environment']['SHARDSTREAM_FP8_EXECUTION']==mode
    fingerprint_keys=['work_count','fragment_count','chunk_count','batch_count','channel_count',
        'active_peer_payload_bytes','active_peer_channel_limits','payload_bytes','fp8_wire_bytes',
        'local_step_bytes','network_step_bytes','registered_fifo_depth','gin_fifo_depth',
        'gin_connection_count','channel_limit','payload_buffer_bytes','registered_window_bytes',
        'fifo_layout_required_mask','fifo_layout_registered_mask','fifo_layout_task_peer_bytes',
        'fifo_layout_forward_peer_bytes','fifo_layout_compact','gin_type','gin_enabled']
    fingerprints={}
    for p in steady:
        assert p['fp8_execution_staged']==(mode=='staged')
        assert p['fifo_layout_compact']
        sender=p['rank']>=iw
        if mode=='staged' and sender:
            assert 0<p['fp8_staging_bytes']<=p['fp8_wire_bytes']
            assert p['fp8_quantize_job_count']>0 and p['fp8_materialization_time_ms']>0
        else:
            assert p['fp8_staging_bytes']==p['fp8_quantize_job_count']==p['fp8_materialization_time_ms']==0
        fields={k:p[k] for k in fingerprint_keys}
        assert fingerprints.setdefault(str(p['rank']),fields)==fields
    save(directory/'transport-fingerprints.json',fingerprints)
    save(directory/'transfer-profiles.json',steady)
    quant_max=[max(p['fp8_materialization_time_ms'] for p in steady if p['step_id']==v and p['rank']>=iw) for v in sorted(versions)]
    save(directory/'source-materialization.json',dict(mode=mode,versions=sorted(versions),max_source_quantize_ms=quant_max,
        note='Per-update maximum source quantization time; not an additive receiver-wait stack'))
    writer_bytes={r:statistics.median(p['payload_bytes'] for p in steady if p['rank']==r) for r in range(iw,iw+tw)}
    writer_fp8_bytes={r:int(statistics.median(p['fp8_wire_bytes'] for p in steady if p['rank']==r)) for r in range(iw,iw+tw)}
    assert all(n%16400==0 for n in writer_fp8_bytes.values())
    logical_source_bytes=sum(writer_bytes[r]-writer_fp8_bytes[r]+writer_fp8_bytes[r]//16400*32768 for r in writer_bytes)
    relay_chains={}
    rank_nodes={r['rank']+iw:r['hostname'] for r in layouts.values()}
    # The catalog is not exported verbatim; worker edges establish real paths.
    for root in range(iw,iw+tw):
        for route_id,head,forward in routes[root]['routes']:
            assert forward==-1
            previous,current=root,head
            chain=[]
            while current>=0:
                assert current not in chain and current<iw
                chain.append(current)
                links=[r for r in routes[current]['routes'] if r[0]==route_id]
                assert len(links)==1 and links[0][1]==previous
                previous,current=current,links[0][2]
            instance_size=outcome['rollout']['tp']*outcome['rollout']['dp']
            node_path=[routes[rank]['rollout_node_ids'][rank//instance_size] for rank in chain]
            if case['routing']=='logical':
                assert sum(a!=b for a,b in zip(node_path,node_path[1:]))==len(set(node_path))-1
            relay_chains[str(route_id)]=dict(root=root,chain=chain,node_path=node_path)
    if case['routing']=='logical':
        assert plans and all(p==plans[0] for p in plans)
        assert logical_source_bytes==plans[0]['unique_source_bytes']
        assert entries and all(e==entries[0] for e in entries)
    network=[]
    for row in rows:
        before,after=row['telemetry_before'],row['telemetry_after']
        nodes={f'train-{old["rank"]}':(old,new) for old,new in zip(before['training_ranks'],after['training_ranks']) if old['rdma']}
        nodes.update({host:(old,after['rollout_nodes'][host]) for host,old in before['rollout_nodes'].items()})
        deltas={}
        for node,(old,new) in nodes.items():
            assert len(old['rdma'])==4 and set(old['rdma'])==set(new['rdma'])
            deltas[node]={port:{k:new['rdma'][port][k]-values[k] for k in ('tx_bytes','rx_bytes')}
                          for port,values in old['rdma'].items()}
            assert all(v>=0 for p in deltas[node].values() for v in p.values())
        network.append(dict(version=row['version'],nodes=deltas))
    summary=dict(case_id=case['case_id'],series=case['series'],directory=str(directory.relative_to(ROOT)),
                 revision=case['revision'],placements=outcome['node_placements'],
                 training_hosts=outcome['training_hosts'],rollout_hosts=outcome['rollout_hosts'],
                 layout=outcome['layout'],routing=case['routing'],repeat=case['repeat'],
                 p50_ms=statistics.median(samples),p95_ms=percentile(samples,.95),samples_ms=samples,
                 cold_update_ms=bootstrap['end_to_end_update_ms'],verified_versions=len(rows),
                 verified_bootstraps=1,first_regular_update_ms=rows[0]['end_to_end_update_ms'],
                 fp8_wire_bytes=writer_fp8_bytes,trainer_logical_source_bytes=logical_source_bytes,
                 measured_versions=len(samples),training=outcome['training'],rollout=outcome['rollout'],
                 training_layout=list(layouts.values()),writer_bytes=writer_bytes,
                 trainer_injection_bytes=sum(writer_bytes.values()),lowered=lowered,
                 plan=plans[0] if plans else None,entries=entries[0] if entries else None,
                 relay_chains=relay_chains,rdma_bytes=network)
    save(directory/'summary.json',summary)
    return summary

def audit_placement(outcome):
    tp, dp = outcome['rollout']['tp'], outcome['rollout']['dp']
    size = tp * dp
    instances = outcome['rollout']['instances']
    hosts = outcome['rollout_hosts']
    count, extra = divmod(instances, len(hosts))
    counts = [count + (index < extra) for index in range(len(hosts))]
    placement = outcome['node_placements']
    engines = []
    for host, number in zip(hosts, counts):
        gpus = placement[host]['gpus']
        affinity = placement[host]['affinity']
        assert len(gpus) == number * size and len(set(gpus)) == len(gpus)
        for index in range(number):
            selected = gpus[index * size:(index + 1) * size]
            hcas = [affinity[g]['hca'] for g in selected]
            if tp == 4:
                for offset in range(0, size, 4):
                    assert len(set(hcas[offset:offset+4])) == 4
            engines.append(dict(engine=len(engines),host=host,gpus=selected,hcas=hcas))
    assert len(engines) == instances
    return engines

def audit_affinity(directory, outcome):
    engines = audit_placement(outcome)
    expected = {f"rollout-{e['engine']}.log": e for e in engines}
    checked = []
    for item in json.loads((directory/'manifest.json').read_text()):
        with tarfile.open(directory/item['file']) as archive:
            for name in archive.getnames():
                if name not in expected:
                    continue
                engine = expected[name]
                assert item['ip'] == engine['host']
                text = archive.extractfile(name).read().decode(errors='replace')
                records = []
                for match in re.finditer('SHARDSTREAM_RANK_AFFINITY ', text):
                    records.append(json.JSONDecoder().raw_decode(text[match.end():])[0])
                actual = {(r['physical_gpu'],r['hca']) for r in records}
                assert actual == set(zip(engine['gpus'],engine['hcas'])), (name, actual, engine)
                assert all(r['before_nccl_init'] for r in records)
                checked.append(engine['engine'])
    assert sorted(checked) == list(range(len(engines)))
    save(directory/'affinity-audit.json',dict(verified_engines=checked,before_nccl_init=True))
