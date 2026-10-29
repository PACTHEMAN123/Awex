import os,re,subprocess,time
from pathlib import Path
import ray

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

