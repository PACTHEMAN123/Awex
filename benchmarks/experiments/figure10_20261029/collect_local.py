"""Retrieve completed attempts and independently recompute Figure 10 points."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import tarfile
import time

from protocol import BASE, audit, load, save


def summarize(root):
    runs = []
    for marker in sorted((root / 'evidence').glob('*/complete.json')):
        result = audit(marker.parent)
        assert result == load(marker.parent / 'summary.json')
        runs.append(result)
    assert len({(row['config'], row['repeat']) for row in runs}) == len(runs)
    b = load(root / 'reference-b-audit.json')
    points = []
    for config in 'BFG':
        selected = b['runs'] if config == 'B' else [row for row in runs if row['config'] == config]
        if not selected:
            continue
        points.append(dict(config=config, launches=len(selected),
                           complete_three_launches={row['repeat'] for row in selected} == {0, 1, 2},
                           p50_ms=statistics.median(row['p50_ms'] for row in selected),
                           p50_min_ms=min(row['p50_ms'] for row in selected),
                           p50_max_ms=max(row['p50_ms'] for row in selected),
                           p95_ms=statistics.median(row['p95_ms'] for row in selected),
                           evidence='Figure 9 archived B launches' if config == 'B' else 'New Figure 10 launches'))
    complete = len(runs) == 6 and all(row['complete_three_launches'] for row in points)
    result = dict(complete=complete, new_launches=len(runs), new_measured_updates=len(runs) * 30,
                  new_verified_updates=len(runs) * 35, new_rollout_parameter_checks=len(runs) * 560,
                  new_archives_verified=len(runs) * 4, new_checkpoint_guards=len(runs) * 32,
                  round_cap_bytes=2**31, logical_tensor_cap_bytes=128 * 2**20, points=points, runs=runs)
    save(root / 'audited-results.json', result)
    lines = ['# WeightBridge Figure 10', '',
             f"New launches audited: {len(runs)}/6. Status: {'complete' if complete else 'partial'}.", '',
             '| Config | Launches | p50 (ms) | Launch p50 range (ms) | p95 (ms) | Evidence |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for point in points:
        lines.append(f"| {point['config']} | {point['launches']} | {point['p50_ms']:.6f} | "
                     f"{point['p50_min_ms']:.6f}-{point['p50_max_ms']:.6f} | {point['p95_ms']:.6f} | {point['evidence']} |")
    lines += ['', 'Settings match Figure 9: native WeightBridge, 2 GiB round cap, 128 MiB logical',
              'tensor cap, 5 warmups and 30 timed complete BF16 updates per launch.',
              'Each aggregate is the median of launch statistics; the range is descriptive,',
              'not a confidence interval. B is reused unchanged from Figure 9 and was not',
              'interleaved with F/G. See PROTOCOL.md for layout and harness provenance.', '',
              'F uses TP4/PP1/CP2/EP8; G uses TP2/PP4/CP2/EP4. Both use 16 training',
              'and 16 rollout GPUs across four H20 hosts, with four TP4 rollout engines.',
              'These are parameter-publication latencies, not RL throughput. No manuscript',
              'file is modified by this campaign.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    (root / 'case-archives').mkdir(exist_ok=True)
    (root / 'evidence').mkdir(exist_ok=True)
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
           '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
           '-o', 'ControlPath=/tmp/wbfig10-1029-ssh-%C', 'hole-11.18.56.89']
    code = ('from pathlib import Path; import json; '
            f'p=Path({str(BASE / "results")!r}); a=p/"active.json"; '
            'state=json.loads(a.read_text()) if a.exists() else {}; '
            'print(json.dumps(dict(active=state,attempts=[d.name for d in p.glob("*-attempt*") '
            'if (d/"complete.json").exists() or '
            '(state.get("state") in ("needs-investigation","stopped") and (d/"final-inventory.json").exists())])))')
    previous = None
    while True:
        try:
            snapshot = json.loads(subprocess.check_output(ssh + ['python3 -c ' + shlex.quote(code)], text=True, timeout=90))
            save(root / 'remote-state.json', snapshot)
            changed = False
            for label in snapshot['attempts']:
                assert Path(label).name == label
                destination = root / 'evidence' / label
                if (destination / 'download-verified.json').exists():
                    continue
                archive = root / 'case-archives' / (label + '.tar.gz')
                temporary = archive.with_suffix('.partial')
                with temporary.open('wb') as stream:
                    subprocess.run(ssh + ['tar -cz -C ' + shlex.quote(str(BASE / 'results')) + ' ' + shlex.quote(label)],
                                   stdout=stream, check=True, timeout=300)
                temporary.replace(archive)
                with tarfile.open(archive) as bundle:
                    for member in bundle.getmembers():
                        path = Path(member.name)
                        assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == label
                        assert member.isfile() or member.isdir()
                    bundle.extractall(root / 'evidence')
                for row in load(destination / 'manifest.json'):
                    assert hashlib.sha256((destination / row['file']).read_bytes()).hexdigest() == row['sha256']
                if (destination / 'complete.json').exists():
                    assert audit(destination) == load(destination / 'summary.json')
                save(destination / 'download-verified.json', dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
                print('DOWNLOADED_AND_AUDITED ' + label, flush=True)
                changed = True
            if changed:
                summarize(root)
            if snapshot['active'] != previous:
                print(json.dumps(snapshot['active']), flush=True)
                previous = snapshot['active']
            if args.once or snapshot['active'].get('state') in ('complete', 'needs-investigation', 'stopped'):
                summarize(root)
                return
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
            print('RETRIEVAL_RETRY ' + str(error), flush=True)
            if args.once:
                raise
        time.sleep(30)


if __name__ == '__main__':
    main()
