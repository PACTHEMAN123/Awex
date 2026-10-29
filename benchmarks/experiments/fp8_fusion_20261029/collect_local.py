"""Download and independently audit each completed paired launch."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import tarfile
import time

import evidence

REMOTE = '/tmp/fp8-fusion-resume-1029'
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
       '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
       '-o', 'ControlPath=/tmp/fp8-resume-1029-ssh-%C', 'hole-11.18.56.89']


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def audit(root):
    evidence.ROOT = root
    runs = []
    for marker in sorted(root.glob('tp2x4-*/download-verified.json')):
        directory = marker.parent
        expected = load(directory / 'remote-summary.json')
        evidence.audit_affinity(directory, load(directory / 'outcome.json'))
        actual = evidence.analyze(directory)
        actual.update(mode=expected['mode'], replicas=4, native_sha256=expected['native_sha256'])
        assert json.loads(json.dumps(actual)) == expected
        save(directory / 'summary.json', actual)
        runs.append(actual)
    assert len({(r['mode'], r['repeat']) for r in runs}) == len(runs)
    pairs = []
    for repeat in range(3):
        pair = {r['mode']: r for r in runs if r['repeat'] == repeat}
        if len(pair) != 2:
            continue
        a, b = pair['fused'], pair['staged']
        for key in ['native_sha256', 'writer_bytes', 'fp8_wire_bytes', 'lowered',
                    'relay_chains', 'training', 'rollout']:
            assert a[key] == b[key], (repeat, key)
        assert load(root / a['directory'] / 'transport-fingerprints.json') == load(
            root / b['directory'] / 'transport-fingerprints.json')
        pairs.append(dict(repeat=repeat, fused_p50_ms=a['p50_ms'], staged_p50_ms=b['p50_ms'],
                          staged_over_fused=b['p50_ms'] / a['p50_ms']))
    points = []
    for mode in ['fused', 'staged']:
        selected = [r for r in runs if r['mode'] == mode]
        if selected:
            points.append(dict(mode=mode, launches=len(selected),
                               p50_ms=statistics.median(r['p50_ms'] for r in selected),
                               p50_min_ms=min(r['p50_ms'] for r in selected),
                               p50_max_ms=max(r['p50_ms'] for r in selected),
                               p95_ms=statistics.median(r['p95_ms'] for r in selected)))
    result = dict(complete=len(pairs) == 3, launches=len(runs), timed_updates=len(runs) * 15,
                  verified_regular_updates=len(runs) * 20, verified_bootstraps=len(runs),
                  receiver_checks=len(runs) * 21 * 8, node_archives=len(runs) * 4,
                  points=points, pairs=pairs)
    save(root / 'audited-results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'case-archives').mkdir(exist_ok=True)
    previous = None
    while True:
        try:
            code = ("import json;from pathlib import Path;"
                    f"p=Path({REMOTE + '/results'!r});"
                    "a=p/'active.json';print(json.dumps(dict(active=json.loads(a.read_text()) "
                    "if a.exists() else {},cases=[d.name for d in p.glob('tp2x4-*') "
                    "if (d/'summary.json').exists()])))")
            snapshot = json.loads(subprocess.check_output(SSH + ['python3 -c ' + shlex.quote(code)],
                                                         text=True, timeout=60))
            save(root / 'remote-state.json', snapshot)
            for label in snapshot['cases']:
                assert Path(label).name == label
                directory = root / label
                if (directory / 'download-verified.json').exists():
                    continue
                archive = root / 'case-archives' / (label + '.tar.gz')
                with archive.with_suffix('.partial').open('wb') as stream:
                    subprocess.run(SSH + ['tar -cz -C ' + REMOTE + '/results ' + shlex.quote(label)],
                                   stdout=stream, check=True, timeout=900)
                archive.with_suffix('.partial').replace(archive)
                with tarfile.open(archive) as bundle:
                    for member in bundle.getmembers():
                        path = Path(member.name)
                        assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == label
                        assert member.isfile() or member.isdir()
                    bundle.extractall(root)
                save(directory / 'remote-summary.json', load(directory / 'summary.json'))
                save(directory / 'download-verified.json', dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
                result = audit(root)
                print(json.dumps(dict(downloaded=label, audit=result)), flush=True)
            if previous != snapshot['active']:
                print(json.dumps(snapshot['active']), flush=True)
                previous = snapshot['active']
            if snapshot['active'].get('state') in ('complete', 'failed', 'stopped'):
                print(json.dumps(audit(root)), flush=True)
                return
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            print('RETRIEVAL_RETRY ' + str(error), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
