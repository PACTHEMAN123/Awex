"""Recompute accepted results from per-rank raw measurements, fail on gaps."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def audit(directory, size):
    assert read(directory / 'active.json')['state'] == 'complete'
    inventory = read(directory / 'inventory.json')
    assert len(inventory) == 4 and all(not r['gpu_processes'].strip() for r in inventory)
    assert len({json.dumps(r['wb_python_sha256'], sort_keys=True) for r in inventory}) == 1
    summaries = {r['case']: r for r in read(directory / 'summary.json')}
    assert len(summaries) == 9
    measured, plans, phases = {}, {}, {}
    for mode in ['bf16', 'gather_fp8', 'prepared_fp8']:
        measured[mode], phases[mode] = [], []
        for repeat in range(3):
            case = f'{mode}-r{repeat}'
            raw = read(directory / f'{case}-raw.json')
            initialized = read(directory / f'{case}-initialized.json')
            assert len(initialized) == 4 and len({r['host'] for r in initialized}) == 4
            assert all(r['cuda_visible_devices'] == '0' for r in initialized)
            assert len(raw) == 4
            assert {(r['role'], r['rank']) for r in raw} == {
                (role, rank) for role in ['sender', 'receiver'] for rank in [0, 1]}
            for worker in raw:
                assert [r['epoch'] for r in worker['rows']] == list(range(20))
                if worker['role'] == 'receiver':
                    expected = 2 * size**2 + (2 * (size // 128)**2 if mode != 'bf16' else 0)
                    assert all(r['verified'] and r['checked_elements'] == expected for r in worker['rows'])
            senders = sorted((r for r in raw if r['role'] == 'sender'), key=lambda r: r['rank'])
            samples = [max(r['rows'][i]['complete_ms'] for r in senders) for i in range(5, 20)]
            p50 = statistics.median(samples)
            assert p50 == summaries[case]['p50_ms']
            measured[mode].append(p50)
            plans[case] = [r['plan'] for r in senders]
            wire = 4 * size**2 if mode == 'bf16' else 2 * size**2 + 8 * (size // 128)**2
            assert sum(r['plan']['native_wire_bytes'] for r in senders) == wire
            if mode == 'gather_fp8':
                for r in senders:
                    for row in r['rows']:
                        assert row['logical_gather_egress_bytes'] == 2 * size**2
                        assert row['prepare_ms'] >= row['gather_ms'] + row['quantize_ms']
                        assert row['complete_ms'] >= row['prepare_ms'] + row['native_transfer_ms']
                phases[mode].append({key: statistics.median([
                    max(r['rows'][i][key] for r in senders) for i in range(5, 20)])
                    for key in ['gather_ms', 'quantize_ms', 'prepare_ms', 'native_transfer_ms']})
            if mode == 'prepared_fp8':
                assert all(r['rows'][i]['prepare_outside_timing']['logical_gather_egress_bytes'] == 2 * size**2
                           for r in senders for i in range(20))
            after = read(directory / f'{case}-final-inventory.json')
            assert len(after) == 4 and all(not r['gpu_processes'].strip() for r in after)
    for repeat in range(3):
        assert plans[f'gather_fp8-r{repeat}'] == plans[f'prepared_fp8-r{repeat}']
    return dict(size=size, original_mib=4 * size**2 / 2**20,
        p50_ms={m: statistics.median(v) for m, v in measured.items()},
        launch_p50_ms=measured,
        phase_ms={k: statistics.median([p[k] for p in phases['gather_fp8']])
                  for k in phases['gather_fp8'][0]},
        paired_gather_over_bf16=[a / b for a, b in zip(measured['gather_fp8'], measured['bf16'])],
        extra_gather_mib=4 * size**2 / 2**20,
        fp8_wire_mib=(2 * size**2 + 8 * (size // 128)**2) / 2**20,
        timed_updates=135, verified_receiver_updates=360, native_fp8_plans_identical=True,
        source_revision=inventory[0]['revision'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    rows = [audit(args.root / f'matrix{size}-v1', size) for size in [8192, 16384]]
    (args.root / 'audited-results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report = ['# WeightBridge Pre-Transfer FP8: Experimental Extension', '',
        'Four H20 hosts, one GPU each: two trainer shards and two full-tensor receivers.',
        'Frameworkless matrices, not full-model Figure 9/10/13 results.', '',
        '| BF16 payload | Native BF16 | Gather + quantize + RDMA | Prepared FP8 RDMA only |',
        '|---|---:|---:|---:|']
    for row in rows:
        p = row['p50_ms']
        report.append(f"| {row['original_mib']:.0f} MiB | {p['bf16']:.3f} ms | "
                      f"{p['gather_fp8']:.3f} ms | {p['prepared_fp8']:.3f} ms |")
    report += ['', 'Numbers are medians of three independent launch p50s, each with 5 fixed warmups',
        'and 15 timed updates. All 720 receiver-update full-value checks passed across',
        '18 launches; 270 timed publications. The separate 16 MiB smoke added 10 checks.',
        'Prepared FP8 excludes gathering and quantization and is a lower bound only.', '',
        '## Preparation Phases', '', '| BF16 payload | Gather/reconstruct | Quantize/stage |',
        '|---|---:|---:|']
    for row in rows:
        report.append(f"| {row['original_mib']:.0f} MiB | {row['phase_ms']['gather_ms']:.3f} ms | "
                      f"{row['phase_ms']['quantize_ms']:.3f} ms |")
    report += ['', 'Phase values use per-update maximum rank durations, then launch and replicate medians;',
        'they need not add to the aggregate completion median.', '', '## Traffic', '']
    for row in rows:
        report.append(f"- {row['original_mib']:.0f} MiB BF16: {row['extra_gather_mib']:.0f} MiB extra "
                      f"trainer collective payload + {row['fp8_wire_mib']:.5f} MiB native FP8 trainer egress.")
    report += ['', 'Traffic counts useful payload, not NIC counters or protocol overhead. Receiver relay',
        'traffic is separate. Native FP8 routes/bytes match the transfer-only control.', '',
        '## Boundaries', '',
        '- This extension gathers and quantizes full tensors on every trainer before native send.',
        '- Preparation is not overlapped with RDMA or bounded per internal native round.',
        '- Only uniform, block-aligned 2-D row/column BF16 partitions are supported.',
        '- Full gathering follows the requested proposal; block-aligned FP8 can also be computed locally.',
        '- This is not a lower bound on an optimized quantized WeightBridge implementation.',
        '- No real-model accuracy, vLLM generation, TP/EP/DP mixed-layout integration is established.',
        '- No ShardStream performance comparison is made from these microbenchmarks.', '',
        'See PROTOCOL.md, per-rank raw JSON, initialization records, source bundle and logs.',
        'All accepted cases released their actors and verified all four hosts GPU-idle.', '']
    (args.root / 'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
