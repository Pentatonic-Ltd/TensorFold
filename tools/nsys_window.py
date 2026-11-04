#!/usr/bin/env python3
"""Rank Nsight Systems kernels and API calls within a UTC request window.

Export with: nsys export --type sqlite --output trace.sqlite trace.nsys-rep
Then: python3 tools/nsys_window.py trace.sqlite --start-ns UTC_NS --end-ns UTC_NS
Kernel sums can exceed wall time with overlap; busy time is an interval union.
A gap is time without device activity, not proof of a host bottleneck.
"""
import argparse
import collections
import csv
import json
import pathlib
import sqlite3


def union_ns(intervals):
    total = 0
    right = None
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        total += max(0, hi - max(lo, right if right is not None else lo))
        right = max(hi, right if right is not None else hi)
    return total


def summarize(db, start_ns, end_ns):
    if end_ns <= start_ns:
        raise ValueError('end must be after start')
    epoch = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    lo, hi = start_ns - epoch, end_ns - epoch
    names = dict(db.execute('SELECT id, value FROM StringIds'))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernels = []
    device_intervals = []
    ranked = collections.defaultdict(lambda: [0, 0])
    for start, end, name, correlation in db.execute(
        'SELECT start,end,demangledName,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start < ? AND end > ?', (hi, lo)
    ):
        a, b = max(start, lo), min(end, hi)
        kernels.append((a, b))
        device_intervals.append((a, b))
        ranked[names[name]][0] += b - a
        ranked[names[name]][1] += 1
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in tables:
            device_intervals.extend((max(a, lo), min(b, hi)) for a, b in db.execute(
                f'SELECT start,end FROM {table} WHERE start < ? AND end > ?', (hi, lo)))
    api = collections.defaultdict(lambda: [0, 0])
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        for a, b, name in db.execute(f'SELECT start,end,nameId FROM {table} WHERE start < ? AND end > ?', (hi, lo)):
            api[names[name]][0] += min(b, hi) - max(a, lo)
            api[names[name]][1] += 1
    kernel_sum = sum(v[0] for v in ranked.values())
    wall = hi - lo
    return {
        'start_ns': start_ns, 'end_ns': end_ns, 'wall_ms': wall / 1e6,
        'kernel_sum_ms': kernel_sum / 1e6, 'kernel_busy_ms': union_ns(kernels) / 1e6,
        'device_busy_ms': union_ns(device_intervals) / 1e6,
        'device_idle_ms': (wall - union_ns(device_intervals)) / 1e6,
        'kernel_count': len(kernels),
        'kernels': [{'name': k, 'ms': v[0] / 1e6, 'count': v[1],
                     'kernel_share_pct': v[0] * 100 / max(kernel_sum, 1),
                     'wall_share_pct': v[0] * 100 / wall}
                    for k, v in sorted(ranked.items(), key=lambda x: -x[1][0])],
        'apis': [{'name': k, 'ms': v[0] / 1e6, 'count': v[1]}
                 for k, v in sorted(api.items(), key=lambda x: -x[1][0])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=pathlib.Path)
    parser.add_argument('--start-ns', type=int, required=True)
    parser.add_argument('--end-ns', type=int, required=True)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    with sqlite3.connect(f'file:{args.sqlite}?mode=ro', uri=True) as db:
        result = summarize(db, args.start_ns, args.end_ns)
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + '\n')
        with args.output.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'ms', 'count', 'kernel_share_pct', 'wall_share_pct'])
            writer.writeheader()
            writer.writerows(result['kernels'])
    else:
        print(output)


if __name__ == '__main__':
    main()
