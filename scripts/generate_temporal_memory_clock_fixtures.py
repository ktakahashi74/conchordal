"""Freeze the existing gap-clock oracle against an independent sample union."""

import hashlib
import json
import struct
from pathlib import Path

from temporal_memory_reference import GapClock


def f64_bits(value):
    return {'f64': struct.unpack('<Q', struct.pack('<d', value))[0]}


def generate():
    cases = []
    for rate, hop, capacity, origin, length in (
        (100, 10, 3, 0, 24),
        (100, 10, 8, 5000, 24),
        (48000, 512, 128, 0, 140),
        (48000, 512, 2, 512000, 24),
    ):
        clock = GapClock(7, rate, hop, capacity, origin)
        physical = []
        steps = []

        def record(operation):
            queries = []
            for position in (0., len(physical) / 3, max(0., len(physical) - .375 * hop), len(physical)):
                time = (origin + position) / rate
                bounds = clock.prefix(time)
                full = int(position)
                known = sum(physical[:full])
                if full < len(physical) and physical[full]:
                    known += position - full
                actual = (position - known) / rate
                assert bounds['lower'] <= actual + 1e-10
                assert bounds['upper'] + 1e-10 >= actual
                queries.append([f64_bits(time), f64_bits(bounds['lower']), f64_bits(bounds['upper']), bounds['history_lost'], f64_bits(actual)])
            operation.update(queries=queries, missing=f64_bits(clock.missing), count=clock.count, evictions=clock.evictions)
            steps.append(operation)

        for index in range(length):
            if index % 9 == 2:
                end = clock.end + 2 * hop
                clock.gap(end, end / rate)
                physical.extend([False] * (2 * hop))
                record(dict(kind='gap', end=end, cut=end))
                continue
            skip = hop if index % 7 == 3 else 0
            start, end = clock.end + skip, clock.end + skip + hop
            physical.extend([False] * skip)
            bits = [((i // max(1, hop // 7) + index) % 4 != 0) for i in range(hop)]
            if index % 5 == 0:
                bits = [True] * hop
            if index % 6 == 2:
                bits = [False] * hop
            intervals = []
            for i, known in enumerate(bits):
                if known:
                    if intervals and intervals[-1][1] == start + i:
                        intervals[-1][1] += 1
                    else:
                        intervals.append([start + i, start + i + 1])
            if intervals:
                intervals.append(intervals[0].copy())
            available, cut = end + hop // 8, end + hop // 4
            raw = dict(epoch=7, sample_rate=rate, sample_start=start, sample_end=end,
                       available_end=available / rate, observed=any(bits), known_sample_intervals=intervals)
            assert clock.observe(raw, cut / rate)
            physical.extend(bits)
            record(dict(kind='observe', start=start, end=end, available=available, cut=cut,
                        observed=any(bits), intervals=intervals, accepted=True))
            if index % 11 == 0:
                raw['known_sample_intervals'] = list(reversed(intervals))
                assert not clock.observe(raw, (cut + 1) / rate)
                record(dict(kind='observe', start=start, end=end, available=available, cut=cut + 1,
                            observed=any(bits), intervals=raw['known_sample_intervals'], accepted=False))
        cases.append(dict(rate=rate, hop=hop, capacity=capacity, origin=origin, steps=steps))
    source = Path(__file__).with_name('temporal_memory_reference.py')
    return dict(source=str(source.relative_to(Path(__file__).resolve().parents[1])),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases)


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'tests/fixtures/temporal_cognition/memory-clock.json'
    path.write_text(json.dumps(generate(), indent=2) + '\n')
