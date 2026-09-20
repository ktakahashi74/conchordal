"""Independent all-bin kernels and uncached summation for period-bank fixtures."""

from functools import lru_cache
import json
import math
from pathlib import Path
import struct


PERIODS = [0.125 * 2. ** (i/48) for i in range(241)]


@lru_cache(None)
def kernel(delta):
    return tuple((i, value) for i, period in enumerate(PERIODS)
                 if (value := max(0., 1-abs(math.log2(delta/1000/period))/(1/24))) > 0.)


def peaks(masses, separation):
    total = math.fsum(masses)
    probabilities = [x/total if total else 0. for x in masses]
    runs, start = [], 0
    for end in range(1, 242):
        if end < 241 and probabilities[end] == probabilities[start]:
            continue
        if (probabilities[start] > 0 and (start > 0 or end < 241)
                and (not start or probabilities[start] > probabilities[start-1])
                and (end == 241 or probabilities[start] > probabilities[end])):
            runs.append(start)
        start = end
    selected = []
    for i in sorted(runs, key=lambda j: (-probabilities[j], j)):
        if all(abs(i-j)/48 >= separation for j in selected):
            selected.append(i)
            if len(selected) == 8:
                break
    return probabilities, selected


def main():
    root = Path(__file__).resolve().parents[1]
    sequences = []
    for capacity, count in [(8, 2052), (64, 300), (128, 300), (256, 300)]:
        bank, steps = [], []
        end, prefix = 64, 64
        for i in range(count):
            delta = (16 if capacity == 256 else [125, 250, 333, 500][i % 4]) if i else 0
            if i % 97 == 96:
                delta += 4096
            known = delta if i % 11 < 8 or delta < 100 else delta * [90, 89, 50][i % 3] // 100
            end += delta
            prefix += known
            a = dict(start=end-16, end=end, observed_prefix=prefix, weight=[.25, .5, .75, 1.][i % 4])
            received = end + 16
            bank = [old for old in bank if old['end'] >= received - 32000]
            if len(bank) == capacity:
                bank.pop(0)
            bank.append(a)
            checkpoint = None
            if i < 8 or i % 32 == 31 or i+1 in [512, 1024, 2048, count]:
                rounded, full = [[] for _ in PERIODS], [[] for _ in PERIODS]
                supported = nonzero = 0
                for j, newer in enumerate(bank):
                    for older in bank[:j]:
                        dt = newer['end'] - older['end']
                        if (newer['observed_prefix'] - older['observed_prefix']) * 10 < dt * 9:
                            continue
                        supported += 1
                        touched = False
                        for bin_index, value in kernel(dt):
                            exact = newer['weight'] * older['weight'] * value
                            stored = struct.unpack('<f', struct.pack('<f', exact))[0]
                            rounded[bin_index].append(stored)
                            full[bin_index].append(exact)
                            touched |= stored > 0
                        nonzero += touched
                masses = list(map(math.fsum, rounded))
                exact = list(map(math.fsum, full))
                probabilities, best = peaks(masses, 1/24)
                _, exact_best = peaks(exact, 1/24)
                checkpoint = dict(masses=masses, full_f64=exact, probabilities=probabilities, peaks=best,
                                  full_f64_peaks=exact_best, pairs=len(bank)*(len(bank)-1)//2,
                                  supported_pairs=supported, nonzero_pairs=nonzero)
            steps.append(dict(accent=a, received_at=received, checkpoint=checkpoint))
        sequences.append(dict(capacity=capacity, window_samples=32000, sample_rate=1000,
                              rebuild_controls=[512, 1024, 2048] if capacity == 8 else [1024], steps=steps))
    result = dict(schema='temporal-period-grid-fixture-v1', sequences=sequences)
    path = root/'tests/fixtures/temporal_cognition/periods.json'
    path.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps(dict(sequences=len(sequences), inputs=sum(len(s['steps']) for s in sequences),
                          checkpoints=sum(x['checkpoint'] is not None for s in sequences for x in s['steps']),
                          bytes=path.stat().st_size)))


if __name__ == '__main__':
    main()
