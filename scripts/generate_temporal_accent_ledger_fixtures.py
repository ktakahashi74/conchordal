"""Finite ordered-delivery traces from the accent reference, including cap loss."""

import copy
import hashlib
import json
from pathlib import Path

from temporal_accent_reference import AccentLedger


def event(i):
    start = i * 64
    return dict(epoch=2, generation=3, id=(2, 3, start+32, start+48),
                sample_rate=1, event_interval=[start+32, start+48], time=start+48,
                raw_support_intervals=[(start+j*16, start+(j+1)*16) for j in range(4)],
                raw_support_end=start+64, available_end=start+64,
                weight=[.25, .5, .75][i % 3])


def main():
    root = Path(__file__).resolve().parents[1]
    sequences = []
    for capacity in (2, 64, 128, 256):
        ledger = AccentLedger(2, 3, capacity, 32000)
        steps = []

        def apply(action, clock, accent=None):
            before = copy.deepcopy(ledger.__dict__)
            try:
                delivery = ledger.deliver(accent, clock) if action == 'deliver' else ledger.advance(clock)
                result = 'delivered' if delivery is not None else 'none'
            except ValueError:
                assert ledger.__dict__ == before
                delivery, result = None, 'error'
            start = max(0, ledger.observed_end - 32000)
            snapshots = [ledger.snapshot(s) for s in sorted(set([
                start, max(start, ledger.observed_end - 512), ledger.observed_end,
                ledger.bank[0]['time'] if ledger.bank else start]))]
            summaries = [dict(start=int(s['window'][0]), end=int(s['window'][1]), capacity_valid=s['capacity_valid'],
                              weight=s['weight'], count=s['cumulative_admission_count'],
                              cumulative_weight=s['cumulative_admission_weight'],
                              evicted=s['capacity_evicted_through'],
                              accents=[[a['event_interval'][0], a['event_interval'][1], a['weight']]
                                       for a in s['accents']]) for s in snapshots]
            steps.append(dict(action=action, received_at=clock, accent=accent, result=result,
                              sequence=delivery['sequence'] if delivery else None, snapshots=summaries))

        for i in range(300):
            a = event(i)
            if i % 37 == 0:
                apply('deliver', a['time'], a)
            apply('deliver', a['available_end'], a)
            if i % 43 == 0:
                apply('deliver', a['available_end'], copy.deepcopy(a))
                bad = copy.deepcopy(a)
                bad.update(raw_support_end=a['raw_support_end']+8, available_end=a['available_end']+8)
                bad['raw_support_intervals'][-1] = (a['time'], bad['raw_support_end'])
                apply('deliver', bad['available_end'], bad)
            if i == 280:
                apply('deliver', a['available_end'], event(0))
        apply('advance', ledger.observed_end + 32000)
        apply('deliver', ledger.observed_end, event(300))
        apply('advance', ledger.observed_end + 16)
        sequences.append(dict(capacity=capacity, window_samples=32000, steps=steps))
    reference = root / 'scripts/temporal_accent_reference.py'
    result = dict(schema='temporal-accent-ledger-fixture-v1',
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), sequences=sequences)
    target = root / 'tests/fixtures/temporal_cognition/accent-ledger.json'
    target.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps(dict(sequences=len(sequences), operations=sum(len(s['steps']) for s in sequences),
                          reference_sha256=result['reference_sha256'], fixture_bytes=target.stat().st_size)))


if __name__ == '__main__':
    main()
