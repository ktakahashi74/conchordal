"""Replay registered Python references as independent Rust feature fixtures."""

import copy
import hashlib
import json
from pathlib import Path
import random

from temporal_accent_reference import accent_window
from temporal_descriptor_reference import raw_descriptor


def main():
    root = Path(__file__).resolve().parents[1]
    sequences = []
    for seed in range(24):
        rng = random.Random(seed)
        cfg = dict(means=[0., 0.], deviations=[1., 1.], threshold=1.,
                   rms_floor=[1e-6, .5e-6, 2e-6][seed % 3])
        if seed >= 16:
            cfg.update(means=[.25, .5], deviations=[.5, 2.])
        hops, steps, end = [], [], 0
        for index in range(48):
            start = end + (10 if seed % 4 == 1 and index == 15 else 0)
            end = start + 10
            energy = 2. ** (2 * rng.choice([-24, -2, 0, 1, 2, 3]))
            spectrum = [energy * rng.choice([0., .25, .5, 1.]) for _ in range(9)]
            hop = dict(sample_rate=1, sample_start=start, sample_end=end,
                       raw_support_start=start, raw_support_end=end, available_end=end,
                       epoch=2, generation=3, association_handle=8, grid_id=1,
                       observed=True, association_known=True, energy=energy,
                       bus_energy=energy * 2, spectrum=spectrum,
                       known_sample_intervals=[(start, end)])
            if seed < 8:
                # Exact-rise plateau and known silence provide interpretable peak cases.
                e = [1., 1., 16., 256., 256., 0.][index % 6]
                hop.update(energy=e, bus_energy=2*e, spectrum=[e / 9] * 9)
            elif index % 13 == 4:
                variant = seed % 8
                if variant == 0:
                    hop.update(observed=False, known_sample_intervals=[], energy=None,
                               bus_energy=None, spectrum=None)
                elif variant == 1:
                    hop['known_sample_intervals'] = [(start, start+4), (start+6, end)]
                elif variant == 2:
                    hop['association_known'] = False
                elif variant == 3:
                    hop['energy'] = None
                elif variant == 4:
                    hop['spectrum'] = None
                elif variant == 5:
                    hop['spectrum'] = [0.] * 9
                elif variant == 6:
                    hop.update(energy=0., bus_energy=0., spectrum=[0.] * 9)
                else:
                    hop['observed'] = False
            if seed >= 8 and index >= 24:
                hop[['generation', 'association_handle', 'grid_id'][seed % 3]] += 1
            raw = raw_descriptor(hop, [6. + i / 2 for i in range(9)], end,
                                 hops[-1] if hops else None)
            hops.append(copy.deepcopy(hop))
            detection = accent_window(hops[-4:], cfg['means'], cfg['deviations'],
                                      cfg['threshold'], cfg['rms_floor']) if len(hops) >= 4 else None
            steps.append(dict(input=hop, raw=raw, detection=detection))
        sequences.append(dict(name=f'feature-{seed}', config=cfg, steps=steps))
    refs = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root / 'scripts/temporal_accent_reference.py',
                      root / 'scripts/temporal_descriptor_reference.py']}
    output = dict(schema='temporal-raw-feature-fixture-v1', references=refs, sequences=sequences)
    target = root / 'tests/fixtures/temporal_cognition/features.json'
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(sequences=len(sequences), endpoints=sum(len(s['steps']) for s in sequences),
                         admitted=sum(x['detection'] is not None and x['detection']['accent'] is not None
                                      for s in sequences for x in s['steps']), references=refs)))


if __name__ == '__main__':
    main()
