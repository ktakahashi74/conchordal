"""Freeze the existing chronological retention oracle for its native port."""

import copy
import hashlib
import json
import math
import struct
from pathlib import Path

from temporal_memory_reference import EpisodeRetention, GapClock, assay_recognition


def encoded(value):
    if isinstance(value, float):
        return {'f64': struct.unpack('<Q', struct.pack('<d', value))[0]}
    if isinstance(value, dict):
        return {key: encoded(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(v) for v in value]
    return value


def generate():
    cases = []
    for clock_capacity, rate_capacity, r_max in ((128, 32, 10.), (128, 2, 100.), (128, 32, .2), (2, 32, 10.)):
        clock = GapClock(1, 100, 10, clock_capacity)
        acquisition = []
        for index in range(40):
            lo, hi = index * 10, (index + 1) * 10
            known = index not in (19, 20, 29)
            raw = dict(epoch=1, sample_rate=100, sample_start=lo, sample_end=hi,
                       available_end=hi/100, observed=known,
                       known_sample_intervals=[(lo, hi)] if known else [])
            clock.observe(raw, hi/100)
            acquisition.append(raw)
        memory = EpisodeRetention(clock, tau_sec=20., kappa=4., strength_max=2.,
                                  r_max=r_max, capacity=4, rate_capacity=rate_capacity)
        operations = []
        frozen = None
        scores = {10: .3, 20: -.2, 30: .4, 40: 1., 50: .3, 999: .5}

        def capture(operation):
            operation.update(records=[memory.record(i) for i in range(memory.capacity)],
                             availability=[memory.availability(i, 4.) for i in range(memory.capacity)],
                             eviction_candidate=memory.eviction_candidate(4.),
                             sequence=memory.sequence, replays=memory.replays, evictions=memory.evictions,
                             totals=list(memory.window.totals), rate_count=memory.window.count,
                             rate_losses=memory.window.losses, rate_invalid=memory.window.invalid,
                             rate_unverified=memory.window.envelope_unverified,
                             rate_history_unknown=memory.window.last_time < memory.window.lost_until,
                             recognition=memory.recognition(scores, -.3, 4.),
                             frozen_recognition=None if frozen is None else assay_recognition(frozen, 1, scores, -.3, 4.25))
            operations.append(operation)

        def write(sequence, end, assignments, new=(), unknown=0., unassigned=0., costs=None):
            coarse = None if costs is None else dict(epoch=1, generation=4, query_id=sequence,
                occurrence_id=sequence, support_id=1000+sequence, support_end=end,
                supporting_audio_end=end, available_end=end+.5, entries=costs)
            w = dict(epoch=1, sequence=sequence, occurrence_id=sequence, support_id=1000+sequence,
                     start=end-.1, support_end=end, committed_at=end+.5, delivered_at=end+.5,
                     support=dict(episodes=assignments, unknown_episode_support=unknown,
                         unassigned_available_support=unassigned,
                         available_support=math.fsum(assignments.values())+unknown+unassigned),
                     coarse_snapshot=coarse)
            accepted = memory.apply(w, 4., new)
            capture(dict(kind='write', write=w, cut=4., new_handles=new, accepted=accepted))
            return w

        write(1, 1., {10: 1.}, (10,))
        write(2, 1., {10: .25, 20: .5}, (20,), unknown=.125, unassigned=.125, costs={10: 0.})
        frozen = memory.assay_snapshot(4.)
        capture(dict(kind='assay', target_start=4.))
        write(3, 1.1, {20: 1.}, costs={10: .3, 20: 0.})
        w = write(4, 1.2, {10: .5, 20: .25}, unknown=.125, unassigned=.125, costs={10: .2, 20: .4})
        assert not memory.apply(w, 4.)
        capture(dict(kind='write', write=copy.deepcopy(w), cut=4., new_handles=[], accepted=False))
        write(5, 1.3, {10: 1.}, costs={10: 0., 20: .1})
        write(6, 1.4, {10: 1.}, costs={10: 0., 20: None})
        write(7, 1.5, {30: .6}, (30,), unknown=.4, costs={10: .2})
        write(8, 1.6, {40: 1.}, (40,), costs={10: .1, 20: .2, 30: .3})
        victim = memory.eviction_candidate(4.)
        handle = memory.evict(victim)
        capture(dict(kind='evict', slot=victim, handle=handle))
        write(9, 1.7, {50: 1.}, (50,), costs={10: .2, 20: .2, 30: .2, 40: .2})
        write(10, 2.8, {50: 1.}, costs={10: .2, 20: .2, 30: .2, 40: .2})
        write(11, 3.5, {50: .5}, unknown=.25, unassigned=.25, costs={10: .2})
        cases.append(dict(clock_capacity=clock_capacity, rate_capacity=rate_capacity, r_max=r_max,
                          acquisitions=acquisition, operations=operations, scores=scores, bias=-.3, query_end=4.25))
    root = Path(__file__).resolve().parents[1]
    sources = ['scripts/temporal_memory_reference.py', 'scripts/temporal_occurrence_reference.py',
               'scripts/temporal_cognition_reference.py']
    return encoded(dict(sources={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}, cases=cases))


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1] / 'tests/fixtures/temporal_cognition/retention.json'
    target.write_text(json.dumps(generate(), indent=2) + '\n')
