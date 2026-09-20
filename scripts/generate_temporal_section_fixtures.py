#!/usr/bin/env python3
"""Freeze bounded section records from the independent Python reference."""

import copy
import hashlib
import json
import math
from pathlib import Path

import temporal_section_reference as ref


def assignment(kind):
    result = dict(status="match", cost=.1, supported=True, search_completed=True,
                  search_covered=True, search_nonempty=True,
                  frequency_shift_log2=0., tempo_shift_log2=0.)
    result.update(({}, {"cost": .5}, {"status": "no_memory", "cost": None},
                   {"status": "no_memory", "cost": None}, {"status": "unknown"})[kind])
    return result


def activity(start, end, index):
    duration = end - start
    known = duration * (0.5 if index % 7 == 3 else 1.)
    assigned = known * (0.25 if index % 3 == 1 else 1.)
    valid = [known] * 9
    denominators = [assigned] * 9
    numerators = [0.] * 9
    numerators[index % 4] = assigned
    numerators[4:] = [0.5, assigned * (index % 2), assigned * .25,
                      assigned * .125, assigned * (index % 2)]
    if index % 5 == 2:
        denominators[6] = valid[6] = numerators[6] = 0.
    return dict(window=[start, end], numerators=numerators, denominators=denominators,
                physical_valid_seconds=valid, assignment_seconds=assigned,
                physical_window_seconds=duration)


def checkpoint(history):
    snapshot = history.snapshot()
    c, r = snapshot['cumulative']['values'], snapshot['recent']['values']
    elapsed = history.cumulative.time(5) - history.cumulative.time(4)
    raw = c + [None if a is None or b is None else b-a for a, b in zip(c, r)]
    raw += [math.log1p(elapsed), snapshot['missing_fraction'], None, None]
    return dict(cumulative_hex=history.cumulative.buffer.hex(),
                ring_hex=[r.buffer.hex() for r in history.ring],
                cumulative=c, recent=r, covariates=raw)


def generate():
    cases = []
    for ring in (2, 4, 8):
        for gap in (False, True):
            history = ref.SectionHistory(1, 4, 19, 0., ring)
            steps = []
            for index in range(25):
                start = index * .5
                if gap and index % 4 == 1:
                    start += .125
                end = (index+1) * .5
                a = activity(start, end, index)
                history.observe(a, 1, 4)
                steps.append(dict(kind="observe", activity=a, expected=checkpoint(history)))
                kind = index % 5
                record = dict(epoch=1, record_kind="observed_commit", occurrence_id=index+1,
                              start=start, support_end=end,
                              assignment_seconds=a['assignment_seconds'],
                              membership=.5 if index % 3 == 0 else 1., ordering_known=index % 11 != 5,
                              ending_descriptor=[2. if kind == 2 else 0.] * 6,
                              assignment=assignment(kind), ending_generation=4)
                coverage = .8 if index % 6 == 2 else 1.
                history.commit(record, a, index+1, coverage)
                step = dict(kind="commit", record=record, activity=a,
                            sequence=index+1, adjacency=coverage, expected=checkpoint(history))
                steps.append(step)
                if index % 4 == 0:
                    assert not history.commit(record, a, index+1, coverage)
                    steps.append({**copy.deepcopy(step), "replayed": True})
            cases.append(dict(name=f"ring{ring}_gap{int(gap)}", ring_size=ring, steps=steps))
    categories = []
    variations = [dict(cost=.25, frequency_shift_log2=1/48), dict(cost=.25001, tempo_shift_log2=None),
                  dict(frequency_shift_log2=.1, tempo_shift_log2=None), dict(tempo_shift_log2=None),
                  dict(cost=1.), dict(cost=None), dict(bound_hit=True), dict(supported=False),
                  dict(ambiguous_cutoff=True), dict(search_nonempty=False), dict(cost=1.01),
                  dict(status="no_memory", cost=None), dict(status="pending_unheard"),
                  dict(status="no_memory", search_completed=False),
                  dict(status="no_memory", search_covered=False)]
    for changes in variations:
        a = {**assignment(0), **changes}
        for ending, predecessor in [([0.]*6, None), ([1.]*6, [0.]*6),
                                    ([1.1]+[None]*5, [0.]*6), ([None]*6, [0.]*6)]:
            categories.append(dict(assignment=a, ending=ending, predecessor=predecessor,
                                   expected=ref.correspondence_category(a, ending, predecessor)))
    endings = []
    for variant in range(8):
        hops, accents = [], []
        for i in range(10):
            spectrum = [float(1+i%3),float(2+i%2)]
            if variant == 1: spectrum = [0.,0.]
            hop = dict(epoch=0,generation=2,start=i/4,end=(i+1)/4,observed=not (variant == 2 and i%3 == 1),
                       association_known=True,energy=sum(spectrum),spectrum=spectrum,
                       rise=None if variant == 3 and i%4 == 1 else float(i%3),flux=float(i%2))
            if variant == 4 and i%3 == 0: hop['spectrum'] = None
            hops.append(hop)
            if i in (2,5,8):
                accents.append(dict(epoch=0,generation=2,id=i,time=(i+1)/4,weight=.25,
                                    event_interval=[i/4,(i+1)/4],raw_support_end=(i+2)/4,
                                    available_end=(i+2)/4))
        for span_start,cut in [(0.,2.5),(.375,1.75),(1.8,2.5),(0.,1.875)]:
            generation_start = .5 if variant == 5 else 0.
            span_start = max(span_start,generation_start)
            evicted = cut-.25 if variant == 6 else None
            if variant == 7: accents = []
            expected = ref.ending_descriptor(hops,accents,[1.,3.],0,2,0.,generation_start,
                                              span_start,cut,[0.]*6,[1.]*6,evicted)
            endings.append(dict(hops=hops,accents=accents,span_start=span_start,cut=cut,
                                generation_start=generation_start,evicted=evicted,expected=expected))
    return dict(schema="temporal-section-fixtures-v1",
                reference_source_sha256=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
                cases=cases, categories=categories,endings=endings)


if __name__ == '__main__':
    output = Path('tests/fixtures/temporal_cognition/sections.json')
    data = generate()
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(cases=len(data['cases']), categories=len(data['categories']),
                         checkpoints=sum(len(c['steps']) for c in data['cases']), bytes=output.stat().st_size)))
