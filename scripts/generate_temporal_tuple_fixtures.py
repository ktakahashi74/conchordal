"""Check bounded tuple selection using full Cartesian ranking, not heap traversal."""

import itertools
import json
import random
from decimal import Decimal, localcontext
from pathlib import Path

TAU = [10, 10, 30, 120, 10]


def reference(case):
    lists = []
    truncated = []
    for index, raw in enumerate(case['components']):
        positive = sorted((e for e in raw if e['score'] > 0), key=lambda e: (-e['score'], e['id']))
        admitted = [e for e in positive if e['stay']]
        for entry in positive:
            if entry not in admitted and len(admitted) < case['caps'][index]-1:
                admitted.append(entry)
        truncated.append(len(positive)-len(admitted))
        keep = (-Decimal(str(case['dt']))/TAU[index]).exp()
        total = sum((Decimal(str(e['score'])) for e in admitted), Decimal(0))
        choices = [dict(id=e['id'], weight=keep*Decimal(str(e['score']))/total,
                        stay=e['stay'], boundary=e['boundary']) for e in admitted]
        choices.append(dict(id=None, weight=1-keep if admitted else Decimal(1),
                            stay=case['unknown_parent'][index], boundary=None))
        choices.sort(key=lambda e: (-e['weight'], e['id'] is None, e['id'] or 0))
        lists.append(choices)
    tuples = []
    for entries in itertools.product(*lists):
        weight = Decimal(1)
        for entry in entries:
            weight *= entry['weight']
        ids = [e['id'] for e in entries]
        legal = ([ids[2], ids[3]] not in case['illegal_pairs']
                 and [ids[3], ids[4]] not in case.get('illegal_correspondences', []))
        tuples.append(dict(ids=ids, weight=weight, legal=legal, stay=all(e['stay'] for e in entries),
                           boundary=(entries[2]['boundary'], entries[3]['boundary'])))
    tuples.sort(key=lambda t: (-t['weight'], [(i is None, i or 0) for i in t['ids']]))
    selected = [next(t for t in tuples if all(i is None for i in t['ids']))]
    stay = next((t for t in tuples if t['stay'] and t['legal']), None)
    if stay is not None and stay not in selected:
        selected.append(stay)
    boundary = 0
    for i, category in enumerate(itertools.product(['stay', 'exit'], repeat=2)):
        best = next((t for t in tuples if t['boundary'] == category and t['legal']), None)
        if best is not None:
            boundary |= 1 << i
            if best not in selected:
                selected.append(best)
    pops = 0
    for candidate in tuples[:16]:
        if len(selected) == 16:
            break
        pops += 1
        if candidate['legal'] and candidate not in selected:
            selected.append(candidate)
    denominator = sum(t['weight'] for t in selected)
    return dict(rows=[dict(ids=t['ids'], probability=float(t['weight']/denominator)) for t in selected],
                admitted_product_mass=float(denominator), heap_pops=pops,
                reserved_stay=stay is not None, boundary_mask=boundary, truncated=truncated,
                cartesian_count=len(tuples),
                lists=[[dict(id=e['id'], probability=float(e['weight'])) for e in choices] for choices in lists])


def cases():
    rng = random.Random(20260919)
    result = []
    for index in range(12):
        components = []
        maxima = [4, 18, 5, 4, 18] if index == 11 else [2, 3, 3, 3, 3]
        for component, count in enumerate(maxima):
            entries = []
            for i in range(count):
                stay = i == count-1
                entries.append(dict(id=i, score=10**rng.uniform(-2, 2), stay=stay,
                                    boundary=('stay' if stay else 'exit') if component in [2, 3] else None))
            rng.shuffle(entries)
            components.append(entries)
        case = dict(name=f'tuple-{index}', dt=[.001, .01, .1, 2.][index % 4],
                    caps=[8, 19, 6, 5, 19] if index % 3 else [3, 3, 3, 3, 3],
                    components=components, unknown_parent=[False]*5,
                    illegal_pairs=[[0, 0], [0, 1]] if index % 2 else [])
        case['expected'] = reference(case)
        result.append(case)
    for index in range(12):
        base = result[index]
        case = dict(base, name=f'correspondence-{index}')
        section_ids = [e['id'] for e in base['components'][3]]
        correspondence_ids = [e['id'] for e in base['components'][4]] + [None]
        # Some rows require a weak nonleading witness; others have none or reject stay.
        case['illegal_correspondences'] = [
            [sid, cid] for sid in section_ids for cid in correspondence_ids
            if (sid == 0 and cid != correspondence_ids[-2])
            or (sid == 1 and index % 3 == 0)
            or (sid > 1 and cid in correspondence_ids[:2])
        ]
        case['expected'] = reference(case)
        result.append(case)
    return result


if __name__ == '__main__':
    with localcontext() as ctx:
        ctx.prec = 60
        result = dict(schema='joint-tuples-cartesian-v2', precision=60, cases=cases())
    path = Path('tests/fixtures/temporal_cognition/joint_proposals.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(result["cases"])} Cartesian cases written to {path}')
