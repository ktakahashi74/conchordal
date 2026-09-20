"""Enumerate complete small joint distributions in Decimal probability space."""

import itertools
import json
import random
from decimal import Decimal, localcontext
from pathlib import Path


def pair(parent, extension, prior, transition, potential, resolved=True):
    return dict(parent=parent, extension=extension, prior=prior,
                transition=transition, potential=potential, resolved=resolved)


def mass(row, observed):
    return Decimal(str(row['prior']))*Decimal(str(row['transition']))*(
        Decimal(str(row['potential'])).exp() if observed else Decimal(1))


def beam(rows, weights, cap):
    indices = sorted((i for i, r in enumerate(rows) if r['resolved'] and weights[i] > 0),
                     key=lambda i: (-weights[i], rows[i]['parent'], rows[i]['extension']))
    return dict(rows=[dict(parent=rows[i]['parent'], extension=rows[i]['extension'],
                           mass=float(weights[i])) for i in indices[:cap]],
                explicit_unknown=float(sum((w for r, w in zip(rows, weights) if not r['resolved']), Decimal(0))),
                pruned_mass=float(sum((weights[i] for i in indices[cap:]), Decimal(0))),
                pruned_count=len(indices[cap:]))


def reference(shared, observed):
    shared_mass = []
    marginals = []
    for context in shared:
        groups = context['groups']
        local = [[Decimal(0) for _ in group] for group in groups]
        total = Decimal(0)
        # No log-sum-exp, per-parent normalization, or pruning in this enumeration.
        for indices in itertools.product(*(range(len(group)) for group in groups)):
            weight = mass(context['pair'], observed)
            for group, index in zip(groups, indices):
                weight *= mass(group[index], observed)
            total += weight
            for values, index in zip(local, indices):
                values[index] += weight
        shared_mass.append(total)
        marginals.append([[w/total for w in group] for group in local] if total else None)
    evidence = sum(shared_mass)
    result = beam([c['pair'] for c in shared], [w/evidence for w in shared_mass], 7)
    result['log_evidence'] = float(evidence.ln())
    for out in result['rows']:
        index = next(i for i, c in enumerate(shared) if
                     (c['pair']['parent'], c['pair']['extension']) == (out['parent'], out['extension']))
        out['groups'] = [beam(g, m, 15) for g, m in zip(shared[index]['groups'], marginals[index])]
    return result


def cases():
    rng = random.Random(20260918)
    results = []
    for index in range(24):
        shared = []
        for parent, prior in enumerate([0.3, 0.7]):
            for child, transition in enumerate([0.2, 0.5, 0.3]):
                groups = []
                for _ in range(1+index % 3):
                    group = []
                    for lp, q in enumerate([0.25, 0.75]):
                        for le, t in enumerate([0.1, 0.6, 0.3]):
                            group.append(pair(lp, le, q, t, rng.randint(-20, 20)/4., le != 2))
                    groups.append(group)
                shared.append(dict(pair=pair(parent, child, prior, transition,
                                              rng.randint(-20, 20)/4., child != 2), groups=groups))
        results.append(dict(name=f'enumeration-{index}', observed=index % 4 != 0, shared=shared))
    diffuse = [pair(i, j, 1/16., t, 0., j == 0) for i in range(16)
               for j, t in enumerate([0.98, 0.02])]
    focused = [pair(0, 0, 1., .98, 0.), pair(0, 1, 1., .02, 0., False)]
    results.append(dict(name='diffuse-local-partition', observed=True, shared=[
        dict(pair=pair(0, 0, 1., .45, 0.), groups=[diffuse]),
        dict(pair=pair(0, 1, 1., .45, 0.), groups=[focused]),
        dict(pair=pair(0, 2, 1., .1, 0., False), groups=[focused])]))
    results.append(dict(name='empty-initial-unknown', observed=False, shared=[
        dict(pair=pair(0, 0, 1., 1., 0., False), groups=[])]))
    for observed in [False, True]:
        results.append(dict(name=f'empty-local-product-observed-{observed}', observed=observed,
                            shared=[dict(pair=pair(p, c, q, t, (p-c)/3, c != 2), groups=[])
                                    for p, q in enumerate([.3, .7])
                                    for c, t in enumerate([.2, .5, .3])]))
    results.append(dict(name='empty-local-product-shared-pruning', observed=True,
                        shared=[dict(pair=pair(p, c, 1/8, 1/4, 0., c != 3), groups=[])
                                for p in range(8) for c in range(4)]))
    for case in results:
        case['expected'] = reference(case['shared'], case['observed'])
    return results


if __name__ == '__main__':
    with localcontext() as context:
        context.prec = 60
        result = dict(schema='joint-enumerated-decimal-v1', precision=60, cases=cases(),
                      claim='small fully enumerated numerical kernels, not live proposal assembly')
    destination = Path('tests/fixtures/temporal_cognition/joint.json')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(result["cases"])} exhaustive cases written to {destination}')
