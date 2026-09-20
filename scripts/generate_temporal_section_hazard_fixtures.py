#!/usr/bin/env python3
"""Register bounded gap integrals before reusing duration hazards for sections."""

import hashlib
import itertools
import json
import math
from pathlib import Path

import temporal_cognition_reference as ref


cases = []
for intercept, slope, start, duration in itertools.product(
        (-12., -2., 0., 3., 20.), (-40., -2., 0., 1., 8.),
        (0., .5, 8., 120.), (512/48000, .5, 2., 8., 32., 1200.)):
    result = ref.gap_survival(intercept, slope, start, duration)
    cases.append(dict(input=[intercept, slope, start, start+duration],
                      expected=result['integrated_hazard'], status=result['status']))
heads = []
for index in range(12):
    means = [(i%5)*.1 for i in range(82)]
    deviations = [.25+(i%3)*.25 for i in range(82)]
    raw = [None if (i+index)%7 == 0 else (i%9)*.1 for i in range(82)]
    raw[78] = math.log1p(index)
    hazard = [(i%3-1)*.01 for i in range(83)]
    hazard[0] = -2.
    hazard[79] = (index%4-1)*.1
    exits = [[(i%5-2)*.02 for i in range(83)], [(i%3-1)*.03 for i in range(83)]]
    exits.append([-a-b for a,b in zip(*exits)])
    x = [1.] + [0. if v is None else (v-m)/sd for v,m,sd in zip(raw,means,deviations)]
    x[79] = -means[78]/deviations[78]
    intercept = math.fsum(a*b for a,b in zip(x,hazard))
    slope = hazard[79]/deviations[78]
    steps = []
    for horizon in (512/48000, .5, 2., 32.):
        result = ref.gap_survival(intercept,slope,index,horizon)
        expected = None
        if result['integrated_hazard'] is not None:
            x[79] = (math.log1p(index+horizon)-means[78])/deviations[78]
            logits = [math.fsum(a*b for a,b in zip(x,row)) for row in exits]
            masses = [math.exp(v-max(logits)) for v in logits]
            exited = -math.expm1(-result['integrated_hazard'])
            expected = [math.exp(-result['integrated_hazard'])] + [exited*m/math.fsum(masses) for m in masses]
        steps.append(dict(lo=index,hi=index+horizon,expected=expected))
    heads.append(dict(means=means,deviations=deviations,raw=raw,hazard=hazard,exits=exits,steps=steps))
output = Path('tests/fixtures/temporal_cognition/section_hazards.json')
output.write_text(json.dumps(dict(schema='temporal-section-hazard-fixtures-v1',
    reference_source_sha256=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(), cases=cases,heads=heads), indent=2)+'\n')
print(json.dumps(dict(cases=len(cases), resolved=sum(c['expected'] is not None for c in cases))))
