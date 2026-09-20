#!/usr/bin/env python3
"""Freeze ongoing cue choices from the registered Python selector."""
import json
from pathlib import Path
from temporal_section_reference import select_section_cue

cases = []
for gaps in (False, True):
    steps = []
    intervals = []
    heard = 0
    for step in range(1, 41):
        end = step * 100
        missing = gaps and 21 <= step <= 24
        weight = 0.0 if 8 <= step <= 14 else (0.125 if step % 3 == 0 else 1.0)
        if not missing:
            intervals.append([2, (end - 100) / 1000, end / 1000, weight])
            heard = end
        spans = [dict(epoch=0, occurrence_id=i, record_kind='ongoing', start=start / 1000,
                      descriptor_support_end=heard / 1000, support_intervals=intervals,
                      query_descriptor=f'whole-prefix-{i}')
                 for i, start in [(50, 0), (20, 300), (10, 300)] if start <= heard]
        expected = select_section_cue(spans, 0, {2}, end / 1000, 0)
        steps.append(dict(end=end, missing=missing, weight=weight,
                          foregrounds=[[s['occurrence_id'], round(s['start'] * 1000), heard] for s in spans],
                          expected=expected))
    cases.append(dict(gaps=gaps, steps=steps))
path = Path(__file__).resolve().parents[1] / 'tests/fixtures/temporal_cognition/section_cues.json'
path.write_text(json.dumps({'schema': 'ongoing-section-cue-v1', 'cases': cases}, indent=2) + '\n')
print(path)
