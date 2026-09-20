#!/usr/bin/env python3
"""Capture detached queries from the registered matcher without mutating spans."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import temporal_descriptor_reference as descriptor
import temporal_matcher_reference as matcher
from generate_temporal_descriptor_fixtures import raw


def generate():
    cases = []
    for capacity in (4, 64, 128):
        for cadence in (1, 2, 4):
            span = descriptor.SpanDescriptor(
                2, 3, 48000, 0, 512, 0., [1.] * 10,
                capacity=capacity, cadence=cadence,
            )
            steps = []
            count = capacity * cadence + 3
            for i in range(count):
                r = raw(i)
                cut = (i + 2) * 512
                span.push(r, cut / 48000)
                step = {
                    'kind': 'push', 'index': i, 'cut': cut,
                    'values_bits': [None if v is None else struct.unpack('<Q', struct.pack('<d', v))[0]
                                    for v in r['values']],
                    'intervals': r['known_sample_intervals'],
                }
                if i in (0, 1, count - 3, count - 2, count - 1):
                    before = span.snapshot()
                    query = matcher.descriptor_query(span, cut / 48000, i + 1, 7, 8)
                    assert before == span.snapshot()
                    step['query'] = {
                        'packed': bytes(query['packed_knots']).hex(),
                        'error_bits': struct.unpack('<Q', struct.pack('<d', query['reconstruction_error']))[0],
                        'audio_end': None if query['supporting_audio_end'] is None else round(query['supporting_audio_end'] * 48000),
                        'end_sample': round(query['query_end'] * 48000),
                        'available_sample': round(query['available_end'] * 48000),
                    }
                steps.append(step)
            end = (count + 100) * 512
            span.gap(end / 48000, end / 48000, end / 48000)
            before = span.snapshot()
            query = matcher.descriptor_query(span, end / 48000, count + 1, 7, 8)
            assert span.snapshot() == before
            steps.append({'kind': 'gap', 'cut': end, 'query': {
                'packed': bytes(query['packed_knots']).hex(),
                'error_bits': struct.unpack('<Q', struct.pack('<d', query['reconstruction_error']))[0],
                'audio_end': round(query['supporting_audio_end'] * 48000),
                'end_sample': end, 'available_sample': end,
            }})
            cases.append({'capacity': capacity, 'cadence': cadence, 'steps': steps})
    return {
        'schema': 'temporal-prefix-fixtures-v1',
        'sources': {str(Path(m.__file__).relative_to(Path.cwd())):
                    hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                    for m in (descriptor, matcher)},
        'cases': cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('tests/fixtures/temporal_cognition/prefixes.json'))
    args = parser.parse_args()
    result = generate()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cases': len(result['cases']),
                      'operations': sum(len(c['steps']) for c in result['cases']),
                      'queries': sum('query' in s for c in result['cases'] for s in c['steps'])}))


if __name__ == '__main__':
    main()
