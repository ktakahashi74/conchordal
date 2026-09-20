"""Audit frozen medoid transfer against actual-body counterfactual PCM features.

No fitting, approval threshold, ordinal prediction, or live action is introduced.
Inputs and the exploratory comparison must be registered before evaluation.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_cell(offsets, at):
    if at < offsets[0] or at > offsets[-1]:
        return None
    return min(range(len(offsets)), key=lambda i: (abs(offsets[i] - at), i))


def evaluate(registration, model_file, profile_file, features, output):
    registered = json.loads(registration.read_text())
    for path, expected in registered['inputs'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'changed registered input: {path}')
    for path in [model_file, profile_file, features / 'manifest.json', features / 'verification.json']:
        if not any(path.resolve() == Path(p).resolve() for p in registered['inputs']):
            raise ValueError(f'unregistered input: {path}')
    model = json.loads(model_file.read_text())
    verified = json.loads((features / 'verification.json').read_text())
    manifest = json.loads((features / 'manifest.json').read_text())
    source = manifest['source_manifest']
    data = profile_file.read_bytes()
    if data[:8] != b'I10AP001':
        raise ValueError('profile magic')
    header_size, count = struct.unpack_from('<II', data, 8)
    header = json.loads(data[16:16 + header_size])
    routing_projection = header.get('routing_projection')
    if routing_projection != registered.get('projection_routing'):
        raise ValueError('unregistered routing projection')
    if routing_projection not in (None, 'routed_body'):
        raise ValueError('unknown routing projection')
    dtype = np.dtype([('mask', '<u4'), ('reserved', '<u4'), ('values', '<f8', (11,))])
    frames = np.frombuffer(data, dtype=dtype, offset=16 + header_size).reshape(count, 375)
    if (header['body_model_version'] != model['model_version']
            or header['sample_rate'] != 48000 or header['hop_samples'] != 512
            or source['horizon_samples'] != 192000 or source['issue_sample'] != 14848
            or frames['reserved'].any() or not np.isfinite(frames['values']).all()):
        raise ValueError('model identity or physical frame contract')
    if [p['record_id'] for p in header['profiles']] != [m['record_id'] for m in model['medoids']]:
        raise ValueError('medoid order')
    offsets = header['action_offsets']
    issue = source['issue_sample']
    classes = header['classes']
    deviations = np.maximum(model['deviations'], 1e-6)
    used_hashes = {}
    coordinates = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    metrics = {str(i): dict(paired=0, actual_only=0, projected_only=0,
                           sum_absolute=0., sum_squared=0., maximum_absolute=0.) for i in coordinates}
    windows = [12000, 48000, 96000, 192000]
    widths = np.array([np.clip(end - np.arange(375) * 512, 0, 512) for end in windows]) / 48000
    rows, owners, ranks = [], [], []

    def read_rows(path):
        relative = str(path.relative_to(features))
        actual = digest(path)
        if verified['files_sha256'].get(relative) != actual:
            raise ValueError(f'changed verified feature: {relative}')
        used_hashes[relative] = actual
        return [json.loads(line) for line in path.read_text().splitlines()]

    for case in source['cases']:
        for bus in range(2):
            prefix = read_rows(features / case / f'prefix-bus{bus}.jsonl')
            descriptors = [r['body_descriptor'] for r in prefix if r['body_descriptor']
                           and r['body_descriptor']['available'] <= issue
                           and r['body_descriptor']['end'] <= issue]
            descriptor = descriptors[-1] if descriptors else None
            candidates = []
            if descriptor and descriptor['active']:
                for index, medoid in enumerate(model['medoids']):
                    common = [i for i in range(6) if descriptor['mask'] & medoid['mask'] & (1 << i)]
                    if not common:
                        continue
                    # Independent vector formulation of the registered standardized distance.
                    delta = (np.array(descriptor['raw_values']) - medoid['raw_values']) / deviations
                    distance = float(np.linalg.norm(delta[common]) / math.sqrt(len(common)))
                    if distance <= .25:
                        candidates.append((distance, -len(common), index))
            match = min(candidates) if candidates else None
            prototype = header['profiles'][match[2]] if match else None
            source_routed = prototype is not None and prototype['recipe']['routing'] != (
                'presentation' if prototype['source_bus'] == 0 else 'habitat')
            actual_routed = case.split('-')[1] != ('presentation' if bus == 0 else 'habitat')
            owner = dict(case=case, bus=bus, descriptor=descriptor,
                         assignment=None if match is None else dict(index=match[2], distance=match[0],
                         common_coordinates=-match[1], record_id=prototype['record_id']),
                         actual_routed=actual_routed, prototype_source_routed=source_routed)
            owners.append(owner)
            branches = []
            for path in sorted((features / case).glob(f'*-bus{bus}.jsonl')):
                if path.name.startswith('prefix-'):
                    continue
                cls, offset = path.name.removesuffix(f'-bus{bus}.jsonl').rsplit('-', 1)
                offset = int(offset)
                actual_rows = read_rows(path)
                if len(actual_rows) != 375:
                    raise ValueError('incomplete future')
                actual = np.array([r['raw']['values'] + [r['energy']] for r in actual_rows], dtype=float)
                cell = nearest_cell(offsets, offset)
                trajectory = None if prototype is None or cell is None else prototype['trajectories'][classes.index(cls)][cell]
                projected = np.full_like(actual, np.nan)
                if trajectory is not None:
                    values = frames[trajectory]['values']
                    masks = frames[trajectory]['mask']
                    projected = np.where((masks[:, None] & (1 << np.arange(11))) != 0, values, np.nan)
                    projected[0, 5] = np.nan
                    before = prefix[-1]['raw']['values'][2]
                    change = projected[0, 2] - before if before is not None else np.nan
                    projected[0, 3:5] = [max(change, 0.), max(-change, 0.)]
                    if routing_projection == 'routed_body' and not actual_routed:
                        # Exact routing algebra; no future spectral support is inferred.
                        projected[:] = np.nan
                        projected[:, 2] = math.log2(1e-6)
                        projected[:, 10] = 0.
                for i in coordinates:
                    truth, prediction = np.isfinite(actual[:, i]), np.isfinite(projected[:, i])
                    paired = truth & prediction
                    error = projected[paired, i] - actual[paired, i]
                    metric = metrics[str(i)]
                    metric['paired'] += int(paired.sum())
                    metric['actual_only'] += int((truth & ~prediction).sum())
                    metric['projected_only'] += int((prediction & ~truth).sum())
                    metric['sum_absolute'] += float(np.abs(error).sum())
                    metric['sum_squared'] += float(np.square(error).sum())
                    metric['maximum_absolute'] = max(metric['maximum_absolute'], float(np.max(np.abs(error), initial=0.)))
                measured = widths @ actual[:, 10]
                predicted = widths @ projected[:, 10] if np.isfinite(projected[:, 10]).all() else None
                row = dict(case=case, bus=bus, class_=cls, offset=offset, cell=cell,
                           selected_offset=None if cell is None else offsets[cell], trajectory=trajectory,
                           actual_energy=measured.tolist(), projected_energy=None if predicted is None else predicted.tolist())
                row['class'] = row.pop('class_')
                rows.append(row)
                branches.append(row)
            control_class = 'continue' if '-two_tones-' in case else 'skip'
            control = next(r for r in branches if r['class'] == control_class and r['offset'] == 0)
            for row in branches:
                row['control_class'] = control_class
                row['actual_control_difference'] = (np.array(row['actual_energy']) - control['actual_energy']).tolist()
                row['projected_control_difference'] = None
                if row['projected_energy'] is not None and control['projected_energy'] is not None:
                    row['projected_control_difference'] = (np.array(row['projected_energy']) - control['projected_energy']).tolist()
            for wi, end in enumerate(windows):
                counts = Counter()
                for left, right in itertools.combinations(branches, 2):
                    if left['projected_control_difference'] is None or right['projected_control_difference'] is None:
                        counts['unsupported_pairs'] += 1
                        continue
                    truth = left['actual_control_difference'][wi] - right['actual_control_difference'][wi]
                    prediction = left['projected_control_difference'][wi] - right['projected_control_difference'][wi]
                    ts, ps = (0 if abs(v) <= 1e-12 else (1 if v > 0 else -1) for v in [truth, prediction])
                    counts['pairs'] += 1
                    counts['reversed' if ts * ps == -1 else 'lost_difference' if ts and not ps
                           else 'invented_difference' if ps and not ts else 'agree'] += 1
                ranks.append(dict(case=case, bus=bus, end_offset=end, **counts))
    for metric in metrics.values():
        metric['mae'] = metric['sum_absolute'] / metric['paired'] if metric['paired'] else None
        metric['rmse'] = math.sqrt(metric['sum_squared'] / metric['paired']) if metric['paired'] else None
    summary = dict(owners=len(owners), matched_owners=sum(o['assignment'] is not None for o in owners),
                   matched_masks=dict(Counter(str(o['descriptor']['mask']) for o in owners if o['assignment'])),
                   routed_owners_matched_to_unrouted=sum(o['actual_routed'] and not o['prototype_source_routed']
                                                       for o in owners if o['assignment']),
                   branch_bus_comparisons=len(rows), unsupported_branches=sum(r['trajectory'] is None for r in rows),
                   rank_totals={str(end): dict(sum((Counter({k:v for k,v in r.items() if k not in ['case','bus','end_offset']})
                                               for r in ranks if r['end_offset']==end),Counter())) for end in windows})
    result = dict(schema='i10-actual-profile-transfer-audit-v1', status='diagnostic_not_accepted',
                  projection_routing=routing_projection,
                  registration_sha256=digest(registration), summary=summary, coordinate_errors=metrics,
                  owners=owners, branches=rows, ranks=ranks, verified_feature_hashes=used_hashes)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registration', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path, default=Path('target/i10-body-prototypes-20260917/model-8.json'))
    parser.add_argument('--profiles', type=Path, default=Path('target/i10-action-profiles-20260918.bin'))
    parser.add_argument('--features', type=Path, default=Path('target/i10-action-features-512-20260918'))
    args = parser.parse_args()
    evaluate(args.registration, args.model, args.profiles, args.features, args.output)
