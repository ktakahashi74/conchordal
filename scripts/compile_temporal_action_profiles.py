"""Freeze descriptor-transfer profiles from verified medoid counterfactual targets.

These are conditional research templates, not learned listener observations or
calibrated action scores. Action time and evaluation time remain separate.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

CLASSES = ['onset_now', 'delayed_onset', 'wait', 'skip', 'continue', 'release', 'gap']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_profiles(actions, features, destination, routed_body=False):
    if destination.exists():
        raise ValueError('use a fresh model path')
    verified = json.loads((features / 'verification.json').read_text())
    registration = json.loads((actions / 'verification.json').read_text())
    for root, hashes in [(actions, registration['files_sha256']), (features, verified['files_sha256'])]:
        if any(sha(root / path) != value for path, value in hashes.items()):
            raise ValueError('changed verified source')
    if verified['input_verification_sha256'] != sha(actions / 'verification.json'):
        raise ValueError('features were acquired from another action corpus')
    manifest = json.loads((actions / 'manifest.json').read_text())
    ids = manifest['cases']
    if len(ids) != 8 or ids != sorted(set(ids)):
        raise ValueError('expected the eight ordered medoids')
    offsets = [(k * 192000 + 15) // 31 for k in range(32)]
    trajectories, by_bytes, profiles = [], {}, []
    for record_id in ids:
        metadata = json.loads((actions / record_id / 'profiles.json').read_text())
        if metadata['offset_samples'] != offsets:
            raise ValueError('changed action-time grid')
        bus = metadata['source_record']['descriptor']['bus']
        descriptor_bus = bus
        if routed_body:
            routing = metadata['source_record']['recipe']['routing']
            routed = [routing in ('both', 'habitat'), routing in ('both', 'presentation')]
            if not any(routed):
                raise ValueError('body profile has no recorded routed bus')
            if not routed[bus]:
                bus = routed.index(True)
        slots = [[None] * 32 for _ in CLASSES]
        for action in metadata['branches']:
            cls, offset = action['action']['class'], action['offset']
            path = features / record_id / f'{cls}-{offset}-bus{bus}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != 375:
                raise ValueError('incomplete registered trajectory')
            data = bytearray()
            for i, row in enumerate(rows):
                start = metadata['issue_sample'] + 512 * i
                if row['role'] != 'counterfactual_actual_audio_target' or (row['raw']['start'], row['raw']['end']) != (start, start+512):
                    raise ValueError('mixed observed/projected source or wrong physical grid')
                coordinates = row['raw']['values'] + [row['energy']]
                mask = sum(1 << k for k, value in enumerate(coordinates) if value is not None)
                data.extend(struct.pack('<II11d', mask, 0, *[0. if v is None else v for v in coordinates]))
            data = bytes(data)
            index = by_bytes.get(data)
            if index is None:
                index = len(trajectories)
                by_bytes[data] = index
                trajectories.append(data)
            slots[CLASSES.index(cls)][offsets.index(offset)] = index
        profiles.append(dict(record_id=record_id, source_bus=bus, source_issue=metadata['issue_sample'],
                             source_descriptor=metadata['source_record']['descriptor'],
                             recipe=metadata['source_record']['recipe'], onset_spec=metadata['new_onset'],
                             continuous_drive=metadata['continuous_drive'], rhythms_at_issue=metadata['rhythms_at_issue'],
                             trajectories=slots))
        if routed_body:
            profiles[-1]['descriptor_source_bus'] = descriptor_bus
    header = dict(schema='i10-action-profiles-v1', body_model_version=manifest['model_version'],
                  source_verification_sha256=sha(features / 'verification.json'),
                  sample_rate=48000, nfft=2048, hop_samples=512, horizon_samples=192000,
                  classes=CLASSES, action_offsets=offsets, frames_per_trajectory=375,
                  trajectory_count=len(trajectories), profiles=profiles,
                  policy='Exact registered action offsets and physical frames; no pitch/gain/rate interpolation. First-frame rise/decline recomputed from actual issue prefix; first-frame flux unsupported. Group energy share recomputed with fixed issue background energy. This descriptor transfer is uncalibrated.',
                  scope='conditional research profile; never observed future or authorization for execution')
    if routed_body:
        header['routing_projection'] = 'routed_body'
        header['policy'] += (' Recorded routed body contribution, conditional on routing to the target bus. '
                             'Actual unrouted contributions have zero energy; this profile does not '
                             'supply their spectral/window support. Descriptor source bus is unchanged.')
    encoded = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    data = b'I10AP001' + struct.pack('<II', len(encoded), len(trajectories)) + encoded + b''.join(trajectories)
    destination.write_bytes(data)
    result = dict(schema='i10-action-profile-compilation-v1', file=str(destination), sha256=sha(destination),
                  header_bytes=len(encoded), file_bytes=len(data), trajectories=len(trajectories),
                  physical_frames=len(trajectories)*375, action_bindings=8*129,
                  source_verification_sha256=header['source_verification_sha256'],
                  compiler_sha256=sha(Path(__file__)))
    destination.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--routed-body', action='store_true',
                        help='compare route-conditioned transfer using the same body on its recorded routed bus')
    args = parser.parse_args()
    print(json.dumps(compile_profiles(args.actions, args.features, args.output, args.routed_body), indent=2))
