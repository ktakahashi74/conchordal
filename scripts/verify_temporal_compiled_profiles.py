"""Read back every compiled binding against its separately verified feature target."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def verify(model, features):
    data = model.read_bytes()
    if data[:8] != b'I10AP001':
        raise ValueError('wrong model format')
    header_size, count = struct.unpack_from('<II', data, 8)
    header = json.loads(data[16:16+header_size])
    if header.get('routing_projection') not in (None, 'routed_body'):
        raise ValueError('unknown routing projection')
    start = 16+header_size
    if len(data)-start != count*375*96:
        raise ValueError('incomplete binary payload')
    source = features/'verification.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != header['source_verification_sha256']:
        raise ValueError('changed source verification')
    hashes = json.loads(source.read_text())['files_sha256']
    bindings = frames = unknown = zeros = 0
    used = set()
    trajectory_hashes = set()
    for i in range(count):
        trajectory_hashes.add(hashlib.sha256(data[start+i*36000:start+(i+1)*36000]).digest())
    if len(trajectory_hashes) != count:
        raise ValueError('duplicate stored trajectories')
    for profile in header['profiles']:
        if header.get('routing_projection') == 'routed_body':
            original = profile['source_descriptor']['bus']
            routing = profile['recipe']['routing']
            routed = [routing in ('both', 'habitat'), routing in ('both', 'presentation')]
            if not any(routed):
                raise ValueError('missing routed body source')
            expected = original if routed[original] else routed.index(True)
            if profile['descriptor_source_bus'] != original or profile['source_bus'] != expected:
                raise ValueError('routing projection changed descriptor identity or source rule')
        for class_name, slots in zip(header['classes'], profile['trajectories'], strict=True):
            for offset, index in zip(header['action_offsets'], slots, strict=True):
                if index is None:
                    continue
                path = features/profile['record_id']/f'{class_name}-{offset}-bus{profile["source_bus"]}.jsonl'
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != hashes[str(path.relative_to(features))]:
                    raise ValueError(f'changed target {path}')
                rows = [json.loads(line) for line in raw.splitlines()]
                if len(rows) != 375 or not 0 <= index < count:
                    raise ValueError('wrong trajectory length/index')
                used.add(index)
                bindings += 1
                for row_number, row in enumerate(rows):
                    mask, reserved, *values = struct.unpack_from('<II11d', data, start+(index*375+row_number)*96)
                    if reserved or mask >> 11:
                        raise ValueError('invalid frame control word')
                    source_values = row['raw']['values']+[row['energy']]
                    for coordinate, (value, expected) in enumerate(zip(values, source_values, strict=True)):
                        known = bool(mask & (1 << coordinate))
                        if known != (expected is not None):
                            raise ValueError('support mask differs')
                        if struct.pack('<d', value) != struct.pack('<d', 0. if expected is None else expected):
                            raise ValueError('descriptor bits differ')
                        unknown += not known
                        zeros += known and value == 0.
                    frames += 1
    if used != set(range(count)):
        raise ValueError('unreferenced trajectory')
    return dict(schema='i10-compiled-profile-verification-v1', model_sha256=hashlib.sha256(data).hexdigest(),
                source_verification_sha256=header['source_verification_sha256'],
                profiles=len(header['profiles']), bindings=bindings, unique_trajectories=count,
                compared_frames=frames, compared_coordinates=frames*11,
                unknown_coordinates=unknown, known_zero_coordinates=zeros, bit_exact=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('features', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.model, args.features)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
