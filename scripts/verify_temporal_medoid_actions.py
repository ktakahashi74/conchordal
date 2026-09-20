"""Verify actual-action targets against their frozen development medoids.

PCM, routing, physical energies, class equivalences and unchanged pre-action
samples are checked independently of the Rust acquisition. This is not a
spectral transfer or head-calibration test.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 'b3eb8cba69007e16aedb013d9ee8ba617b3ecb4d3420afb492768bfeba02fd28'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, model_root):
    model = json.loads((model_root / 'model-8.json').read_text())
    unsigned = {k: v for k, v in model.items() if k != 'model_version'}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() != VERSION:
        raise ValueError('frozen model contents changed')
    manifest = json.loads((root / 'manifest.json').read_text())
    ids = [m['record_id'] for m in model['medoids']]
    if model['model_version'] != VERSION or manifest['model_version'] != VERSION or manifest['schema'] != 'i10-medoid-action-targets-v1' or manifest['cases'] != ids:
        raise ValueError('wrong registered model or case matrix')
    offsets = [(k * 192000 + 15) // 31 for k in range(32)]
    expected = [(c, t) for c in ['onset_now', 'delayed_onset', 'wait', 'skip', 'continue', 'release', 'gap']
                for t in offsets if (t == 0 if c in ['onset_now', 'skip', 'continue'] else t > 0 if c in ['delayed_onset', 'wait'] else True)]
    counts = dict(cases=0, branches=0, energy_coordinates=0, prefix_samples=0, equivalent_pairs=0, delayed_prefix_pairs=0, release_tails=0, quiet_release_tails=0)
    largest = 0.
    hashes = {'manifest.json': digest(root / 'manifest.json')}
    for name in ids:
        directory = root / name
        profiles = json.loads((directory / 'profiles.json').read_text())
        source = next(r for r in model['records'] if r['record_id'] == name)
        issue = source['descriptor']['end']
        if (profiles['source_record'] != source or profiles['model_version'] != VERSION or profiles['issue_sample'] != issue
                or profiles['record_id'] != name or profiles['offset_samples'] != offsets
                or [profiles[k] for k in ['sample_rate', 'hop_samples', 'horizon_samples', 'assay_period_samples']] != [48000, 512, 192000, 9600]):
            raise ValueError('prototype source, clock or recipe changed')
        route = source['recipe']['routing']
        routing = [route != 'presentation', route != 'habitat']
        if profiles['routing'] != routing or [(b['action']['class'], b['offset']) for b in profiles['branches']] != expected:
            raise ValueError('wrong routing or action matrix')
        for key in ['scenario', 'report', 'wav']:
            if digest(model_root / source[key]) != source['hashes'][key]:
                raise ValueError('original medoid input changed')
        if digest(model_root / 'extraction.toml') != source['hashes']['config']:
            raise ValueError('original extraction configuration changed')
        report = [json.loads(line) for line in (directory / 'ordinary.jsonl').read_text().splitlines()]
        descriptor = next(r for r in report if r['type'] == 'body_descriptor' and r['bus'] == source['descriptor']['bus'] and r['end'] == issue)
        if any(descriptor[k] != v for k, v in source['descriptor'].items()):
            raise ValueError('ordinary descriptor no longer matches the medoid')
        hashes[f'{name}/profiles.json'] = digest(directory / 'profiles.json')
        hashes[f'{name}/ordinary.jsonl'] = digest(directory / 'ordinary.jsonl')
        for bus in range(2):
            path = directory / f'prefix-bus{bus}.f32le'
            prefix = np.fromfile(path, dtype='<f4')
            if len(prefix) != issue or not np.isfinite(prefix).all() or (not routing[bus] and prefix.any()):
                raise ValueError('invalid private prefix')
            counts['prefix_samples'] += len(prefix)
            hashes[str(path.relative_to(root))] = digest(path)
        waves = {}
        for branch in profiles['branches']:
            action = branch['action']; cls, offset = action['class'], branch['offset']; at = issue + offset
            target = dict(class_=cls, issued_at=issue, at=at, excitation_at=at if cls in ['onset_now', 'delayed_onset'] else None,
                          release_at=at if cls in ['release', 'gap'] else None, reconsider_at=at if cls == 'wait' else None,
                          consumes_due_opportunity=cls == 'skip', withhold_until=at+9600 if cls == 'gap' else None)
            target['class'] = target.pop('class_')
            if action != target or branch['pcm'] != [f'{cls}-{offset}-bus{bus}.f32le' for bus in range(2)]:
                raise ValueError('wrong class timing or opportunity bookkeeping')
            for bus, filename in enumerate(branch['pcm']):
                path = directory / filename
                audio = np.fromfile(path, dtype='<f4')
                if len(audio) != 192000 or not np.isfinite(audio).all() or (not routing[bus] and audio.any()):
                    raise ValueError('invalid private future PCM')
                energy = np.mean(audio.astype(float).reshape(-1, 512)**2, axis=1)
                reported = np.asarray(branch['energy_per_hop'][bus])
                if reported.shape != energy.shape or not np.isfinite(reported).all() or not np.allclose(reported, energy, rtol=2e-14, atol=1e-20):
                    raise ValueError('physical-hop energy mismatch')
                counts['energy_coordinates'] += len(energy)
                largest = max(largest, float(np.max(np.abs(reported-energy))))
                waves[(cls, offset, bus)] = audio
                hashes[str(path.relative_to(root))] = digest(path)
            counts['branches'] += 1
        for bus in range(2):
            natural = waves[('continue', 0, bus)]
            for cls, offset in [('skip', 0)] + [('wait', t) for t in offsets[1:]]:
                if not np.array_equal(waves[(cls, offset, bus)], natural):
                    raise ValueError('natural continuation differs across bookkeeping-only classes')
                counts['equivalent_pairs'] += 1
            for offset in offsets:
                if not np.array_equal(waves[('release', offset, bus)], waves[('gap', offset, bus)]):
                    raise ValueError('gap incorrectly removes release tail')
                counts['equivalent_pairs'] += 1
            for cls in ['delayed_onset', 'release', 'gap']:
                for offset in offsets[1:]:
                    audio = waves[(cls, offset, bus)]
                    if not np.array_equal(audio[:offset], natural[:offset]):
                        raise ValueError('future action changed its pre-action PCM')
                    counts['delayed_prefix_pairs'] += 1
            if routing[bus]:
                if np.array_equal(waves[('onset_now', 0, bus)], natural):
                    raise ValueError('new excitation has no acoustic effect')
                for offset in offsets[:-1]:
                    tail = waves[('release', offset, bus)][offset:offset+9600]
                    # A naturally decayed modal body need not exceed an audibility threshold.
                    if np.any(np.abs(natural[offset:offset+512]) > 1e-6) and not tail.any():
                        raise ValueError('active release tail replaced by zero')
                    counts['release_tails' if np.any(np.abs(tail) > 1e-6) else 'quiet_release_tails'] += 1
        counts['cases'] += 1
        print(f'verified {name}', flush=True)
    return dict(schema='i10-medoid-action-verification-v1', counts=counts, model_version=VERSION,
                max_energy_absolute_error=largest, files_sha256=hashes,
                claim='offline conditional targets only; no live legality, spectral projection accuracy or calibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--model-root', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.model_root)
    (args.root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, indent=2))
