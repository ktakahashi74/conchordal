"""Verify issued release plans and fixed energy proxies against saved real recipes."""

import argparse
import json
from pathlib import Path

import numpy as np

from verify_temporal_policy_defaults import require, sha, verify as verify_policy


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def gain(envelope, tick, off=None):
    start = envelope['onset']
    hold = envelope['hold_end']
    end = envelope['release_end']
    if off is not None and off < hold:
        hold, end = off, min(2**64 - 1, off + envelope['release_ticks'])
    if tick < start or tick >= end:
        return 0.0
    attack = min(envelope['attack_ticks'], max(1, hold - start))
    pos = tick - start
    f = np.float32
    if pos < attack:
        level = min(f(1), f(pos + 1) / f(attack))
    elif envelope['decay_ticks'] and pos < attack + envelope['decay_ticks']:
        sustain = f(envelope['sustain_level'])
        level = sustain + (f(1) - sustain) * np.exp(-f(envelope['decay_lambda']) * f(pos - attack))
    else:
        level = f(envelope['sustain_level'])
    release = f(1) if tick < hold else f(end - tick) / f(envelope['release_ticks'])
    return float(f(level * release))


def verify(root, inputs, previous=None):
    verify_policy(root, inputs)
    results = []
    registration = json.loads((inputs / 'registration.json').read_text())
    for case in registration['cases']:
        directory = root / case['id']
        trace = rows(directory / 'policy-inputs.jsonl')
        report = rows(directory / 'ordinary.jsonl')
        recipes = {(r['now'], b['source_id'], b['source_generation'], t['tone_id']): t
                   for r in trace for b in r['batches'] for t in b['tones']}
        plans, coordinates, changed, error = 0, 0, 0, 0.0
        outside_saved_trace = 0
        old_loss, new_loss = 0.0, 0.0
        for out in report:
            if out['type'] != 'self_sound_outcome' or not out.get('prediction'):
                continue
            prediction = out['prediction']
            plan = prediction['scheduled_release']
            require(all(e['model'] == 'source_energy_log1p_residual_v2'
                        for e in prediction['source_energy']), 'model version')
            if out['action'] != 'onset':
                require(plan is None, 'release command must not reuse onset receipt')
                continue
            key = tuple(out[k] for k in ['issued_at_sample', 'source_id', 'source_generation', 'tone_id'])
            if key[0] >= registration['issue_sample'] + registration['horizon_samples']:
                outside_saved_trace += 1
                continue
            tone = recipes[key]
            receipt = tone['opportunity']
            off = receipt['planned_release_at'] if receipt else None
            expected = None if off is None else dict(
                apply_at_sample=key[0] + (off - key[0]) // 512 * 512, off_sample=off)
            require(plan == expected, 'issued release receipt or delivery hop')
            plans += plan is not None
            envelope = prediction['envelope']
            require(envelope['onset'] == tone['onset'], 'envelope onset')
            hold = tone['hold_ticks'] if tone['hold_ticks'] is not None else 60 * 48000
            end = min(2**64 - 1, tone['onset'] + hold)
            require(envelope['hold_end'] == end, 'base envelope anticipates release')
            adsr = tone['adsr']
            for field, seconds in [('attack_ticks', 'attack_sec'), ('release_ticks', 'release_sec')]:
                ticks = max(1, int(float(np.float32(adsr[seconds])) * 48000 + 0.5))
                require(envelope[field] == ticks, 'recipe envelope duration')
            require(envelope['release_end'] == min(2**64 - 1, end + envelope['release_ticks']),
                    'base release end')
            amplitude = float(np.float32(np.float32(tone['amp']) * np.float32(tone['body']['amp_scale'])))
            for bus, energy in enumerate(prediction['source_energy']):
                routed = out['buses'][bus]['routed']
                for k, (left, right) in enumerate(energy['windows']):
                    tick = (left + right) // 2
                    active_off = off if plan is not None and tick >= plan['apply_at_sample'] else None
                    fixed = amplitude**2 * gain(envelope, tick, active_off)**2 / 2 if routed else 0.0
                    old = amplitude**2 * gain(envelope, tick)**2 / 2 if routed else 0.0
                    actual = energy['predictions'][0][k]
                    error = max(error, abs(fixed - actual))
                    require(abs(fixed - actual) <= max(1e-14, abs(fixed) * 5e-7), 'fixed energy proxy')
                    changed += abs(fixed - old) > 1e-14
                    coordinates += 1
                    if energy['target'][k] is not None:
                        old_loss += (old - energy['target'][k])**2
                        new_loss += (fixed - energy['target'][k])**2
        unchanged = 0
        common_frames = 0
        unmatched_frames = None
        if previous is not None:
            prior = previous / case['id']
            require(trace == rows(prior / 'policy-inputs.jsonl'), 'producer inputs changed')
            for path in prior.rglob('*.f32le'):
                require(sha(path) == sha(directory / path.relative_to(prior)), 'actual PCM changed')
                unchanged += 1
            prior_report = rows(prior / 'ordinary.jsonl')
            observed = [{(r['observation']['bus'], r['observation']['source_epoch'],
                          r['observation']['frame_id']): r['observation']
                         for r in records if r['type'] == 'temporal_observation'
                         and r['observation']['frame_id'] is not None}
                        for records in [report, prior_report]]
            common = observed[0].keys() & observed[1].keys()
            require(bool(common), 'no common observed frames')
            for identity in common:
                require({k: v for k, v in observed[0][identity].items() if k != 'delivery_delay_us'} ==
                        {k: v for k, v in observed[1][identity].items() if k != 'delivery_delay_us'},
                        'passive observation changed')
            common_frames = len(common)
            unmatched_frames = [len(o) - common_frames for o in observed]
            before = {(r['issued_at_sample'], r['tone_id'], r['action']): r for r in prior_report
                      if r['type'] == 'self_sound_outcome'}
            for out in report:
                if out['type'] != 'self_sound_outcome':
                    continue
                old = before[(out['issued_at_sample'], out['tone_id'], out['action'])]
                require({k: v for k, v in out.items() if k != 'prediction'} ==
                        {k: v for k, v in old.items() if k != 'prediction'}, 'action observation changed')
                if out.get('prediction'):
                    require([e['target'] for e in out['prediction']['source_energy']] ==
                            [e['target'] for e in old['prediction']['source_energy']], 'energy teacher changed')
        results.append(dict(case=case['id'], scheduled_onsets=plans, fixed_coordinates=coordinates,
                            onsets_outside_saved_trace=outside_saved_trace,
                            changed_fixed_coordinates=changed, max_absolute_error=error,
                            old_fixed_squared_error_sum=old_loss, new_fixed_squared_error_sum=new_loss,
                            unchanged_common_passive_frames=common_frames,
                            unmatched_passive_frames=unmatched_frames,
                            unchanged_pcm_files=unchanged))
    return dict(schema='i10-scheduled-release-verification-v1', cases=results,
                limits='Fixed proxy arithmetic and issue-known input consumption; no whole-body accuracy, calibration or shared candidate transfer acceptance.',
                hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.glob('*/*.jsonl'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.inputs, args.previous)
    (args.root / 'scheduled-release-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'hashes'}, indent=2))
