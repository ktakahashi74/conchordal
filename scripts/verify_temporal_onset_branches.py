"""Independently audit seven-class local-default comparisons at real onset receipts."""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from verify_temporal_policy_defaults import require, sha

CLASSES = ['onset_now', 'delayed_onset', 'wait', 'skip', 'continue', 'release', 'gap']


def envelope(spec):
    adsr = spec['adsr']
    require(adsr is not None, 'registered recipe must have explicit ADSR')
    ticks = lambda sec: max(1, math.floor(max(0, sec) * 48000 + .5))
    sustain = min(1, max(0, adsr['sustain_level']))
    decay = ticks(adsr['decay_sec']) if sustain < 1 else 0
    hold = spec['hold_ticks'] if spec['hold_ticks'] is not None else 60 * 48000
    end = min((1 << 64) - 1, spec['onset'] + hold)
    release = ticks(adsr['release_sec'])
    return dict(onset=spec['onset'], hold_end=end,
                release_end=min((1 << 64) - 1, end + release),
                attack_ticks=ticks(adsr['attack_sec']), decay_ticks=decay,
                sustain_level=sustain,
                decay_lambda=float(np.float32(np.float32(6.908) / np.float32(decay))) if decay else 0.,
                release_ticks=release)


def apply_off(env, at):
    if at < env['hold_end']:
        env['hold_end'] = at
        env['release_end'] = at + env['release_ticks']


def verify(root, inputs):
    manifest = json.loads((root / 'manifest.json').read_text())
    registration = json.loads((inputs / 'registration.json').read_text())
    require(manifest['schema'] == 'i10-onset-branches-v1'
            and manifest['registration'] == registration, 'registration identity')
    source = Path(registration['source_directory'])
    for name, digest in registration['source_sha256'].items():
        require(sha(source / name) == digest, f'source changed: {name}')
    require([c['id'] for c in manifest['cases']] == registration['cases'], 'case census')
    results = []
    for case in manifest['cases']:
        original = source / case['id']
        hops = [json.loads(s) for s in (original / 'policy-inputs.jsonl').read_text().splitlines()]
        chosen = next((i for i, h in enumerate(hops) if h['now'] >= registration['after_sample']
                       and any(t['opportunity'] for b in h['batches'] for t in b['tones'])), None)
        if chosen is None:
            require(case['status'] == 'no_granted_recipe_in_recorded_interval'
                    and not (root / case['id']).exists(), 'fabricated clock opportunity')
            results.append(dict(case=case['id'], status=case['status']))
            continue
        directory = root / case['id']
        data = json.loads((directory / 'branches.json').read_text())
        current = hops[chosen]
        require(len(current['batches']) == 1, 'multiple sources')
        batch = current['batches'][0]
        require(len(batch['tones']) == len(batch['cmds']) == 1, 'unregistered simultaneous actions')
        recipe = batch['tones'][0]
        grant = recipe['opportunity']
        decision, start = grant['at'], current['now']
        end = (decision + 192000 + 511) // 512 * 512
        require((data['render_start'], data['decision_sample'], data['render_end']) == (start, decision, end), 'physical support')
        require((data['sample_rate'], data['hop_samples']) == (48000, 512), 'physical grid')
        require(data['recipe'] == recipe and data['rhythms'] == current['rhythms']
                and data['routing'] == batch['routing'] and data['policy'] == batch['body_policy'], 'issue input/recipe')
        require(data['kick'] == batch['cmds'][0]['On']['kick']
                and recipe['tone_id'] == batch['cmds'][0]['On']['tone_id'], 'onset strength or identity')
        require(data['local_default'] == 'onset_now-0', 'local default mapping')
        existing, planned_offs = {}, []
        for h in hops[:chosen]:
            for b in h['batches']:
                specs = {t['tone_id']: t for t in b['tones']}
                for cmd in b['cmds']:
                    if 'On' in cmd:
                        spec = specs[cmd['On']['tone_id']]
                        existing[spec['tone_id']] = envelope(spec)
                        at = spec['opportunity']['planned_release_at']
                        if at is not None and at >= start:
                            planned_offs.append((spec['tone_id'], at))
                    elif 'Off' in cmd and cmd['Off']['tone_id'] in existing:
                        apply_off(existing[cmd['Off']['tone_id']], cmd['Off']['off_tick'])
        existing = {tone: env for tone, env in existing.items() if env['release_end'] > start - 512}
        require(all(e['onset'] < start for e in existing.values()), 'unregistered queued onset')
        planned_offs = sorted(set(planned_offs), key=lambda x: (x[1], x[0]))
        require(data['planned_offs'] == [list(x) for x in planned_offs], 'lost or borrowed release plan')
        require(data['envelopes'] == [[tone, e] for tone, e in sorted(existing.items())], 'prefix envelope reconstruction')
        planned = copy.deepcopy(existing)
        for tone, at in planned_offs:
            if tone in planned:
                apply_off(planned[tone], at)
        require(data['planned_envelopes'] == [[tone, e] for tone, e in sorted(planned.items())], 'planned envelope projection')
        expected = {}
        require(len(data['candidates']) == len(CLASSES) * len(registration['offsets_samples']), 'candidate census')
        for candidate, (cls, offset) in zip(data['candidates'], ((c, t) for c in CLASSES for t in registration['offsets_samples'])):
            name, at = f'{cls}-{offset}', decision + offset
            active = [tone for tone, e in sorted(planned.items()) if e['onset'] <= at < e['release_end']]
            require(candidate['name'] == name and candidate['active_tones'] == active, 'candidate identity/activity')
            legal = dict(onset_now=offset == 0, delayed_onset=offset > 0, wait=offset >= 2400,
                         skip=offset == 0 and grant['intrinsic_due_at'] == decision,
                         continue_=offset == 0 and bool(active), release=bool(active),
                         gap=grant['intrinsic_period_ticks'] is not None)[cls if cls != 'continue' else 'continue_']
            action = None
            if legal:
                action = dict(issued_at=decision, at=at, excitation_at=at if cls in ['onset_now', 'delayed_onset'] else None,
                              release_at=at if cls in ['release', 'gap'] and active else None,
                              reconsider_at=at if cls == 'wait' else None, consumes_due_opportunity=cls == 'skip',
                              withhold_until=at + grant['intrinsic_period_ticks'] if cls == 'gap' else None)
                action['class'] = cls
                expected[name] = (action, active)
            require(candidate['input'] == action, 'class timing/bookkeeping')
        require([b['name'] for b in data['branches']] == list(expected), 'branch census')
        waves = {}
        pair_count = 0
        for branch in data['branches']:
            name = branch['name']
            action, active = expected[name]
            require(branch['input'] == action, 'branch input')
            off_commands = list(planned_offs)
            if action['release_at'] is not None:
                off_commands += [(tone, action['release_at']) for tone in active]
            spec = None
            if action['excitation_at'] is not None:
                spec = copy.deepcopy(recipe)
                spec['opportunity'] = None
                spec['onset'] = action['excitation_at']
                if grant['planned_release_at'] is not None:
                    off_commands += [(spec['tone_id'], spec['onset'] + grant['planned_release_at'] - decision)]
            require(branch['onset'] == spec, 'fixed onset recipe or timing')
            off_commands = sorted(set(off_commands), key=lambda x: (x[1], x[0]))
            require(branch['off_commands'] == [list(x) for x in off_commands], 'branch release schedule')
            branch_envelopes = copy.deepcopy(existing)
            if spec:
                branch_envelopes[spec['tone_id']] = envelope(spec)
            for tone, at in off_commands:
                if tone in branch_envelopes:
                    apply_off(branch_envelopes[tone], at)
            last_end = max((e['release_end'] for e in branch_envelopes.values()), default=decision)
            waves[name] = []
            for bus in range(2):
                pcm = np.fromfile(directory / f'{name}-bus{bus}.f32le', dtype='<f4')
                require(len(pcm) == end - start and np.isfinite(pcm).all(), 'PCM physical support')
                routed = data['routing']['to_habitat' if bus == 0 else 'to_presentation']
                require(routed or not np.any(pcm), 'unrouted PCM')
                require(not np.any(pcm[max(0, last_end - start):]), 'natural release tail endpoint')
                waves[name].append(pcm)
                default = waves['onset_now-0'][bus]
                require(np.array_equal(pcm[:decision-start].view('<u4'), default[:decision-start].view('<u4')), 'pre-decision PCM changed')
                energy = (pcm.astype(float).reshape(-1, 512)**2).mean(axis=1)
                saved = branch['buses'][bus]
                require(saved['bus'] == bus and np.shape(saved['energy_per_hop']) == energy.shape
                        and np.allclose(saved['energy_per_hop'], energy, rtol=2e-14, atol=1e-20), 'hop energy')
                require([p['samples'] for p in saved['paired']] == [12000,48000,96000,192000], 'paired windows')
                for pair in saved['paired']:
                    begin, stop = decision - start, decision - start + pair['samples']
                    actual = float(np.mean(pcm[begin:stop].astype(float)**2))
                    baseline = float(np.mean(default[begin:stop].astype(float)**2))
                    require(np.allclose([pair['mean_square'], pair['default_mean_square'], pair['difference']],
                                        [actual, baseline, actual-baseline], rtol=2e-12, atol=1e-19), 'paired local-default energy')
                    pair_count += 1
                if action['class'] == 'gap' and name.replace('gap-', 'release-') in waves:
                    require(np.array_equal(pcm.view('<u4'), waves[name.replace('gap-', 'release-')][bus].view('<u4')), 'gap tail lost')
        # Wait is enumerated before skip; compare after every branch exists.
        if 'skip-0' in waves:
            for name in waves:
                if name.startswith('wait-') or name == 'continue-0':
                    require(all(np.array_equal(a.view('<u4'), b.view('<u4'))
                                for a,b in zip(waves[name],waves['skip-0'])), 'wait/skip/continue PCM mismatch')
        for bus in range(2):
            original_pcm = (original / f'prefix-bus{bus}.f32le').read_bytes() + (original / f'actual-default-bus{bus}.f32le').read_bytes()
            require((directory / f'prefix-bus{bus}.f32le').read_bytes() == original_pcm[:start*4], 'actual prefix mismatch')
            require(waves['onset_now-0'][bus][:512].tobytes() == original_pcm[start*4:(start+512)*4], 'actual default hop mismatch')
        results.append(dict(case=case['id'], status='acquired', decision_sample=decision,
                            pre_decision_samples=decision-start, branches=len(waves),
                            excluded=len(data['candidates'])-len(waves), paired_windows=pair_count,
                            future_hops=(end-start)//512, rendered_buses=2))
    return dict(schema='i10-onset-branches-verification-v1', cases=results,
                hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                        if p.is_file() and (p.name=='branches.json' or p.suffix=='.f32le')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.inputs)
    (args.root / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'}, indent=2))
