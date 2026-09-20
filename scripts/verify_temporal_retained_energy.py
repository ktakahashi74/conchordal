"""Reconstruct command-order retained tone priors from saved causal renderer inputs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from verify_temporal_policy_defaults import require, sha, verify as verify_policy
from verify_temporal_scheduled_release import gain, rows


def frozen_envelope(spec):
    adsr = spec['adsr']
    ticks = lambda key: max(1, int(float(np.float32(adsr[key])) * 48000 + 0.5))
    hold = spec['hold_ticks'] if spec['hold_ticks'] is not None else 60 * 48000
    end = min(2**64 - 1, spec['onset'] + hold)
    decay = ticks('decay_sec') if adsr['sustain_level'] < 1 else 0
    return dict(onset=spec['onset'], hold_end=end,
                release_end=min(2**64 - 1, end + ticks('release_sec')),
                attack_ticks=ticks('attack_sec'), decay_ticks=decay,
                sustain_level=float(np.float32(adsr['sustain_level'])),
                decay_lambda=float(np.float32(np.float32(6.908) / np.float32(decay))) if decay else 0.,
                release_ticks=ticks('release_sec'))


def control_snapshot(tone, now):
    """Rebuild the corpus' issued EntrainPulse state with f32 scalar steps."""
    spec = tone['modulator']
    require(list(spec) == ['EntrainPulse'], 'control corpus must register other modulators separately')
    m = spec['EntrainPulse']
    dt = np.float32(1 / 48000)
    state, level = m['initial_state'], np.float32(m['initial_env_level'])
    onset = tone['envelope']['onset']
    require(tone['kick'] > 0, 'corpus requires an actual positive excitation')
    steps = now - tone['born']
    if now > onset:
        state, level, steps = 'Attack', np.float32(0), now - onset
    floor = np.float32(m['sustain_level'])
    attack = np.float32(np.float32(m['attack_step']) * dt)
    decay = np.float32(math.exp(float(np.float32(-np.float32(m['decay_rate']) * dt))))
    for _ in range(steps):
        if state == 'Attack':
            level = np.float32(level + attack)
            if level >= 1:
                state, level = 'Decay', np.float32(1)
        elif state == 'Decay':
            level = np.float32(level * decay)
            if level <= np.float32(floor + np.float32(.001)):
                level = floor
                if floor <= 0:
                    state = 'Idle'
                # The registered positive sustain is a fixed point.
                if floor > 0:
                    break
        else:
            require(m['autonomous_pulse'] is None, 'unreconstructed autonomous phase')
            break
    pulse = m['autonomous_pulse']
    model = {k: float(np.float32(m[k])) for k in ['attack_step', 'decay_rate', 'sustain_level']}
    model.update(state=state, env_level=float(level), autonomous_retrigger=bool(pulse and pulse['retrigger']))
    return dict(issued_at=now, sample_dt=float(dt), starts_at=max(now, onset),
                kick_at=max(now, onset) if now <= onset else None,
                model={'EntrainPulse': model})


def control_amplitude(control, tick, constant):
    require(tick >= control['issued_at'], 'acausal amplitude target')
    until = control.get('valid_until')
    require(until is None or tick < until, 'amplitude crosses unmodeled update')
    smoothing = control.get('amplitude_smoothing')
    if smoothing is None:
        return constant
    current, target, alpha = [float(np.float32(smoothing[k])) for k in ['current', 'target', 'alpha']]
    require(all(math.isfinite(v) for v in [current, target, alpha]) and min(current,target) >= 0 and 0 <= alpha <= 1, 'invalid amplitude state')
    start = control['issued_at']
    updates = control.get('amplitude_updates')
    if updates:
        for event in updates['events'][:updates['len']]:
            at = max(control['issued_at'], event['at_sample'])
            if at > tick:
                break
            remaining = (1-alpha)**(at-start)
            current = current*remaining + target*(1-remaining)
            target, start = float(np.float32(event['target'])), at
    remaining = (1-alpha)**(tick-start+1)
    return current*remaining + target*(1-remaining)


def control_for(tone, intervention=None):
    control = dict(tone['control'])
    clear_at = max(tone['envelope']['hold_end'], control['issued_at'])
    for release in [tone.get('scheduled_release', tone.get('plan')), intervention]:
        if release:
            clear_at = min(clear_at, max(release['off_sample'], release['apply_at_sample'], control['issued_at']))
    if control.get('valid_until') is not None and control['valid_until'] > clear_at:
        control.pop('valid_until')
    updates = control.get('amplitude_updates')
    if updates:
        events = [e for e in updates['events'][:updates['len']] if max(e['at_sample'], control['issued_at']) <= clear_at]
        control['amplitude_updates'] = dict(events=events, len=len(events))
    return control


def control_gain(control, tick):
    require(tick >= control['issued_at'], 'acausal control target')
    until = control.get('valid_until')
    require(until is None or tick < until, 'gain crosses unmodeled update')
    if control['starts_at'] is None or tick < control['starts_at']:
        return 0.
    m = control['model']['EntrainPulse']
    state, level = m['state'], m['env_level']
    count = tick - control['issued_at'] + 1
    if control['kick_at'] is not None and tick >= control['kick_at']:
        state, level, count = 'Attack', 0., tick - control['kick_at'] + 1
    floor = m['sustain_level']
    require(not m['autonomous_retrigger'] or (state != 'Idle' and floor > 0), 'unknown autonomous future')
    if state == 'Idle':
        return 0.
    dt = np.float32(control['sample_dt'])
    if state == 'Attack':
        step = float(np.float32(np.float32(m['attack_step']) * dt))
        attack = max(1, math.ceil((1 - level) / step))
        if count < attack:
            return max(0., level + count * step)
        level, count = 1., count - attack
    if count:
        decay = float(np.float32(math.exp(float(np.float32(-np.float32(m['decay_rate']) * dt)))))
        level *= decay ** count
        if level <= floor + float(np.float32(.001)):
            level = floor
    return level if level > 1e-6 else 0.


def verify(root, inputs, previous, controls=False):
    verify_policy(root, inputs)
    registration = json.loads((inputs / 'registration.json').read_text())
    result = []
    for case in registration['cases']:
        directory = root / case['id']
        trace = rows(directory / 'policy-inputs.jsonl')
        report = rows(directory / 'ordinary.jsonl')
        prior = previous / case['id']
        require(trace == rows(prior / 'policy-inputs.jsonl'), 'causal renderer inputs changed')
        old = {(r['issued_at_sample'], r['tone_id'], r['action']): r
               for r in rows(prior / 'ordinary.jsonl') if r['type'] == 'self_sound_outcome'}
        reports = {}
        for r in report:
            if r['type'] != 'self_sound_outcome':
                continue
            key = (r['issued_at_sample'], r['tone_id'], r['action'])
            require(key not in reports, 'duplicate command key in this corpus')
            reports[key] = r
            require({k: v for k, v in r.items() if k != 'prediction'} ==
                    {k: v for k, v in old[key].items() if k != 'prediction'}, 'action observation changed')
            if r.get('prediction'):
                require([e['target'] for e in r['prediction']['source_energy']] ==
                        [e['target'] for e in old[key]['prediction']['source_energy']], 'energy teacher changed')
        retained = {}
        forecasts = coordinates = nonzero = changed = 0
        maximum_error = 0.
        old_loss = new_loss = 0.
        old_model_loss, new_model_loss = [0.] * 3, [0.] * 3
        scored_windows = 0
        for hop in trace:
            now = hop['now']
            retained = {k: v for k, v in retained.items() if v['envelope']['release_end'] > now}
            for batch in hop['batches']:
                source, generation = batch['source_id'], batch['source_generation']
                for cmd in batch['cmds']:
                    kind, cmd = next(iter(cmd.items()))
                    require(kind != 'Update', 'this reconstruction requires fixed tone parameters')
                    key = (source, cmd['tone_id'])
                    if kind == 'On':
                        spec = next(t for t in batch['tones'] if t['tone_id'] == cmd['tone_id'])
                        receipt = spec['opportunity']
                        off = receipt['planned_release_at'] if receipt else None
                        plan = None if off is None else dict(apply_at_sample=now + (off - now) // 512 * 512,
                                                            off_sample=off)
                        retained[key] = dict(envelope=frozen_envelope(spec), generation=generation,
                                             amplitude=float(np.float32(np.float32(spec['amp']) * np.float32(spec['body']['amp_scale']))),
                                             routing=[batch['routing']['to_habitat'], batch['routing']['to_presentation']], plan=plan,
                                             modulator=spec['render_modulator'], born=now, kick=cmd['kick']['strength'])
                    elif key in retained and retained[key]['generation'] == generation:
                        envelope = retained[key]['envelope']
                        if cmd['off_tick'] < envelope['hold_end']:
                            envelope['hold_end'] = cmd['off_tick']
                            envelope['release_end'] = min(2**64 - 1, cmd['off_tick'] + envelope['release_ticks'])
                    out_key = (now, cmd['tone_id'], 'onset' if kind == 'On' else 'release')
                    out = reports[out_key]
                    prediction = out.get('prediction')
                    if prediction is None:
                        continue
                    forecasts += 1
                    command = retained[key]
                    for name, expected in command['envelope'].items():
                        value = prediction['envelope'][name]
                        require(np.float32(value) == np.float32(expected) if name in ['sustain_level', 'decay_lambda']
                                else value == expected, 'command envelope reconstruction')
                    plan = command['plan'] if kind == 'On' else None
                    require(prediction['scheduled_release'] == plan, 'command plan changed')
                    if controls:
                        for tone in retained.values():
                            tone['control'] = control_snapshot(tone, now)
                        actual = prediction['control']
                        expected = command['control']
                        require({k: v for k, v in actual.items() if k not in ['model', 'sample_dt']} ==
                                {k: v for k, v in expected.items() if k not in ['model', 'sample_dt']}, 'control timing or start changed')
                        require(np.float32(actual['sample_dt']) == np.float32(expected['sample_dt']), 'control sample duration')
                        for name, value in expected['model']['EntrainPulse'].items():
                            found = actual['model']['EntrainPulse'][name]
                            require(abs(found - value) < 2e-6 if name == 'env_level' else
                                    np.float32(found) == np.float32(value) if isinstance(value, float) else found == value,
                                    'actual issued control state differs from sample-stepped reconstruction')
                    entries = [(k, v) for k, v in sorted(retained.items()) if k[0] == source]
                    require(len(entries) <= 64, 'corpus unexpectedly exceeds retained inventory cap')
                    for bus, energy in enumerate(prediction['source_energy']):
                        require(energy['model'] in (['source_energy_log1p_residual_v4','source_energy_log1p_residual_v5','source_energy_log1p_residual_v6','source_energy_log1p_residual_v7'] if controls else ['source_energy_log1p_residual_v3']), 'model version')
                        previous_energy = old[out_key]['prediction']['source_energy'][bus]
                        if not controls:
                            require(energy['command_fixed_energy'] == previous_energy['predictions'][0],
                                    'original command-only proxy changed')
                        else:
                            require(all(energy['retained']['control_supported']), 'unexpected unsupported corpus control')
                        require(energy['predictions'][2] == previous_energy['predictions'][2],
                                'action-independent control changed')
                        body = [v for k, v in entries if k != key and v['generation'] == generation and v['routing'][bus]]
                        require(energy['retained']['complete'] and energy['retained']['scanned_entries'] == len(entries)
                                and energy['retained']['included_tones'] == len(body), 'retained owner/routing census')
                        for k, (left, right) in enumerate(energy['windows']):
                            tick = (left + right) // 2
                            total = 0.
                            for tone in body:
                                p = tone['plan']
                                off = p['off_sample'] if p is not None and tick >= p['apply_at_sample'] else None
                                total += (control_amplitude(tone['control'], tick, tone['amplitude']) if controls else tone['amplitude'])**2 * gain(tone['envelope'], tick, off)**2 / 2 * (control_gain(tone['control'], tick)**2 if controls else 1.)
                            off = plan['off_sample'] if plan is not None and tick >= plan['apply_at_sample'] else None
                            issued = (control_amplitude(command['control'], tick, command['amplitude']) if controls else command['amplitude'])**2 * gain(command['envelope'], tick, off)**2 / 2 if command['routing'][bus] else 0.
                            if controls:
                                issued *= control_gain(command['control'], tick)**2
                            for actual, expected in [(energy['retained']['fixed_energy'][k], total),
                                                     (energy['command_fixed_energy'][k], issued),
                                                     (energy.get('incoherent_fixed_energy',energy['predictions'][0])[k], total + issued)]:
                                maximum_error = max(maximum_error, abs(actual - expected))
                                require(abs(actual - expected) <= max(1e-14, abs(expected)*5e-7), 'retained or command fixed energy')
                            if energy['model'] in ['source_energy_log1p_residual_v5','source_energy_log1p_residual_v6','source_energy_log1p_residual_v7']:
                                coherent = energy['coherent_sine_energy'][k]
                                require(energy['predictions'][0][k] == (coherent if coherent is not None else energy['incoherent_fixed_energy'][k]), 'declared coherent choice or fallback')
                            target = energy['target'][k]
                            if target is not None:
                                scored_windows += 1
                                for model in range(3):
                                    old_model_loss[model] += (old[out_key]['prediction']['source_energy'][bus]['predictions'][model][k] - target)**2
                                    new_model_loss[model] += (energy['predictions'][model][k] - target)**2
                                old_fixed = old[out_key]['prediction']['source_energy'][bus]['predictions'][0][k]
                                old_loss += (old_fixed - target)**2
                                new_loss += (energy['predictions'][0][k] - target)**2
                                changed += old_fixed != energy['predictions'][0][k]
                            nonzero += total > 0
                            coordinates += 1
        unchanged = 0
        for path in prior.rglob('*.f32le'):
            require(sha(path) == sha(directory / path.relative_to(prior)), 'actual PCM changed')
            unchanged += 1
        observed = [{(r['observation']['bus'], r['observation']['source_epoch'], r['observation']['frame_id']): r['observation']
                     for r in source if r['type'] == 'temporal_observation' and r['observation']['frame_id'] is not None}
                    for source in [report, rows(prior / 'ordinary.jsonl')]]
        common = observed[0].keys() & observed[1].keys()
        require(bool(common), 'no common passive frames')
        for identity in common:
            require({k: v for k, v in observed[0][identity].items() if k != 'delivery_delay_us'} ==
                    {k: v for k, v in observed[1][identity].items() if k != 'delivery_delay_us'},
                    'passive observation changed')
        result.append(dict(case=case['id'], verified_forecasts=forecasts, bus_window_coordinates=coordinates,
                           nonzero_retained_coordinates=nonzero, changed_fixed_predictions=changed,
                           max_absolute_error=maximum_error, unchanged_pcm_files=unchanged,
                           paired_scored_windows=scored_windows,
                           old_model_squared_error_sum=old_model_loss, new_model_squared_error_sum=new_model_loss,
                           unchanged_common_passive_frames=len(common),
                           unmatched_passive_frames=[len(o)-len(common) for o in observed],
                           old_fixed_squared_error_sum=old_loss, new_fixed_squared_error_sum=new_loss))
    return dict(schema='i10-control-amplitude-verification-v1' if controls else 'i10-retained-energy-verification-v1', cases=result,
                limits=('Causal control state/component arithmetic and actual target/PCM identity. Analytic control-modulated sine-envelope proxy omits coherent phase, backend dynamics, sine impulse boost and later commands; v5 carrier means are checked separately with seeded replay and explicit wave summation, not by this component verifier; not calibrated candidate/default transfer.' if controls else
                        'Causal input/component arithmetic and actual target/PCM identity. Incoherent sine-envelope proxy omits phase interference, backend/modulator dynamics and later commands; not calibrated candidate/default transfer.'),
                hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.glob('*/*.jsonl'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('previous', type=Path)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    result = verify(args.root, args.inputs, args.previous, args.controls)
    (args.root / ('control-amplitude-verification.json' if args.controls else 'retained-energy-verification.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'hashes'}, indent=2))
