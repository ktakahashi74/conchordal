"""Verify causal seven-class body priors and compare them with actual local-default PCM."""

import argparse
import copy
import itertools
import math
import cmath
import json
from pathlib import Path

import numpy as np

from verify_temporal_policy_defaults import require, sha
from verify_temporal_scheduled_release import gain, rows
from verify_temporal_retained_energy import frozen_envelope, control_snapshot, control_gain, control_amplitude, control_for


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(actual.keys() - {'sine', 'bank'} == expected.keys() - {'sine', 'bank'}, label + ' keys')
        for k, value in expected.items():
            same(actual[k], value, label + '.' + k)
    elif isinstance(expected, float):
        require(abs(actual - expected) < 2e-6 if label.endswith('env_level') else
                np.float32(actual) == np.float32(expected), label)
    else:
        require(actual == expected, label)


def tone_from_spec(spec, cmd, batch, now):
    off = spec['opportunity']['planned_release_at'] if spec.get('opportunity') else None
    return dict(envelope=frozen_envelope(spec), generation=batch['source_generation'],
                frequency=spec['freq_hz'], body_kind=spec['body']['kind'], tone_id=spec['tone_id'],
                amplitude=float(np.float32(np.float32(spec['amp']) * np.float32(spec['body']['amp_scale']))),
                routing=[batch['routing']['to_habitat'], batch['routing']['to_presentation']],
                plan=None if off is None else dict(apply_at_sample=now+(off-now)//512*512, off_sample=off),
                modulator=spec['render_modulator'], born=now, kick=cmd['kick']['strength'])


def kernel_input(tone, now):
    return dict(amplitude=tone['amplitude'], envelope=tone['envelope'],
                control=control_snapshot(tone, now), scheduled_release=tone['plan'])


def tone_energy(tone, tick, intervention, modulated=True):
    offs = [p['off_sample'] for p in [tone['scheduled_release'], intervention]
            if p is not None and tick >= p['apply_at_sample']]
    outer = gain(tone['envelope'], tick, min(offs) if offs else None)
    if outer == 0:
        return 0.
    control = control_for(tone, intervention) if modulated else None
    return (control_amplitude(control, tick, tone['amplitude']) if modulated else tone['amplitude'])**2 * outer**2 / 2 * (control_gain(control, tick)**2 if modulated else 1.)


def expected_sine(tone, now):
    if tone['body_kind'] != 'Sine':
        return None
    mask = 2**64 - 1
    rotate = lambda x, n: ((x << n) | (x >> (64-n))) & mask
    def mix(x):
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    onset = tone['envelope']['onset']
    seed = mix(1 ^ rotate(onset,21) ^ rotate(tone['tone_id'],42) ^ 0x9e3779b97f4a7c15)
    bits = mix((seed+0x9e3779b97f4a7c15)&mask) >> 11
    tau = np.float32(2*math.pi)
    angle = float(np.float32(np.float32(bits/2**53)*tau))
    x,y = np.float32(math.cos(angle)),np.float32(math.sin(angle))
    angle = float(np.float32(np.float32(tau*np.float32(tone['frequency']))/np.float32(48000)))
    c,d = np.float32(math.cos(angle)),np.float32(math.sin(angle))
    decay = np.float32(math.exp(float(np.float32(-np.float32(1/48000)/np.float32(.08)))))
    boost = np.float32(np.float32(tone['kick'])*np.float32(.2))
    for tick in range(onset,now):
        x,y = np.float32(np.float32(c*x)-np.float32(d*y)),np.float32(np.float32(d*x)+np.float32(c*y))
        if tick < tone['envelope']['release_end']:
            boost = np.float32(boost*decay)
    return dict(first_sample=max(onset,now),state=[float(x),float(y)],rotation=[float(c),float(d)],
                boost=float(boost),boost_decay=float(decay))


def check_sine(actual, tone, now):
    expected = expected_sine(tone,now)
    if expected is None:
        require(actual is None, 'non-sine carrier invented')
        return
    require(actual['first_sample'] == expected['first_sample'], 'carrier start')
    for field in ['state','rotation']:
        require(max(abs(a-b) for a,b in zip(actual[field],expected[field])) < 2e-5,
                'carrier '+field+' differs from independent seeded scalar replay')
    require(abs(actual['boost']-expected['boost']) < 1e-6 and actual['boost_decay']==expected['boost_decay'], 'sine boost state')


SPAN = 750


def released(tone, tick, intervention):
    """Renderer envelope after every release that has been delivered by `tick`."""
    envelope = dict(tone['envelope'])
    for plan in [tone['scheduled_release'], intervention]:
        if plan is not None and tick >= plan['apply_at_sample'] and plan['off_sample'] < envelope['hold_end']:
            envelope['hold_end'] = plan['off_sample']
            envelope['release_end'] = plan['off_sample']+envelope['release_ticks']
    return envelope


def release_end(tone, tick, intervention):
    return released(tone, tick, intervention)['release_end']


def attack_of(envelope):
    return min(envelope['attack_ticks'], max(envelope['hold_end']-envelope['onset'], 1))


def control_attack(tone):
    control = tone['control']
    if control is None or 'EntrainPulse' not in control['model']:
        return None
    model = control['model']['EntrainPulse']
    step = float(np.float32(model['attack_step'])*np.float32(control['sample_dt']))
    if not math.isfinite(step) or step <= 0:
        return None
    if control['kick_at'] is not None:
        origin, level = max(control['kick_at'], control['issued_at']), 0.
    elif model['state'] == 'Attack':
        origin, level = control['issued_at'], model['env_level']
    else:
        return None
    return [origin, origin+int(max(math.ceil((1-level)/step), 1))]


def breakpoints(tone, intervention):
    out = []
    envelope = dict(tone['envelope'])
    for plan in [None, tone['scheduled_release'], intervention]:
        if plan is not None:
            if plan['off_sample'] >= envelope['hold_end']:
                continue
            out.append(plan['apply_at_sample'])
            envelope['hold_end'] = plan['off_sample']
            envelope['release_end'] = plan['off_sample']+envelope['release_ticks']
        attack = attack_of(envelope)
        out += [envelope['onset'], envelope['onset']+attack, envelope['onset']+attack+envelope['decay_ticks'],
                envelope['hold_end'], envelope['release_end']]
    if tone.get('sine') is not None: out.append(tone['sine']['first_sample'])
    if tone.get('bank') is not None: out.append(tone['bank']['first_sample'])
    control = tone['control']
    if control is not None:
        out += [at for at in [control['starts_at'], control['kick_at']] if at is not None]
        if control_attack(tone) is not None: out.append(control_attack(tone)[1])
    require(len(out) <= 24, 'breakpoint capacity')
    return out


def fast_span(tone, start, stop, intervention):
    envelope = released(tone, start, intervention)
    begin = envelope['onset']+attack_of(envelope)
    end = begin+envelope['decay_ticks']
    inside = lambda a, b: start < b and stop > a
    out = []
    lam = float(np.float32(envelope['decay_lambda']))
    if inside(begin, end) and lam > 0: out.append(int(.1/(2*lam)))
    if inside(envelope['onset'], begin): out.append(attack_of(envelope)//32)
    attack = control_attack(tone)
    if attack is not None and inside(*attack): out.append((attack[1]-attack[0])//32)
    return min(out) if out else None


def lane_point(bank, lane, tick):
    """Independent form of one lane: magnitude and phase for the oscillator, matrix powers
    collapsed to an eigen-decomposition for the resonator. Returns (z, omega, log_decay)
    with sample value `Re(z exp((log_decay + i omega)(n - tick)))`."""
    first = bank['first_sample']
    kind, response = next(iter(bank['response'].items()))
    steps = tick-first+1
    held = max(tick, first)
    x, y = lane['state']
    if kind == 'Oscillator':
        rotation = complex(*lane['step'])
        omega = cmath.phase(rotation)
        if held < response['refresh_at']:
            mask = lane['held']
        else:
            refreshed = held-(held-response['refresh_at'])%response['refresh_period']
            env = response['spectral_env']*response['spectral_decay']**(refreshed-first)
            floor = response['spectral_floor']
            mask = lane['gain']*min(max(floor+(1-floor)*env, floor), 1.)**lane['damping']
        drive = response['drive_env']*response['drive_decay']**(held-first)
        z = complex(x, y)*cmath.exp(1j*omega*steps)*abs(rotation)**steps*mask*(1+.25*drive)
        # Sample is the imaginary part of the rotating state.
        return z/1j, omega, 0.
    r, e = lane['step']
    values, vectors = np.linalg.eig(np.array([[r, -r*e], [r*r*e, r*(1-r*e*e)]], dtype=np.complex128))
    weights = np.linalg.solve(vectors, np.array([x, y], dtype=np.complex128))
    k = int(np.argmax(values.imag))
    # The conjugate pair sums to twice the real part of one branch.
    z = 2*lane['gain']*vectors[1, k]*weights[k]*values[k]**steps
    return z, cmath.phase(values[k]), math.log(abs(values[k]))


def trend(energies, lo, hi, tick):
    a, b, c = energies
    mean = (a+4*b+c)/6
    n = hi-lo
    rate = math.log(c/a)/(2*(n-1)) if a > 0 and c > 0 and n > 1 else 0.
    if not math.isfinite(rate) or abs(rate*n) > 4:
        rate = 0.
    shape = 1. if rate == 0 else math.sinh(rate*n)/(rate*n)
    return mean/shape*math.exp(2*rate*(tick-(lo+(n-1)/2))), rate


def span_carriers(tone, intervention, a, b, tick):
    """(z, omega, log_decay, start, end) carriers of one tone for one span, or None."""
    envelope = released(tone, tick, intervention)
    sine, bank = tone.get('sine'), tone.get('bank')
    if sine is None and bank is None:
        return None
    first = (sine or bank)['first_sample']
    start, end = max(first, envelope['onset']), envelope['release_end']
    lo, hi = max(a, start), min(b, end)
    def sine_carrier(energy, rate):
        if energy == 0:
            return []
        if control_gain(control_for(tone, intervention), sine['first_sample']) <= 0 or tick < sine['first_sample']:
            return None
        elapsed = tick-sine['first_sample']
        rotation = complex(*sine['rotation'])
        z = complex(*sine['state'])*rotation**(elapsed+1)
        z *= math.sqrt(2*energy)*(1+sine['boost']*sine['boost_decay']**elapsed)
        return [(z/1j, cmath.phase(rotation), rate, start, end)]
    if lo >= hi or (sine is not None and tick < lo):
        energy = tone_energy(tone, tick, intervention)
        if sine is not None:
            return sine_carrier(energy, 0.)
        if energy == 0:
            return []
        return [(z*math.sqrt(2*energy), omega, decay, start, end)
                for z, omega, decay in [lane_point(bank, lane, tick) for lane in bank['lanes'][:bank['len']]]]
    nodes = [lo, lo+(hi-lo)//2, hi-1]
    energies = [tone_energy(tone, at, intervention) for at in nodes]
    if all(e == 0 for e in energies):
        return []
    if sine is not None:
        return sine_carrier(*trend(energies, lo, hi, tick))
    out = []
    for lane in bank['lanes'][:bank['len']]:
        z, omega, decay = lane_point(bank, lane, tick)
        base = abs(z)
        if base <= 0:
            return None
        ratios = [1. if 'Resonator' in bank['response'] else abs(lane_point(bank, lane, at)[0])/base for at in nodes]
        energy, rate = trend([e*g*g for e, g in zip(energies, ratios)], lo, hi, tick)
        out.append((z*math.sqrt(2*energy), omega, decay+rate, start, end))
    return out


def coherent_energy(tones,left,right,tick):
    del tick
    edges = {left, right}
    for tone, intervention in tones:
        edges |= {at for at in breakpoints(tone, intervention) if left < at < right}
    require(len(edges) <= 130, 'span edge capacity')
    edges = sorted(edges)
    total = 0.
    for start, stop in zip(edges, edges[1:]):
        limits = [v for v in [fast_span(tone, start, stop, intervention) for tone, intervention in tones] if v is not None]
        limit = max(min([SPAN]+limits), 16)
        spans = -(-(stop-start)//limit)
        for k in range(spans):
            a, b = [start+(stop-start)*edge//spans for edge in [k, k+1]]
            middle = a+(b-a)//2
            indices = np.arange(a, b, dtype=np.int64)
            wave = np.zeros(b-a)
            for tone, intervention in tones:
                carriers = span_carriers(tone, intervention, a, b, middle)
                if carriers is None:
                    return None
                for z, omega, decay, first, last in carriers:
                    active = (indices >= first) & (indices < last)
                    offset = (indices-middle).astype(np.float64)
                    wave += np.where(active, (z*np.exp((decay+1j*omega)*offset)).real, 0.)
            total += float(np.sum(wave*wave))
    return total/(right-left)


def verify(root, inputs):
    registration = json.loads((inputs / 'registration.json').read_text())
    coherent = bool(registration.get('coherent_sine'))
    model_count = 4 if coherent else 3
    source = Path(registration['source_directory'])
    targets = Path(registration['target_directory'])
    for directory, field in [(source, 'source_sha256'), (targets, 'target_sha256')]:
        for name, digest in registration[field].items():
            require(sha(directory / name) == digest, 'registered source or actual target changed')
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['registration'] == registration, 'prediction registration differs')
    require([c['id'] for c in manifest['cases']] == registration['cases'], 'registered case coverage')
    results = []
    for case in manifest['cases']:
        name = case['id']
        trace = rows(source / name / 'policy-inputs.jsonl')
        stop = next((i for i, h in enumerate(trace) if h['now'] >= registration['after_sample'] and
                     any(t['opportunity'] for b in h['batches'] for t in b['tones'])), None)
        if stop is None:
            require(case['status'] == 'no_granted_recipe_in_recorded_interval', 'invented grant')
            results.append(dict(case=name, status='no_granted_recipe_in_recorded_interval'))
            continue
        prefix, current = trace[:stop], trace[stop]
        now = current['now']
        data = json.loads((root / name / 'predictions.json').read_text())
        target = json.loads((targets / name / 'branches.json').read_text())
        require(data['schema'] == 'i10-seven-class-energy-v1' and data['model'] == registration['model'], 'prediction version')
        require(case['status'] == 'predicted' and case['branches'] == len(data['predictions']), 'manifest branch coverage')
        require(data['render_start'] == now == target['render_start'], 'causal issue hop')
        batch = current['batches'][0]
        recipe = batch['tones'][0]
        cmd = batch['cmds'][0]['On']
        decision = recipe['opportunity']['at']
        require(data['decision_sample'] == decision == target['decision_sample'], 'decision sample')
        require(data['recipe'] == recipe == target['recipe'], 'fixed actual recipe')
        require(data['receipt'] == recipe['opportunity'], 'actual receipt')
        require(data['candidates'] == target['candidates'], 'class timing, activity or legality changed')
        require(data['local_default'] == target['local_default'] == 'onset_now-0', 'local default')
        require(data['routing'] == [batch['routing']['to_habitat'], batch['routing']['to_presentation']], 'actual routing')
        inventory = {}
        for hop in prefix:
            inventory = {k: v for k, v in inventory.items() if v['envelope']['release_end'] > hop['now']}
            for owner in hop['batches']:
                require((owner['source_id'], owner['source_generation']) == (1, 0), 'single registered owner')
                for item in owner['cmds']:
                    kind, command = next(iter(item.items()))
                    require(kind != 'Update', 'unregistered parameter trajectory')
                    key = command['tone_id']
                    if kind == 'On':
                        spec = next(t for t in owner['tones'] if t['tone_id'] == key)
                        inventory[key] = tone_from_spec(spec, command, owner, hop['now'])
                    elif key in inventory:
                        env = inventory[key]['envelope']
                        if command['off_tick'] < env['hold_end']:
                            env['hold_end'] = command['off_tick']
                            env['release_end'] = command['off_tick'] + env['release_ticks']
        require([r[0] for r in data['retained']] == sorted(inventory), 'actual retained inventory')
        for tone_id, route, model in data['retained']:
            same(route, inventory[tone_id]['routing'], 'retained route')
            same(model, kernel_input(inventory[tone_id], now), 'retained control')
            if coherent: check_sine(model['sine'],inventory[tone_id],now)
        issued = tone_from_spec(recipe, cmd, batch, now)
        same(data['issued'], kernel_input(issued, now), 'issued control')
        if coherent: check_sine(data['issued']['sine'],issued,now)
        by_target = {b['name']: b for b in target['branches']}
        require([p['name'] for p in data['predictions']] == list(by_target), 'branch set changed')
        local_default = np.array([np.fromfile(targets / name / f'onset_now-0-bus{bus}.f32le', dtype='<f4').astype(np.float64)
                                  for bus in range(2)])
        means, teachers = {}, {}
        numerical_error = 0.
        coherent_error = 0.
        points = 0
        for branch in data['predictions']:
            actual = by_target[branch['name']]
            action = branch['input']
            require(action == actual['input'], 'candidate input')
            active = next(c['active_tones'] for c in target['candidates'] if c['name'] == branch['name'])
            off = action['release_at']
            intervention = None if off is None else dict(apply_at_sample=now+(off-now)//512*512, off_sample=off)
            require(branch['intervention'] == intervention, 'candidate release delivery')
            if action['excitation_at'] is not None:
                at = action['excitation_at']
                shifted = copy.deepcopy(recipe)
                shifted['onset'] = at
                shifted['opportunity'] = None
                born = now + (at-now)//512*512
                added = tone_from_spec(shifted, cmd, batch, born)
                plan = recipe['opportunity']['planned_release_at']
                if plan is not None:
                    plan += at - decision
                    added['plan'] = dict(apply_at_sample=now+(plan-now)//512*512, off_sample=plan)
                expected_added = kernel_input(added, born)
                same(branch['added'], expected_added, 'shifted recipe projection')
                if coherent: check_sine(branch['added']['sine'],added,born)
            else:
                require(branch['added'] is None, 'invented excitation')
            branch_means, branch_actual = [], []
            require(len(branch['buses']) == 2, 'two-bus coverage')
            for bus, windows in enumerate(branch['buses']):
                require(len(windows) == len(registration['window_samples']), 'physical-window coverage')
                wave = np.fromfile(targets / name / f"{branch['name']}-bus{bus}.f32le", dtype='<f4').astype(np.float64)
                bus_means, bus_actual = [], []
                for index, (window, width) in enumerate(zip(windows, registration['window_samples'])):
                    require((window['start'], window['end']) == (decision, decision+width), 'physical window')
                    require(len(window['points']) == 16 and len(window['energies']) == 16, 'quadrature coverage')
                    control_means = [0.] * model_count
                    for k, (tick, predicted) in enumerate(zip(window['points'], window['energies'])):
                        edges = [decision+(width*edge+15)//16 for edge in [k, k+1]]
                        require(tick == sum(edges)//2, 'midpoint coordinate')
                        expected = [0., 0., 0.]
                        for tone_id, route, model in data['retained']:
                            if route[bus]:
                                extra = intervention if tone_id in active else None
                                expected[0] += tone_energy(model, tick, extra)
                                expected[1] += tone_energy(model, tick, extra, modulated=False)
                        if data['routing'][bus] and branch['added'] is not None:
                            energy = tone_energy(branch['added'], tick, None)
                            expected[0] += energy
                            expected[1] += tone_energy(branch['added'], tick, None, modulated=False)
                            expected[2] += energy
                        if coherent:
                            contributors = [(model,intervention if tone_id in active else None)
                                            for tone_id,route,model in data['retained'] if route[bus]]
                            if data['routing'][bus] and branch['added'] is not None:
                                contributors.append((branch['added'],None))
                            phase = coherent_energy(contributors,*edges,tick)
                            declared = window['coherent_energies'][k]
                            require((phase is None) == (declared is None), 'coherent support mask')
                            if phase is not None:
                                # Node energies come from the same replicated control gain as the
                                # incoherent check below, so they share its relative tolerance.
                                coherent_error = max(coherent_error, abs(phase-declared)/max(abs(declared), 1e-300))
                                require(abs(phase-declared) <= max(1e-11, abs(declared)*5e-7), 'coherent carrier-product mean')
                            require(abs(window['incoherent_energies'][k]-expected[0])<1e-14, 'incoherent control changed')
                            expected.insert(0,declared if phase is not None else expected[0])
                        numerical_error = max(numerical_error, abs(predicted-expected[0]))
                        require(abs(predicted-expected[0]) <= max(1e-14, abs(expected[0])*5e-7), 'body energy point')
                        for m in range(model_count):
                            control_means[m] += expected[m]*(edges[1]-edges[0])/width
                        points += 1
                    require(abs(window['mean']-control_means[0]) < 1e-14, 'quadrature mean')
                    start = decision-now
                    energies = [float(np.mean(x[start:start+width]**2)) for x in [wave, local_default[bus]]]
                    paired = actual['buses'][bus]['paired'][index]
                    require(max(abs(energies[0]-paired['mean_square']), abs(energies[1]-paired['default_mean_square'])) < 1e-14,
                            'actual physical-window target')
                    bus_means.append(control_means)
                    bus_actual.append(energies[0])
                branch_means.append(bus_means)
                branch_actual.append(bus_actual)
            means[branch['name']] = np.array(branch_means)
            teachers[branch['name']] = np.array(branch_actual)
        default = means['onset_now-0']
        truth_default = teachers['onset_now-0']
        records, ranks = [], []
        for bus in range(2):
            for index, width in enumerate(registration['window_samples']):
                loss, delta_loss = np.zeros(model_count), np.zeros(model_count)
                for branch in data['predictions']:
                    key = branch['name']
                    predictions = means[key][bus, index]
                    truth = teachers[key][bus, index]
                    window = branch['buses'][bus][index]
                    require(abs(window['default_mean']-default[bus, index, 0]) < 1e-14, 'paired default prediction')
                    require(abs(window['difference']-(predictions[0]-default[bus, index, 0])) < 1e-14, 'paired prediction difference')
                    loss += (predictions-truth)**2
                    delta_loss += ((predictions-default[bus, index])-(truth-truth_default[bus, index]))**2
                records.append(dict(bus=bus, samples=width, branches=len(means),
                                    squared_error_sum=loss.tolist(), paired_squared_error_sum=delta_loss.tolist()))
                for model in range(model_count):
                    counts = dict(pairs=0, reversed=0, invented_difference=0, missed_difference=0, agreement=0)
                    for a, b in itertools.combinations(means, 2):
                        truth = np.sign(teachers[a][bus, index]-teachers[b][bus, index])
                        pred = np.sign(means[a][bus, index, model]-means[b][bus, index, model])
                        counts['pairs'] += 1
                        field = ('agreement' if truth == pred else 'invented_difference' if truth == 0 else
                                 'missed_difference' if pred == 0 else 'reversed')
                        counts[field] += 1
                    ranks.append(dict(bus=bus, samples=width, model=model, **counts))
        results.append(dict(case=name, status='verified', branches=len(means), verified_points=points,
                            maximum_arithmetic_error=numerical_error, maximum_coherent_relative_error=coherent_error, errors=records, ranks=ranks))
    return dict(schema='i10-seven-class-energy-verification-v1', models=(['coherent_or_incoherent'] if coherent else [])+['actual_controls', 'without_inner_controls', 'without_retained_tones'],
                cases=results, limits='Actual-window energy fidelity and local-default energy ranks only; not contextual/ordinal ranks, later real-policy prediction, shared-profile transfer acceptance or live scheduling.',
                prediction_sha256={str(p.relative_to(root)): sha(p) for p in sorted(root.glob('*/predictions.json'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('inputs', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.inputs)
    (args.root / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'prediction_sha256'}, indent=2))
