"""Audit raw candidate heads against frozen PCM features and closed-form hazards.

The observed residual prefix is checked by Rust fixtures and normal-path
noninterference, not reconstructed from a decimated report. This script checks
its combination with independently reconstructed future comparisons. No fitted
rating, full alternative mixture, or actual-Voice fidelity is established here.
"""

import argparse
import hashlib
import json
import math
import struct
import tomllib
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    if actual is None or expected is None:
        assert actual is expected, (actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-12), (actual, expected)


def ordinal(values, config):
    if all(v is None for v in values):
        return None
    x = [1.] + [0. if v is None else (v-m)/max(s, 1e-6)
                 for v, m, s in zip(values, config['means'], config['deviations'])]
    x += [float(v is None) for v in values]
    score = sum(a*b for a, b in zip(x, config['coefficients']))
    cdf = [0.] + [1./(1.+math.exp(score-cut)) for cut in config['cutpoints']] + [1.]
    return [b-a for a, b in zip(cdf, cdf[1:])]


def noninterference(before_report, after_report, before_wav, after_wav):
    rows = [[json.loads(line) for line in path.open()] for path in (before_report, after_report)]
    observations = []
    for items in rows:
        observed = {}
        for row in items:
            if row['type'] != 'temporal_observation':
                continue
            value = row['observation'].copy()
            value.pop('action_profile_features')
            value.pop('action_profile_resources', None)
            value.pop('worker_resources', None)
            value.pop('delivery_delay_us')
            observed[value['bus'], value['frame_id'], value['state']] = value
        observations.append(observed)
    common = observations[0].keys() & observations[1].keys()
    for key in common:
        assert observations[0][key] == observations[1][key], key
    terminals = {key[0] for key in common if key[2] == 'finished'}
    assert terminals == {0, 1}, 'both terminal observations must match'
    counts = {}
    for kind in ('onset', 'body_default', 'body_descriptor', 'self_sound_outcome',
                 'self_sound_descriptor_prediction', 'population_step'):
        a, b = [[r for r in items if r['type'] == kind] for items in rows]
        assert a == b, kind
        counts[kind] = len(a)
    assert digest(before_wav) == digest(after_wav)
    return dict(common_observations=len(common), terminal_buses=sorted(terminals),
                existing_records=counts, wav_sha256=digest(after_wav),
                before_report_sha256=digest(before_report), after_report_sha256=digest(after_report))


def raw_mixtures(report):
    frozen = {}
    cells = set()
    counts = dict(cells=0, heads=0, prior_only=0, mixed_groups=0,
                  fully_future_windows=0, aligned_issue_heads=0)
    buses = set()
    for line in report.open():
        row = json.loads(line)
        if row.get('type') != 'temporal_observation':
            continue
        observation = row['observation']
        table = observation['action_profile_features']
        if table is None:
            continue
        issue = table['raw_head_issue']
        key = issue['bus'], issue['epoch'], issue['issued_at']
        assert issue['issued_at'] == table['issued_at']
        close(issue['observed_coverage'], table['issue_observed_coverage'])
        if key in frozen:
            assert frozen[key] == issue, 'issue weights changed inside cached table'
        frozen[key] = issue
        for cell in table['latest']+[table['latest_residual']]:
            if cell is None:
                continue
            identity = (*key, cell['group']['generation'], cell['prototype'], cell['class'],
                        cell['action_at'], cell['evaluation_at'])
            if identity in cells:
                continue
            cells.add(identity)
            counts['cells'] += 1
            buses.add(issue['bus'])
            result = cell['raw_unreweighted_heads']
            assert result is not None and result['issued_at'] == issue['issued_at']
            selected = next(g for g in issue['groups'] if g and g['group'] == cell['group'])
            close(result['selected_group_weight'], selected['acoustic_weight'])
            close(result['issue_known_mass'], selected['known_mass'])
            close(1.-cell['issue_phrase_unknown'], selected['known_mass'])
            counts['fully_future_windows'] += cell['window_start'] >= cell['issued_at']
            for head, name in enumerate(['closure', 'continuation']):
                totals = [0.]*5
                held = local = 0.
                for group in issue['groups']:
                    if group is None or group['group'] == cell['group']:
                        continue
                    categories = group['categories'][head]
                    if categories is None:
                        continue
                    weight = group['acoustic_weight']*group['known_mass']
                    held += weight
                    for i, value in enumerate(categories):
                        totals[i] += weight*value
                for path in cell['continuation_paths']:
                    if path is None:
                        continue
                    categories = cell['closure'] if head == 0 else path['categories']
                    if categories is None:
                        continue
                    weight = selected['acoustic_weight']*path['issue_mass']
                    local += weight
                    for i, value in enumerate(categories):
                        totals[i] += weight*value
                support = min(1., held+local)
                coverage = issue['observed_coverage']
                r = coverage*support
                expected = [coverage*v+(1.-r)*prior for v, prior in zip(totals, issue['priors'][head])]
                actual = result[name]
                close(actual['observed_coverage'], coverage)
                close(actual['held_supported_mass'], held)
                close(actual['selected_supported_mass'], local)
                close(actual['supported_mass'], support)
                close(actual['reported_support_mass'], r)
                for a, b in zip(actual['categories'], expected):
                    close(a, b)
                close(actual['expected_rating'], sum(i*p/4 for i, p in enumerate(expected)))
                counts['heads'] += 1
                counts['prior_only'] += support == 0.
                counts['mixed_groups'] += held > 0. and local > 0.
                ordinary = observation['phrase']
                if (ordinary and ordinary['end_sample'] == issue['issued_at'] and
                        cell['evaluation_at'] == issue['issued_at']):
                    for a, b in zip(actual['categories'], ordinary[name]['categories']):
                        close(a, b)
                    counts['aligned_issue_heads'] += 1
    assert counts['heads'] > 0 and buses == {0, 1}
    return dict(**counts, frozen_issue_snapshots=len(frozen), buses=sorted(buses),
                claim='raw unreweighted base-head mixture, not shared-context consequence posterior')


def future_accent_density(cell, header, data, base, config):
    issue, end, start = cell['issued_at'], cell['evaluation_at'], cell['window_start']
    density = cell.get('accent_density')
    if density is None:
        return None
    assert density['value'] == cell['continuation_context'][1]
    # These windows need no inaccessible observed detector history.
    if start < issue + 2048:
        return None
    slot = header['action_offsets'].index(cell['action_at']-issue)
    klass = header['classes'].index(cell['class'])
    trajectory = header['profiles'][cell['prototype']]['trajectories'][klass][slot]
    records = []
    means, deviations = config['accent_means'], config['accent_deviations']
    for frame in range(min(375, (end-issue)//512)):
        mask, reserved, *values = struct.unpack_from('<II11d', data, base+(trajectory*375+frame)*96)
        assert reserved == 0
        salience = None
        if frame > 0 and mask & (1 << 3) and mask & (1 << 5):
            salience = sum(max(0., (value-mean)/max(sd, 1e-6))
                           for value, mean, sd in zip([values[3], values[5]], means, deviations))/2.
        records.append((bool(mask & (1 << 10)), salience))
    samples, weight, peaks, latest = 0, 0., 0, None
    for right in range(3, len(records)):
        if not all(r[0] for r in records[right-3:right+1]):
            continue
        saliences = [r[1] for r in records[right-2:right+1]]
        if any(v is None for v in saliences):
            continue
        lo, hi = issue+(right-1)*512, issue+right*512
        samples += max(0, min(hi,end)-max(lo,start,issue))
        left, middle, following = saliences
        if start <= hi <= end and middle > 1. and middle > left and middle >= following:
            weight += min(1., middle-1.)
            peaks += 1
            latest = [lo,hi]
    assert density['observed_samples'] == 0 and density['observed_weight'] == 0.
    assert density['projected_samples'] == samples
    assert density['projected_accents'] == peaks
    assert density['latest_projected_interval'] == latest
    close(density['projected_weight'], weight)
    expected = weight*48000/samples if samples > 0 and samples >= .9*(end-start) else None
    close(density['value'].get('value'), expected)
    assert density['value']['origin'] == ('unsupported' if expected is None else 'projected')
    return dict(positive=expected is not None and expected > 0., known_zero=expected == 0.,
                unsupported=expected is None, accents=peaks)


def projected_arrival(cell, table, header, data, base, config):
    forecast = cell.get('arrival')
    context = cell['continuation_context'][0]
    issue, end = cell['issued_at'], cell['evaluation_at']
    if forecast is None:
        if end > issue:
            assert context['origin'] == 'unsupported'
        return None
    frozen = next(f for f in table['arrival_issues'] if f and f['group'] == cell['group'])
    engine = frozen['engine']
    assert forecast['group'] == frozen['group']
    assert forecast['issued_at'] == frozen['issued_at'] == issue
    assert forecast['evaluated_at'] == end
    assert frozen['sample_rate'] == 48000 and engine['cut'] == issue
    assert engine['last']['available_end'] <= issue
    assert engine['context']['source_end'] <= engine['context']['available'] <= issue
    last = engine['last']['event_end']
    original_unknown = engine['uncertain_since'] is not None
    assert forecast['original_reset_unknown'] == original_unknown
    assert forecast['original_last_accent'] == last
    count, candidate, unknown = 0, None, original_unknown
    frames = min(375, (end-issue)//512)
    if end > issue:
        # The first three receipts touch the deliberately unknown seam flux.
        unknown = True
        slot = header['action_offsets'].index(cell['action_at']-issue)
        klass = header['classes'].index(cell['class'])
        trajectory = header['profiles'][cell['prototype']]['trajectories'][klass][slot]
        rows = []
        for j in range(frames):
            mask, reserved, *values = struct.unpack_from('<II11d', data, base+(trajectory*375+j)*96)
            assert reserved == 0
            salience = None
            if j > 0 and mask & (1 << 3) and mask & (1 << 5):
                salience = sum(max(0., (v-m)/max(sd, 1e-6)) for v,m,sd in
                               zip([values[3], values[5]], config['accent_means'], config['accent_deviations']))/2.
            rows.append((bool(mask & (1 << 10)), salience))
        for right in range(3, frames):
            values = [r[1] for r in rows[right-2:right+1]]
            if not all(r[0] for r in rows[right-3:right+1]) or any(v is None for v in values):
                unknown = True
                continue
            left, middle, following = values
            if middle > 1. and middle > left and middle >= following:
                last = issue + right*512
                candidate = last
                count += 1
                unknown = False
    tail = issue+frames*512 != end
    assert forecast['incomplete_tail'] == tail
    assert forecast['candidate_last_accent'] == candidate
    assert forecast['candidate_accents'] == count
    assert forecast['reset_unknown'] == (original_unknown or unknown or tail)
    close(forecast['elapsed_seconds'], (end-last)/48000.)
    probability = forecast['probability'].get('value')
    if forecast['reset_unknown']:
        assert probability is None and forecast['evaluations'] == 0
    elif probability is not None:
        cfg = engine['config']
        assert cfg['model'] == 'hazard' and cfg['coefficients'] == [0.]*18, 'constant-hazard assay required'
        close(probability, -math.expm1(-math.log(2.)*cfg['horizon_sec']))
    assert forecast['probability']['origin'] == ('unsupported' if probability is None else
                                                'observed' if end == issue else 'projected')
    assert forecast['horizon_end'] == end+math.ceil(engine['config']['horizon_sec']*48000)
    if end > issue:
        expected = probability if forecast['horizon_end']-end == 48000 else None
        close(context.get('value'), expected)
        assert context['origin'] == ('unsupported' if expected is None else 'projected')
    return dict(point=probability is not None, future_point=end > issue and probability is not None,
                candidate_accents=count, masked=probability is None, partial_tail=tail)


def verify(report, config_path, model):
    config = tomllib.loads(config_path.read_text())['temporal_phrase']
    assert config['hazard'] == [0., 1.] + [0.]*24, 'closed-form assay coefficients required'
    data = model.read_bytes()
    model_hash = hashlib.sha256(data).hexdigest()
    assert data[:8] == b'I10AP001'
    header_size, trajectories = struct.unpack_from('<II', data, 8)
    header = json.loads(data[16:16+header_size])
    base = 16+header_size
    assert len(data) == base + trajectories*375*96
    cells = {}
    duplicates = projected = projected_samples = closure_count = continuation_count = 0
    buses = set()
    varying_closure = set()
    survival_values = set()
    maximum_hazard_error = 0.
    arrival_checks = dict(cells=0, point=0, future_point=0, candidate_accents=0, masked=0, partial_tail=0)
    accent_checks = dict(windows=0, positive=0, known_zero=0, unsupported=0, accents=0)
    accent_config = tomllib.loads(config_path.read_text())['temporal_acoustic']
    for line in report.open():
        row = json.loads(line)
        if row.get('type') != 'temporal_observation':
            continue
        observation = row['observation']
        assert observation['action_enabled'] is False
        table = observation['action_profile_features']
        if table is None:
            continue
        assert bytes(table['profile_sha256']).hex() == model_hash
        for cell in table['latest'] + [table['latest_residual']]:
            if cell is None:
                continue
            group = cell['group']
            key = (group['bus'], group['epoch'], group['generation'], cell['prototype'],
                   cell['class'], cell['issued_at'], cell['action_at'], cell['evaluation_at'])
            if key in cells:
                assert cells[key] == cell, 'same frozen cell changed'
                duplicates += 1
                continue
            cells[key] = cell
            buses.add(group['bus'])
            issue, end = cell['issued_at'], cell['evaluation_at']
            accent_result = future_accent_density(cell, header, data, base, accent_config)
            if accent_result is not None:
                accent_checks['windows'] += 1
                for name, value in accent_result.items():
                    accent_checks[name] += value
            residual = cell['residual']
            reference = residual['future_reference']
            if reference is not None:
                assert reference['group'] == group
                assert (reference['source_start'] <= reference['source_end'] <=
                        reference['available'] <= reference['issued_at'] <= issue)
                slot = header['action_offsets'].index(cell['action_at']-issue)
                klass = header['classes'].index(cell['class'])
                trajectory = header['profiles'][cell['prototype']]['trajectories'][klass][slot]
                assert trajectory is not None
                total = 0.
                count = 0
                for frame in range(375):
                    lo = max(cell['window_start'], issue+512*frame, reference['expected_start'])
                    hi = min(end, issue+512*(frame+1), reference['expected_end'])
                    if hi <= lo:
                        continue
                    mask, reserved, *values = struct.unpack_from('<II11d', data, base+(trajectory*375+frame)*96)
                    assert reserved == 0
                    values = [v if mask & (1 << i) else None for i, v in enumerate(values)]
                    energy = values.pop()
                    background = cell['held_background_energy']
                    values[6] = None if energy is None or background is None else (
                        energy/(energy+background) if energy+background else 0.)
                    if frame == 0:
                        values[5] = None
                        previous = cell['issue_log_rms']
                        delta = None if values[2] is None or previous is None else values[2]-previous
                        values[3] = None if delta is None else max(0., delta)
                        values[4] = None if delta is None else max(0., -delta)
                    squares = [((v-target)/scale)**2 for v, target, scale in
                               zip(values, reference['values'], reference['scales'])
                               if v is not None and target is not None]
                    if squares:
                        total += sum(squares)/len(squares)*(hi-lo)
                        count += hi-lo
                assert count == residual['projected_samples'] and count > 0
                close(residual['projected_mean'], total/count)
                projected += 1
                projected_samples += count
            else:
                assert residual['projected_samples'] == 0
                assert residual['projected_mean'] is None
            count = residual['observed_samples']+residual['projected_samples']
            value = residual['value'].get('value')
            expected = None if count == 0 else sum(
                (residual[f'{origin}_mean'] or 0.)*residual[f'{origin}_samples']
                for origin in ('observed', 'projected'))/count
            close(value, expected)
            origin = 'unsupported' if count == 0 else 'projected' if reference else 'observed'
            assert residual['value']['origin'] == origin
            inputs = [value]+[None]*13
            result = ordinal(inputs, config['closure'])
            if result is None:
                assert cell['closure'] is None
            else:
                for actual, expected in zip(cell['closure'], result):
                    close(actual, expected)
                closure_count += 1
                varying_closure.add(tuple(cell['closure']))
            context = cell['continuation_context']
            arrival = projected_arrival(cell, table, header, data, base, accent_config)
            if arrival is not None:
                arrival_checks['cells'] += 1
                for key, value in arrival.items():
                    arrival_checks[key] += value
            inputs[10], inputs[12], inputs[13] = [v.get('value') for v in context]
            paths = [p for p in cell['continuation_paths'] if p is not None]
            close(sum(p['issue_mass'] for p in paths)+cell['issue_phrase_unknown'], 1.)
            for path in paths:
                start = path['foreground_start']
                survival = path['survival'].get('value')
                if start is None:
                    close(survival, 1.)
                elif cell['articulation'] is None:
                    assert survival is None
                else:
                    assert start <= issue
                    elapsed = (end-start)/48000.
                    integral = lambda t: (2.+t)*math.log(2.+t)-(2.+t)
                    expected = integral(elapsed+1.)-integral(elapsed)
                    error = abs(-math.log(survival)-expected)
                    # Use the existing hazard integrator's certified error contract.
                    assert error <= 1e-9+1e-7*expected, error
                    maximum_hazard_error = max(maximum_hazard_error, error)
                    survival_values.add(round(survival, 10))
                inputs[11] = survival
                expected = ordinal(inputs, config['continuation'])
                if expected is None:
                    assert path['categories'] is None
                else:
                    for actual, probability in zip(path['categories'], expected):
                        close(actual, probability)
                    continuation_count += 1
    assert buses == {0, 1} and projected > 0
    assert len(varying_closure) > 1 and len(survival_values) > 1
    return dict(report_sha256=digest(report), config_sha256=digest(config_path),
                model_sha256=digest(model), unique_cells=len(cells), duplicate_cells=duplicates,
                future_residual_cells=projected, future_compared_samples=projected_samples,
                closure_distributions=closure_count, continuation_distributions=continuation_count,
                distinct_closure_distributions=len(varying_closure),
                distinct_survival_values=len(survival_values), buses=sorted(buses),
                maximum_hazard_absolute_error=maximum_hazard_error,
                hazard_tolerance='1e-9 + 1e-7 * integrated hazard (existing numerical contract)',
                projected_accent_density=accent_checks,
                projected_arrival=arrival_checks,
                observed_residual_prefix='reported mean; not independently reconstructed here',
                claim='raw arithmetic and frozen support; no calibration or full mixture acceptance')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before-report', type=Path)
    parser.add_argument('--before-wav', type=Path)
    parser.add_argument('--after-wav', type=Path)
    parser.add_argument('--raw-mixtures', action='store_true')
    args = parser.parse_args()
    result = verify(args.report, args.config, args.model)
    if args.raw_mixtures:
        result['raw_mixtures'] = raw_mixtures(args.report)
    if any((args.before_report, args.before_wav, args.after_wav)):
        assert all((args.before_report, args.before_wav, args.after_wav))
        result['noninterference'] = noninterference(
            args.before_report, args.report, args.before_wav, args.after_wav)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
