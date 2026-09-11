"""Causal descriptor queries, bounded coarse anchors and subsequence matching.

Only centroid transposition and affine time scaling are transformation operators.
Other acoustic coordinates remain residuals. These are unfitted numerical
references, not inferred episode assignments, perceptual evidence or runtime code.
"""

from array import array
import math

from temporal_descriptor_reference import BoundedDescriptor


def descriptor_query(span, observed_end, query_id, occurrence_id, support_id):
    """Close a copy of the current prefix without changing live cadence or state."""
    if (not math.isfinite(observed_end) or observed_end < span.last_cut or observed_end < span.cursor
            or any(not isinstance(x, int) or x < 0 for x in (query_id, occurrence_id, support_id))):
        raise ValueError('current causal cut and stable nonnegative query identities required')
    bank = BoundedDescriptor(span.bank.scales, span.bank.capacity)
    bank.storage[:] = span.bank.storage
    bank.count, bank.reconstruction_error = span.bank.count, span.bank.reconstruction_error
    bank.merges = span.bank.merges
    bank.priority_coordinate_evaluations = span.bank.priority_coordinate_evaluations
    for active, block in ((span.has_pending, span.pending), (span.has_gap, span.gap_pending)):
        if active:
            bank.append(block)
    packed = bank.freeze(observed_end)
    result = bank.snapshot()
    result.update(epoch=span.epoch, generation=span.generation, query_id=query_id,
                  occurrence_id=occurrence_id, support_id=support_id,
                  start=span.start, query_end=span.cursor,
                  supporting_audio_end=span.last_supported_audio_end,
                  available_end=max((k['available_end'] for k in result['knots']), default=span.start),
                  issued_at=observed_end, complete=True, superseded=False,
                  scales=list(bank.scales), packed_knots=packed,
                  copy_payload_bytes=(bank.capacity+1)*320,
                  scope='Detached numerical query; allocation/exports must be charged to O04.')
    return result


def committed_episode(span, write, episode_id, generation):
    """Freeze a heard span using its sealed occurrence and original support time."""
    if (any(not isinstance(x, int) or x < 0 for x in (episode_id, generation))
            or write['epoch'] != span.epoch or write['start'] != span.start
            or write['support_end'] != span.cursor or write['support']['available_support'] <= 0
            or not span.cursor <= write['committed_at'] <= write['delivered_at']):
        raise ValueError('positive sealed occurrence with matching original span required')
    snapshot = descriptor_query(span, write['committed_at'], write['sequence'],
                                write['occurrence_id'], write['support_id'])
    span.finish(write['committed_at'])
    return {'knots': snapshot['knots'], 'scales': snapshot['scales'],
            'episode_id': episode_id, 'generation': generation, 'epoch': span.epoch,
            'source_generation': span.generation, 'occurrence_id': write['occurrence_id'],
            'support_id': write['support_id'], 'first_observed_end': write['support_end'],
            'committed_at': write['committed_at'], 'available_end': write['delivered_at'],
            'committed': True, 'compression_error': snapshot['reconstruction_error']}


def _validate(knots, observed_end, max_knots):
    if not math.isfinite(observed_end) or len(knots) > max_knots:
        raise ValueError('finite observation cut and bounded descriptor lengths required')
    previous = None
    for knot in knots:
        if (len(knot['values']) != 10
                or any(x is not None and not math.isfinite(x) for x in knot['values'])
                or not all(math.isfinite(knot[k]) for k in
                           ('start', 'end', 'time', 'observed_sec', 'raw_support_start',
                            'raw_support_end', 'available_end'))
                or not knot['raw_support_start'] <= knot['start'] < knot['end']
                <= knot['raw_support_end'] <= knot['available_end'] <= observed_end
                or not 0 <= knot['observed_sec'] <= knot['end']-knot['start']+1e-12
                or knot['time'] > knot['raw_support_end']
                or previous and (knot['start'] != previous['end']
                                 or knot['time'] <= previous['time']
                                 or knot['epoch'] != previous['epoch']
                                 or knot['generation'] != previous['generation'])):
            raise ValueError('ordered, same-generation, causally available ten-coordinate knots required')
        previous = knot


def _median(values):
    # Bounded insertion sort exposes the declared comparison work.
    ordered, comparisons = [], 0
    for value in values:
        position = len(ordered)
        while position:
            comparisons += 1
            if ordered[position-1] <= value:
                break
            position -= 1
        ordered.insert(position, value)
    n = len(ordered)
    value = None if not n else ordered[n//2] if n % 2 else (ordered[n//2-1]+ordered[n//2])/2
    return value, comparisons


def _round_lower(value, step):
    lower = math.floor(value/step)
    return (lower if value/step-lower <= .5 else lower+1)*step


def _residual(cue, reference, shift, scales):
    total, count = 0., 0
    for j, sd in enumerate(scales):
        a, b = cue['values'][j], reference['values'][j]
        if a is not None and b is not None:
            delta = (a-b-(shift if j == 0 else 0.))/sd
            total += delta*delta
            count += 1
    return total, count


def anchor_transform(cue, reference, anchor, bounds=(2., 2.), grid=1/16):
    """Median pitch difference and supported original interval ratio, up to8/7."""
    if (not isinstance(anchor, int) or not 0 <= anchor < len(reference)
            or len(bounds) != 2 or not all(math.isfinite(x) and x > 0 for x in (*bounds, grid))):
        raise ValueError('existing anchor and positive finite log2 bounds/grid required')
    n = min(8, len(cue), len(reference)-anchor)
    pitches, tempos, previous_timing = [], [], False
    for i in range(n):
        a, b = cue[i], reference[anchor+i]
        if a['values'][0] is not None and b['values'][0] is not None:
            pitches.append(a['values'][0]-b['values'][0])
        timing = all(k['observed_sec'] > 0 and not k['gap']
                     and any(v is not None for v in k['values']) for k in (a, b))
        if i:
            ca, rb = cue[i-1], reference[anchor+i-1]
            if previous_timing and timing:
                ci, ri = a['time']-ca['time'], b['time']-rb['time']
                if ci > 0 and ri > 0:
                    tempos.append(math.log2(ri/ci))
        previous_timing = timing
    pitch, pc = _median(pitches)
    tempo, tc = _median(tempos)
    estimates = [pitch, tempo]
    rounded = [_round_lower(x, grid) if x is not None else 0. for x in estimates]
    outside = any(x is not None and abs(x) > bound for x, bound in zip(estimates, bounds))
    boundary = outside or any(x is not None and (abs(x) == bound or abs(y) >= bound)
                              for x, y, bound in zip(estimates, rounded, bounds))
    return {'frequency_shift_log2': rounded[0] if pitch is not None else None,
            'tempo_shift_log2': rounded[1] if tempo is not None else None,
            'unrounded': estimates, 'applied': rounded,
            'pitch_samples': len(pitches), 'interval_samples': len(tempos),
            'pitch_estimate_residual_rms': math.sqrt(math.fsum((x-pitch)**2 for x in pitches)/len(pitches)) if pitches else None,
            'tempo_estimate_residual_rms': math.sqrt(math.fsum((x-tempo)**2 for x in tempos)/len(tempos)) if tempos else None,
            'median_comparisons': pc+tc, 'out_of_range': outside,
            'bound_hit': boundary, 'unidentified': [x is None for x in estimates],
            'anchor': anchor, 'grid_id': [_round_lower(x/grid, 1.) for x in rounded]}


def coarse_cost(cue, reference, anchor, transform, scales):
    if transform['out_of_range'] or transform['bound_hit']:
        return {'cost': None, 'valid_coordinates': 0, 'paired_steps': 0}
    total, count = 0., 0
    n = min(8, len(cue), len(reference)-anchor)
    for i in range(n):
        value, weight = _residual(cue[i], reference[anchor+i], transform['applied'][0], scales)
        total += value
        count += weight
    return {'cost': total/count if count else None, 'valid_coordinates': count, 'paired_steps': n}


def refine_anchor(cue, reference, coarse, scales, bounds=(2., 2.), grid=1/64):
    """At most four bracketing combinations, never the full transform grid."""
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError('positive finite refinement grid required')
    if coarse['out_of_range'] or coarse['bound_hit']:
        return []
    choices = []
    for value in coarse['unrounded']:
        if value is None:
            choices.append([0.])
        else:
            choices.append(sorted({math.floor(value/grid)*grid, math.ceil(value/grid)*grid}))
    output = []
    for pitch in choices[0]:
        for tempo in choices[1]:
            transform = dict(coarse)
            transform.update(applied=[pitch, tempo],
                             frequency_shift_log2=None if coarse['unidentified'][0] else pitch,
                             tempo_shift_log2=None if coarse['unidentified'][1] else tempo,
                             grid_id=[round(pitch/grid), round(tempo/grid)],
                             bound_hit=any(abs(x) >= b for x, b in zip((pitch, tempo), bounds)))
            score = coarse_cost(cue, reference, coarse['anchor'], transform, scales)
            if score['cost'] is not None:
                transform.update(coarse_cost=score['cost'])
                output.append(transform)
    return sorted(output, key=lambda t: (t['coarse_cost'], t['grid_id']))


def subsequence_dtw(cue, reference, transform, scales, observed_end,
                    band_radius=16, insertion_penalty=1., deletion_penalty=1., max_knots=128):
    """Two DP rows and bounded traceback; free outside-subsequence reference.

Diagonal pairs average common coordinate residuals. A fully missing pair costs
zero but carries no matched evidence. Insert/delete operations cost their explicit
penalties. Exact ties prefer diagonal, insertion, deletion, then earlier endpoint.
"""
    _validate(cue, observed_end, max_knots)
    _validate(reference, observed_end, max_knots)
    if (len(scales) != 10 or not all(math.isfinite(x) and x > 0 for x in scales)
            or not all(math.isfinite(x) and x > 0 for x in (insertion_penalty, deletion_penalty))
            or band_radius is not None and (not isinstance(band_radius, int) or band_radius < 0)):
        raise ValueError('ten positive scales, penalties and nonnegative band radius required')
    n, m = len(cue), len(reference)
    empty = {'status': 'unknown', 'supported': False, 'cost': None, 'path': [],
             'dp_cells': 0, 'band_edge_hit': False, 'missing_pair_steps': 0}
    if not n or not m or transform['out_of_range'] or transform['bound_hit']:
        return empty
    anchor = transform['anchor']
    if not 0 <= anchor < m:
        raise ValueError('transform anchor must belong to this reference')
    width = m if band_radius is None else min(m, 2*band_radius+1)
    previous = array('d', [0.]*(m+1))
    current = array('d', [math.inf]*(m+1))
    parents = bytearray(n*width)
    starts = array('H', [0]*n)
    ends = array('H', [0]*n)
    times = array('d', (k['time'] for k in reference))
    ratio = 2.**transform['applied'][1]
    edge, cells, time_comparisons = False, 0, 0
    for i, knot in enumerate(cue):
        if band_radius is None:
            lo, hi = 0, m
        else:
            predicted = reference[anchor]['time']+ratio*(knot['time']-cue[0]['time'])
            left, right = 0, m
            while left < right:
                middle = (left+right)//2
                time_comparisons += 1
                if times[middle] < predicted:
                    left = middle+1
                else:
                    right = middle
            center = min(m-1, left)
            if left and (left == m or predicted-times[left-1] <= times[left]-predicted):
                center = left-1
            lo, hi = max(0, center-band_radius), min(m, center+band_radius+1)
        starts[i], ends[i] = lo, hi
        for j in range(m+1):
            current[j] = math.inf
        if lo == 0:
            current[0] = (i+1)*insertion_penalty
        for j in range(lo, hi):
            residual, valid = _residual(knot, reference[j], transform['applied'][0], scales)
            step = residual/valid if valid else 0.
            choices = (previous[j]+step,
                       previous[j+1]+insertion_penalty,
                       current[j]+deletion_penalty)
            operation = min(range(3), key=lambda k: (choices[k], k))
            current[j+1] = choices[operation]
            parents[i*width+j-lo] = operation+1
            cells += 1
        previous, current = current, previous
    j = min(range(1, m+1), key=lambda k: (previous[k], k))
    total = previous[j]
    if not math.isfinite(total):
        return {**empty, 'dp_cells': cells, 'band_disconnected': True,
                'band_ranges': list(zip(starts, ends)), 'band_time_comparisons': time_comparisons}
    path, i, inserted, deleted, missing, matched, valid_coordinates = [], n, 0, 0, 0, 0, 0
    coordinate_error, coordinate_count = [0.]*10, [0]*10
    while i:
        if j == 0:
            path.append(('insert', i-1, None))
            inserted += 1
            i -= 1
            continue
        lo, hi = starts[i-1], ends[i-1]
        if not lo <= j-1 < hi:
            raise AssertionError('traceback escaped the recorded DP band')
        if band_radius is not None and ((j-1 == lo and lo > 0) or (j == hi and hi < m)):
            edge = True
        operation = parents[(i-1)*width+j-1-lo]
        if operation == 1:
            path.append(('pair', i-1, j-1))
            supported = 0
            for k, sd in enumerate(scales):
                a, b = cue[i-1]['values'][k], reference[j-1]['values'][k]
                if a is not None and b is not None:
                    delta = (a-b-(transform['applied'][0] if k == 0 else 0.))/sd
                    coordinate_error[k] += delta*delta
                    coordinate_count[k] += 1
                    supported += 1
            valid_coordinates += supported
            matched += bool(supported)
            missing += not supported
            i, j = i-1, j-1
        elif operation == 2:
            path.append(('insert', i-1, j-1))
            inserted += 1
            i -= 1
        elif operation == 3:
            path.append(('delete', None, j-1))
            deleted += 1
            j -= 1
        else:
            raise AssertionError('unreachable traceback state')
    path.reverse()
    motion_error, interval_error = [], []
    for left, right in zip(path, path[1:]):
        if (left[0] != 'pair' or right[0] != 'pair'
                or right[1] != left[1]+1 or right[2] != left[2]+1):
            continue
        a, b, c, d = cue[left[1]], cue[right[1]], reference[left[2]], reference[right[2]]
        if any(k['gap'] for k in (a, b, c, d)):
            continue
        pitches = [k['values'][0] for k in (a, b, c, d)]
        if all(x is not None for x in pitches):
            motion_error.append(((pitches[1]-pitches[0])-(pitches[3]-pitches[2]))**2)
        if all(k['observed_sec'] > 0 and any(v is not None for v in k['values']) for k in (a, b, c, d)):
            interval_error.append((math.log2((d['time']-c['time'])/(b['time']-a['time']))
                                   - transform['applied'][1])**2)
    observed = sum(k['observed_sec'] > 0 for k in cue)
    supported = observed > 0 and valid_coordinates > 0
    return {'status': 'match' if supported else 'unknown', 'supported': supported,
            'cost': total/observed if supported else None, 'total_cost': total,
            'observed_cue_steps': observed, 'matched_cue_steps': matched,
            'matched_step_fraction': matched/observed if observed else 0.,
            'missing_pair_steps': missing, 'inserted_steps': inserted, 'deleted_steps': deleted,
            'coordinate_residuals': [value/count if count else None
                                     for value, count in zip(coordinate_error, coordinate_count)],
            'coordinate_counts': coordinate_count, 'path': path,
            'relative_pitch_motion_rms': math.sqrt(math.fsum(motion_error)/len(motion_error)) if motion_error else None,
            'matched_interval_log2_residual_rms': math.sqrt(math.fsum(interval_error)/len(interval_error)) if interval_error else None,
            'relative_pitch_motion_pairs': len(motion_error), 'matched_interval_pairs': len(interval_error),
            'frequency_shift_log2': transform['frequency_shift_log2'],
            'tempo_shift_log2': transform['tempo_shift_log2'],
            'pitch_estimate_residual_rms': transform.get('pitch_estimate_residual_rms'),
            'tempo_estimate_residual_rms': transform.get('tempo_estimate_residual_rms'),
            'bound_hit': transform['bound_hit'], 'band_edge_hit': edge,
            'dp_cells': cells, 'dp_payload_bytes': (m+1)*16+n*width+n*4+m*8,
            'band_ranges': list(zip(starts, ends)), 'band_time_comparisons': time_comparisons,
            'field_visits': cells*20, 'traceback_coordinate_visits': len(path)*10,
            'reference_start': j, 'search_completed': True}


def match_query(query, episodes, observed_end, candidate_limit=16, anchor_spacing=4,
                anchor_limit=32, transform_limit=4, band_radius=16, max_episodes=256,
                bounds=(2., 2.), coarse_grid=1/16, fine_grid=1/64,
                insertion_penalty=1., deletion_penalty=1., max_knots=128):
    """All-episode coarse cache and bounded selected-episode full comparisons.

Stable identities survive ranking; no availability/focus/assignment is inferred.
An ambiguous candidate cutoff remains explicit epistemic uncertainty.
"""
    if (not query['complete'] or query['superseded'] or query['available_end'] > observed_end
            or query['query_end'] > observed_end
            or not all(isinstance(x, int) and x > 0 for x in
                       (candidate_limit, anchor_spacing, anchor_limit, transform_limit, max_episodes, max_knots))
            or len(episodes) > max_episodes):
        raise ValueError('current completed query and positive bounded search inventory required')
    cue, scales = query['knots'], query['scales']
    if len(scales) != 10 or not all(math.isfinite(x) and x > 0 for x in scales):
        raise ValueError('frozen positive ten-coordinate scales required')
    _validate(cue, observed_end, max_knots)
    cache, candidates, seen, anchor_evaluations, comparisons = [], [], set(), 0, 0
    for episode in episodes:
        identity, generation = episode['episode_id'], episode['generation']
        if identity in seen:
            raise ValueError('episode aliases require upstream canonicalization')
        seen.add(identity)
        if (episode['epoch'] != query['epoch'] or not episode['committed']
                or episode['available_end'] > observed_end
                or episode['first_observed_end'] >= query['query_end']):
            continue
        reference = episode['knots']
        _validate(reference, observed_end, max_knots)
        if episode['scales'] != scales:
            raise ValueError('changed development scales require descriptor reconstruction')
        best, bound_anchors = None, 0
        anchor_total = (len(reference)+anchor_spacing-1)//anchor_spacing
        for anchor in range(0, min(len(reference), anchor_limit*anchor_spacing), anchor_spacing):
            transform = anchor_transform(cue, reference, anchor, bounds, coarse_grid)
            score = coarse_cost(cue, reference, anchor, transform, scales)
            bound_anchors += transform['bound_hit']
            anchor_evaluations += 1
            comparisons += transform['median_comparisons']
            if score['cost'] is not None and (best is None or (score['cost'], anchor) < (best['cost'], best['anchor'])):
                best = {'cost': score['cost'], 'anchor': anchor, 'transform': transform,
                        'valid_coordinates': score['valid_coordinates']}
        cache.append({'episode_id': identity, 'episode_generation': generation,
                      'cost': None if best is None else best['cost'],
                      'similarity': None if best is None else math.exp(-best['cost']),
                      'anchor': None if best is None else best['anchor'],
                      'unidentified': [True, True] if best is None else best['transform']['unidentified'],
                      'valid_coordinates': 0 if best is None else best['valid_coordinates'],
                      'bound_anchors': bound_anchors,
                      'anchor_cap_loss': anchor_total > anchor_limit})
        if best is not None:
            candidates.append({**best, 'episode': episode})
    candidates.sort(key=lambda c: (c['cost'], c['episode']['episode_id']))
    selected, excluded = candidates[:candidate_limit], candidates[candidate_limit:]
    cutoff_tie = bool(excluded and selected and excluded[0]['cost'] == selected[-1]['cost'])
    matches, refinements, dp_cells = [], 0, 0
    for candidate in selected:
        episode = candidate['episode']
        transforms = refine_anchor(cue, episode['knots'], candidate['transform'], scales, bounds, fine_grid)
        refinements += max(1, math.prod(1 if x is None or x/fine_grid == math.floor(x/fine_grid) else 2
                                       for x in candidate['transform']['unrounded']))
        for transform in transforms[:transform_limit]:
            result = subsequence_dtw(cue, episode['knots'], transform, scales, observed_end,
                                     band_radius, insertion_penalty, deletion_penalty, max_knots)
            dp_cells += result['dp_cells']
            matches.append({**result, 'episode_id': episode['episode_id'],
                            'episode_generation': episode['generation'], 'anchor': candidate['anchor'],
                            'transform_grid_id': transform['grid_id'], 'cutoff_tie': cutoff_tie,
                            'ambiguous_cutoff': cutoff_tie,
                            'search_covered': not cutoff_tie and not result['band_edge_hit']
                            and (len(episode['knots'])+anchor_spacing-1)//anchor_spacing <= anchor_limit,
                            'search_nonempty': bool(cache)})
    matches.sort(key=lambda c: (c['cost'] if c['cost'] is not None else math.inf,
                               c['episode_id'], c['transform_grid_id']))
    return {'query_id': query['query_id'], 'occurrence_id': query['occurrence_id'],
            'support_id': query['support_id'], 'epoch': query['epoch'],
            'generation': query['generation'],
            'query_end': query['query_end'], 'supporting_audio_end': query['supporting_audio_end'],
            'available_end': query['available_end'], 'completed_at': observed_end,
            'complete': True, 'superseded': False, 'coarse_entries': cache, 'matches': matches,
            'excluded_episode_ids': [c['episode']['episode_id'] for c in excluded],
            'cutoff_tie': cutoff_tie, 'search_nonempty': bool(cache),
            'status': 'completed' if matches else 'unknown_unretrieved',
            'anchor_evaluations': anchor_evaluations, 'median_comparisons': comparisons,
            'refinement_evaluations': refinements, 'dp_cells': dp_cells,
            'limitations': ['single best anchor', 'density/tempo approximation',
                            'unfitted scales', 'no bank lifecycle or queue scheduling']}


def coarse_commit_snapshot(result, episode_handles, received_at):
    """Bind cache entries to existing opaque generation-qualified ledger handles.

Absent/retired bindings stay absent, so the ledger returns unknown interference.
This does not allocate handles, reinterpret assignments or refresh original time.
"""
    if (not math.isfinite(received_at) or not math.isfinite(result['completed_at'])
            or not result['available_end'] <= result['completed_at'] <= received_at):
        raise ValueError('actual receipt cannot precede source availability or worker completion')
    if (any(not isinstance(v, int) or v < 0 for v in episode_handles.values())
            or len(set(episode_handles.values())) != len(episode_handles)):
        raise ValueError('distinct stable episode-generation ledger handles required')
    entries = {}
    for entry in result['coarse_entries']:
        key = (entry['episode_id'], entry['episode_generation'])
        if key in episode_handles:
            entries[episode_handles[key]] = entry['cost']
    return {'epoch': result['epoch'], 'generation': result['generation'],
            'occurrence_id': result['occurrence_id'],
            'support_id': result['support_id'], 'query_id': result['query_id'],
            'support_end': result['query_end'], 'available_end': received_at,
            'supporting_audio_end': result['supporting_audio_end'],
            'completed': result['complete'], 'superseded': result['superseded'], 'entries': entries}
