"""Independent causal raw-feature arithmetic; no retained descriptor or search state."""

import math


def raw_descriptor(hop, log2_bins, observed_end, previous=None):
    """Extract ten observables, with complete original support at the given cut.

An unavailable predecessor masks differences only. The returned raw record is
shared by all retained spans; extraction must not be repeated for each path.
"""
    rate, lo, hi = (hop[k] for k in ('sample_rate', 'sample_start', 'sample_end'))
    if (not isinstance(rate, int) or rate <= 0 or not isinstance(lo, int)
            or not isinstance(hi, int) or not 0 <= lo < hi or not log2_bins):
        raise ValueError('canonical samples and ordered log2 frequency bins required')
    start, end = lo / rate, hi / rate
    raw_start = hop.get('raw_support_start', start)
    raw_end, available = hop['raw_support_end'], hop['available_end']
    if (not all(math.isfinite(x) for x in (raw_start, raw_end, available, observed_end))
            or not 0 <= raw_start <= start < end <= raw_end <= available):
        raise ValueError('ordered full source support and availability required')
    if available > observed_end:
        return None
    intervals = []
    for a, b in sorted(hop.get('known_sample_intervals', [(lo, hi)] if hop['observed'] else [])):
        if not isinstance(a, int) or not isinstance(b, int) or not lo <= a <= b <= hi:
            raise ValueError('known sample intervals must be inside the canonical hop')
        if a == b:
            continue
        if intervals and a <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]))
        else:
            intervals.append((a, b))
    acquired = sum(b-a for a, b in intervals)
    values = [None] * 10
    energy, spectrum, bus_energy = (hop[k] for k in ('energy', 'spectrum', 'bus_energy'))
    for value in (energy, bus_energy):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError('finite nonnegative energy required')
    if energy is not None and bus_energy is not None and energy > bus_energy + 1e-12:
        raise ValueError('assigned energy cannot exceed bus energy')
    if spectrum is not None and len(spectrum) != len(log2_bins):
        raise ValueError('spectrum aligned to the frequency grid required')
    known = acquired == hi-lo and hop['association_known']
    adjacent = (known and previous is not None
                and previous['sample_end'] == lo and previous['sample_rate'] == rate
                and previous['observed'] and previous['association_known']
                and all(previous[k] == hop[k] for k in
                        ('epoch', 'generation', 'association_handle', 'grid_id')))
    ps, pe = None, None
    if adjacent:
        lower, right = previous['sample_start'], previous['sample_end']
        left = lower
        acquired_previous = 0
        for a, b in sorted(previous.get('known_sample_intervals', [(left, right)])):
            if not isinstance(a, int) or not isinstance(b, int) or not lower <= a <= b <= right:
                raise ValueError('known predecessor samples must be inside their canonical hop')
            acquired_previous += max(0, b-max(a, left))
            left = max(left, b)
        adjacent = acquired_previous == previous['sample_end']-previous['sample_start']
    if adjacent:
        pstart = previous.get('raw_support_start', previous['sample_start'] / rate)
        pend, pavail = previous['raw_support_end'], previous['available_end']
        if (not all(math.isfinite(x) for x in (pstart, pend, pavail))
                or not 0 <= pstart <= previous['sample_start'] / rate
                < previous['sample_end'] / rate <= pend <= pavail):
            raise ValueError('ordered predecessor source evidence required')
        adjacent = pavail <= observed_end
        if adjacent:
            pe, ps = previous['energy'], previous['spectrum']
            if pe is not None and (not math.isfinite(pe) or pe < 0):
                raise ValueError('finite nonnegative predecessor energy required')
            if ps is not None and len(ps) != len(log2_bins):
                raise ValueError('aligned predecessor spectrum required')
    mass = weighted = previous_mass = flux = 0.
    # Validate and summarize both supplied spectra in the first bin pass.
    for j, frequency in enumerate(log2_bins):
        if not math.isfinite(frequency) or j and frequency <= log2_bins[j-1]:
            raise ValueError('finite strictly increasing log2 frequency bins required')
        current = spectrum[j] if spectrum is not None else None
        prior = ps[j] if ps is not None else None
        for value in (current, prior):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError('finite nonnegative spectral mass required')
        if current is not None:
            mass += current
            weighted += current * frequency
        if prior is not None:
            previous_mass += prior
        if current is not None and prior is not None:
            flux += max(0., .5*math.log2(max(current, 1e-12))
                           - .5*math.log2(max(prior, 1e-12)))
    if known:
        if energy is not None:
            values[2] = math.log2(max(math.sqrt(energy), 1e-6))
        if energy is not None and bus_energy is not None:
            values[6] = energy / bus_energy if bus_energy else 0.
        if mass > 0:
            center = weighted / mass
            spread, low, middle, high = 0., 0., 0., 0.
            # Second pass computes spread and all centered mass fractions.
            for x, w in zip(log2_bins, spectrum):
                delta = x - center
                spread += w / mass * delta * delta
                if delta < -.5:
                    low += w / mass
                elif delta > .5:
                    high += w / mass
                else:
                    middle += w / mass
            values[0:2] = [center, math.sqrt(spread)]
            values[7:10] = [low, middle, high]
    if adjacent:
        if pe is not None and values[2] is not None:
            delta = values[2] - math.log2(max(math.sqrt(pe), 1e-6))
            values[3:5] = [max(0., delta), max(0., -delta)]
        if (spectrum is not None and ps is not None and not (energy and mass == 0)
                and not (pe and previous_mass == 0)):
            values[5] = flux / len(log2_bins)
        if any(values[j] is not None for j in (3, 4, 5)):
            raw_start, raw_end = min(raw_start, pstart), max(raw_end, pend)
            available = max(available, pavail)
    return {'epoch': hop['epoch'], 'generation': hop['generation'],
            'sample_start': lo, 'sample_end': hi, 'sample_rate': rate,
            'start': start, 'end': end, 'time': end, 'values': values,
            'observed': acquired > 0, 'gap': acquired < hi-lo,
            'known_sample_intervals': intervals,
            'raw_support_start': raw_start, 'raw_support_end': raw_end,
            'available_end': available}
