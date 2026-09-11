"""Causal raw descriptors and fixed-payload temporal moment compression.

Inputs are acquired, assigned group energies, not source identities. This offline
reference does not implement NSGT, grouping inference, beam ownership or matching.
Python object/temporary costs are separate from the asserted binary payloads.
"""

import hashlib
import json
import math
import struct


COORDINATES = ('centroid_log2', 'spread_log2', 'rms_log2', 'rise_log2',
               'decline_log2', 'flux_log2', 'energy_share', 'mass_low',
               'mass_center', 'mass_high')
KNOT_BYTES = 320


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


class DescriptorKnot:
    """Exactly 320 payload bytes; masks and coverage derive from stored weights.

0:240 contains ten W/mean/S f64 triples; 240:256 representative time/weight;
256:304 six f64 support-start/end, raw-start/end, availability, missing seconds;
304:320 epoch/generation u64. No per-knot arrays are stored outside this payload.
"""

    __slots__ = ('data',)
    _snapshot_format = struct.Struct('<38d2Q')

    def __init__(self, data=None):
        self.data = bytearray(KNOT_BYTES) if data is None else data
        if len(self.data) != KNOT_BYTES:
            raise ValueError('descriptor knot payload must be 320 bytes')

    def number(self, index):
        return struct.unpack_from('<d', self.data, 8 * index)[0]

    def set_number(self, index, value):
        struct.pack_into('<d', self.data, 8 * index, value)

    @classmethod
    def from_raw(cls, raw, start=None, end=None):
        start = raw['start'] if start is None else start
        end = raw['end'] if end is None else end
        if (not raw['start'] <= start < end <= raw['end']
                or len(raw['values']) != 10
                or any(v is not None and not math.isfinite(v) for v in raw['values'])
                or not raw['raw_support_start'] <= raw['start'] < raw['end']
                <= raw['raw_support_end'] <= raw['available_end']
                or raw['time'] != raw['end']):
            raise ValueError('nonempty clipped support and finite raw descriptor required')
        for key in ('epoch', 'generation'):
            if not isinstance(raw[key], int) or not 0 <= raw[key] < 2**64:
                raise ValueError('stable unsigned epoch and generation handles required')
        knot = cls()
        duration = end - start
        if 'known_sample_intervals' in raw:
            observed = math.fsum(max(0., min(end, b/raw['sample_rate'])
                                    - max(start, a/raw['sample_rate']))
                                 for a, b in raw['known_sample_intervals'])
        else:
            observed = duration if raw['observed'] and not raw['gap'] else 0.
        for j, value in enumerate(raw['values']):
            if value is not None and observed:
                struct.pack_into('<ddd', knot.data, 24*j, observed, value, 0.)
        struct.pack_into('<8dQQ', knot.data, 240,
                         raw['time'] if observed else (start+end)/2, observed,
                         start, end, raw['raw_support_start'], raw['raw_support_end'],
                         raw['available_end'], max(0., duration-observed) if raw['gap'] else 0.,
                         raw['epoch'], raw['generation'])
        return knot

    def merge(self, other):
        if (self.number(33) != other.number(32)
                or self.data[304:320] != other.data[304:320]):
            raise ValueError('contiguous blocks in the same epoch/generation required')
        for j in range(10):
            wa, ma, sa = struct.unpack_from('<ddd', self.data, 24*j)
            wb, mb, sb = struct.unpack_from('<ddd', other.data, 24*j)
            weight = wa + wb
            if wa and wb:
                delta = mb - ma
                mean = ma + (wb / weight) * delta
                error = sa + sb + (wa / weight) * wb * delta * delta
            elif wb:
                mean, error = mb, sb
            else:
                mean, error = ma, sa
            struct.pack_into('<ddd', self.data, 24*j, weight, mean, error)
        a, b = self.number(31), other.number(31)
        endpoint = other.number(33)
        if a and b:
            time = self.number(30) + b/(a+b)*(other.number(30)-self.number(30))
        elif a or b:
            time = self.number(30) if a else other.number(30)
        else:
            time = (self.number(32)+endpoint)/2
        self.set_number(30, time)
        self.set_number(31, a+b)
        self.set_number(33, endpoint)
        self.set_number(34, min(self.number(34), other.number(34)))
        self.set_number(35, max(self.number(35), other.number(35)))
        self.set_number(36, max(self.number(36), other.number(36)))
        self.set_number(37, self.number(37)+other.number(37))

    def snapshot(self):
        values = self._snapshot_format.unpack_from(self.data)
        duration = values[33]-values[32]
        coverage = [values[3*j]/duration for j in range(10)]
        return {'start': values[32], 'end': values[33],
                'time': values[30], 'observed_sec': values[31],
                'raw_support_start': values[34], 'raw_support_end': values[35],
                'available_end': values[36], 'gap_sec': values[37],
                'gap': values[37] > 0,
                'gap_location_lost': 0 < values[37] < duration,
                'coverage': coverage,
                'values': [values[3*j+1] if coverage[j] >= .9 else None for j in range(10)],
                'moments': [values[3*j:3*j+3] for j in range(10)],
                'epoch': values[38], 'generation': values[39]}


class PackedKnots:
    """Owned immutable moments, with read-only Python views decoded on access.

This sequence is one source for both the numerical oracle and native matching;
it never caches separately mutable value dictionaries.
"""

    __slots__ = ('_data',)

    def __init__(self, data):
        if type(data) is not bytes or len(data) % KNOT_BYTES:
            raise ValueError('immutable complete 320-byte descriptor records required')
        self._data = data

    def __len__(self):
        return len(self._data)//KNOT_BYTES

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return DescriptorKnot(self._data[index*KNOT_BYTES:(index+1)*KNOT_BYTES]).snapshot()


class BoundedDescriptor:
    """Fixed knot bank and one insertion scratch; exports are detached copies."""

    def __init__(self, standard_deviations, capacity=128):
        if (len(standard_deviations) != 10
                or not all(math.isfinite(x) and x >= 0 for x in standard_deviations)
                or not isinstance(capacity, int) or capacity < 3):
            raise ValueError('ten frozen global scales and capacity >= 3 required')
        self.scales = tuple(max(x, 1e-6) for x in standard_deviations)
        self.capacity, self.count = capacity, 0
        self.storage = bytearray(capacity * KNOT_BYTES)
        self.scratch = DescriptorKnot()
        self.reconstruction_error = 0.
        self.merges = self.priority_coordinate_evaluations = 0
        self.frozen = False

    def knot(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        # Views are transient; no per-record Python objects are retained.
        return DescriptorKnot(memoryview(self.storage)[index*KNOT_BYTES:(index+1)*KNOT_BYTES])

    def append(self, knot):
        if self.frozen:
            raise ValueError('committed descriptors are immutable')
        if self.count:
            last = self.knot(self.count-1)
            if (last.number(33) != knot.number(32)
                    or last.data[304:320] != knot.data[304:320]):
                raise ValueError('contiguous generation-local descriptor blocks required')
        self.scratch.data[:] = knot.data
        self.reconstruction_error += math.fsum(knot.number(3*j+2)/(sd*sd)
                                               for j, sd in enumerate(self.scales))
        if self.count == self.capacity:
            best, cost = 1, math.inf
            for i in range(1, self.count-1):
                a, b = self.knot(i), self.knot(i+1)
                increment = 0.
                for j, sd in enumerate(self.scales):
                    wa, wb = a.number(3*j), b.number(3*j)
                    if wa and wb:
                        delta = (a.number(3*j+1)-b.number(3*j+1))/sd
                        increment += (wa/(wa+wb))*wb*delta*delta
                self.priority_coordinate_evaluations += 10
                if increment < cost:
                    best, cost = i, increment
            self.knot(best).merge(self.knot(best+1))
            for i in range(best+1, self.count-1):
                self.knot(i).data[:] = self.knot(i+1).data
            self.reconstruction_error += cost
            self.merges += 1
        else:
            self.count += 1
        self.knot(self.count-1).data[:] = self.scratch.data

    def freeze(self, observed_end):
        if (not math.isfinite(observed_end)
                or any(self.knot(i).number(36) > observed_end for i in range(self.count))):
            raise ValueError('commit cut excludes required original evidence')
        self.frozen = True
        return bytes(memoryview(self.storage)[:self.count*KNOT_BYTES])

    def snapshot(self):
        return {'knots': [self.knot(i).snapshot() for i in range(self.count)],
                'reconstruction_error': self.reconstruction_error,
                'merges': self.merges, 'frozen': self.frozen,
                'priority_coordinate_evaluations': self.priority_coordinate_evaluations,
                'bank_payload_bytes': len(self.storage), 'scratch_payload_bytes': KNOT_BYTES}


class SpanDescriptor:
    """Consume ordered raw records once, pairing on the generation's sample grid.

Explicit contiguous gap reports are held as one block until observation resumes.
An inferred acquisition gap on resumption can cause at most three insertions.
One partial block and one gap block add 640 payload bytes beyond bank/scratch.
"""

    def __init__(self, epoch, generation, sample_rate, first_hop_start, hop_samples,
                 span_start, standard_deviations, capacity=128, cadence=2, span_end=None):
        if (not all(isinstance(x, int) for x in
                    (epoch, generation, sample_rate, first_hop_start, hop_samples, cadence))
                or min(epoch, generation, first_hop_start) < 0
                or sample_rate <= 0 or hop_samples <= 0 or cadence not in (1, 2, 4)
                or not math.isfinite(span_start) or span_start < first_hop_start/sample_rate
                or span_end is not None and (not math.isfinite(span_end) or span_end <= span_start)):
            raise ValueError('generation clock, physical span and 1/2/4-hop cadence required')
        self.epoch, self.generation, self.rate = epoch, generation, sample_rate
        self.origin, self.hop_samples, self.cadence = first_hop_start, hop_samples, cadence
        self.start, self.end, self.cursor = span_start, span_end, span_start
        self.bank = BoundedDescriptor(standard_deviations, capacity)
        self.pending, self.gap_pending = DescriptorKnot(), DescriptorKnot()
        self.has_pending = self.has_gap = False
        self.last_raw_end = None
        self.last_supported_audio_end = None
        self.last_digest = None
        self.last_cut = -math.inf
        self.insertions = self.max_insertions = self.last_insertions = 0

    def _flush(self, gap=False):
        if gap and self.has_gap:
            self.bank.append(self.gap_pending)
            self.has_gap = False
        elif not gap and self.has_pending:
            self.bank.append(self.pending)
            self.has_pending = False
        else:
            return
        self.insertions += 1

    def gap(self, end, available_end, observed_end):
        if (self.bank.frozen or not self.cursor < end <= available_end <= observed_end
                or observed_end < self.last_cut or not math.isfinite(observed_end)
                or self.end is not None and end > self.end):
            raise ValueError('ordered observed gap within the open span required')
        self._flush()
        raw = {'epoch': self.epoch, 'generation': self.generation,
               'start': self.cursor, 'end': end, 'time': end, 'values': [None]*10,
               'observed': False, 'gap': True, 'raw_support_start': self.cursor,
               'raw_support_end': end, 'available_end': available_end}
        block = DescriptorKnot.from_raw(raw)
        if self.has_gap:
            self.gap_pending.merge(block)
        else:
            self.gap_pending.data[:] = block.data
            self.has_gap = True
        self.cursor, self.last_cut = end, observed_end

    def push(self, raw, observed_end):
        if self.bank.frozen:
            raise ValueError('committed span cannot consume observations')
        if (raw['epoch'], raw['generation']) != (self.epoch, self.generation):
            raise ValueError('new generation needs a new descriptor, without raw inheritance')
        lo, hi = raw['sample_start'], raw['sample_end']
        if (raw['sample_rate'] != self.rate or hi-lo != self.hop_samples
                or lo < self.origin or (lo-self.origin) % self.hop_samples
                or raw['start'] != lo/self.rate or raw['end'] != hi/self.rate
                or raw['available_end'] > observed_end or observed_end < self.last_cut
                or not math.isfinite(observed_end)):
            raise ValueError('canonical eligible hop on a monotone observation clock required')
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).digest()
        if hi == self.last_raw_end and digest == self.last_digest:
            self.last_cut, self.last_insertions = observed_end, 0
            return False
        if self.last_raw_end is not None and lo < self.last_raw_end:
            raise ValueError('stale/conflicting raw delivery requires upstream reconciliation')
        start, end = max(raw['start'], self.start), min(raw['end'], self.end or raw['end'])
        if end <= start or end <= self.cursor:
            raise ValueError('raw record has no new support inside this span')
        if start < self.cursor:
            raise ValueError('raw support overlaps an already consumed observation or gap')
        block = DescriptorKnot.from_raw(raw, start, end)
        before = self.insertions
        if start > self.cursor:
            self.gap(start, raw['available_end'], observed_end)
        if not raw['observed']:
            self.gap(end, raw['available_end'], observed_end)
        else:
            if raw['gap']:
                self._flush()
            self._flush(gap=True)
            if self.has_pending:
                self.pending.merge(block)
            else:
                self.pending.data[:] = block.data
                self.has_pending = True
            self.cursor = end
            index = (lo-self.origin)//self.hop_samples
            if (index+1) % self.cadence == 0 or end == self.end or raw['gap']:
                self._flush()
        self.last_raw_end, self.last_digest, self.last_cut = hi, digest, observed_end
        if raw['observed'] and any(value is not None for value in raw['values']):
            self.last_supported_audio_end = max(raw['raw_support_end'],
                                                self.last_supported_audio_end or raw['raw_support_end'])
        self.last_insertions = self.insertions-before
        self.max_insertions = max(self.max_insertions, self.last_insertions)
        assert self.last_insertions <= 3
        return True

    def finish(self, observed_end):
        if (observed_end < self.last_cut or not math.isfinite(observed_end)
                or self.end is not None and self.cursor != self.end):
            raise ValueError('monotone cut and completed declared span required')
        self._flush()
        self._flush(gap=True)
        return self.bank.freeze(observed_end)

    def snapshot(self):
        result = self.bank.snapshot()
        result.update(pending=self.pending.snapshot() if self.has_pending else None,
                      pending_gap=self.gap_pending.snapshot() if self.has_gap else None,
                      max_insertions_per_hop=self.max_insertions,
                      pending_payload_bytes=2*KNOT_BYTES)
        return result
