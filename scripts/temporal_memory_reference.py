"""Bounded physical-gap clock and sealed-occurrence episode retention.

Caller-supplied handles must be globally unique episode-generation identities.
This metadata reference does not allocate descriptor identities, infer assignments
or manage the complete relation graph. Capacity eviction is explicit at its API.
"""

from array import array
import hashlib
import math
import struct

from temporal_cognition_reference import episode_log_availability, recognition_probability
from temporal_occurrence_reference import occurrence_interference


class GapClock:
    """One bus clock; bounded original sample masks plus cumulative missing time.

Old prefix queries return a conservative interval constrained by cumulative gap
mass. Losing mask history is computational uncertainty, not known silence.
"""

    def __init__(self, epoch, sample_rate=48000, hop_samples=512, capacity=128, origin_sample=0):
        if (any(type(x) is not int or x < 0 for x in (epoch, sample_rate, hop_samples, capacity, origin_sample))
                or min(sample_rate, hop_samples, capacity) == 0 or max(epoch, origin_sample) >= 2**64):
            raise ValueError('unsigned epoch/origin and positive sample clock/capacity required')
        self.epoch, self.rate, self.hop, self.capacity = epoch, sample_rate, hop_samples, capacity
        self.origin, self.end, self.missing = origin_sample, origin_sample, 0.
        self.stride = 32+(hop_samples+7)//8
        self.storage = bytearray(capacity*self.stride)
        self.head = self.count = self.evictions = 0
        self.last_digest = None
        self.cut = -math.inf

    def observe(self, raw, observed_end):
        lo, hi = raw['sample_start'], raw['sample_end']
        if (raw['epoch'] != self.epoch or raw['sample_rate'] != self.rate
                or type(lo) is not int or type(hi) is not int or lo < self.origin or hi >= 2**64
                or hi-lo != self.hop or (lo-self.origin) % self.hop
                or not math.isfinite(observed_end) or observed_end < self.cut
                or not math.isfinite(raw['available_end'])
                or not hi/self.rate <= raw['available_end'] <= observed_end):
            raise ValueError('ordered causal canonical bus acquisition record required')
        mask = bytearray(self.stride-32)
        intervals = raw['known_sample_intervals']
        if len(intervals) > self.hop:
            raise ValueError('sample-interval input exceeds its canonical hop')
        for a, b in intervals:
            if type(a) is not int or type(b) is not int or not lo <= a <= b <= hi:
                raise ValueError('known sample intervals must lie inside the hop')
            for index in range(a-lo, b-lo):
                mask[index//8] |= 1 << (index % 8)
        if bool(any(mask)) != bool(raw['observed']):
            raise ValueError('acquisition mask and sample intervals disagree')
        digest = hashlib.sha256(struct.pack('<QQd', lo, hi, raw['available_end'])+mask).digest()
        if lo < self.end:
            if hi == self.end and digest == self.last_digest:
                self.cut = observed_end
                return False
            raise ValueError('stale/overlapping acquisition cannot revise the sealed clock')
        self._append(lo, hi, mask, raw['available_end'])
        self.last_digest, self.cut = digest, observed_end
        return True

    def gap(self, sample_end, observed_end):
        if (type(sample_end) is not int or not self.end < sample_end < 2**64
                or not math.isfinite(observed_end) or observed_end < self.cut
                or sample_end/self.rate > observed_end):
            raise ValueError('explicit forward missing-sample notification required')
        self._append(self.end, sample_end, bytes(self.stride-32), observed_end)
        self.last_digest, self.cut = None, observed_end

    def _append(self, lo, hi, mask, available):
        before = self.missing+(lo-self.end)/self.rate
        known = sum(byte.bit_count() for byte in mask)
        at = self.head*self.stride
        struct.pack_into('<QQdd', self.storage, at, lo, hi, before, available)
        self.storage[at+32:at+self.stride] = mask
        self.head = (self.head+1) % self.capacity
        self.evictions += self.count == self.capacity
        self.count = min(self.count+1, self.capacity)
        self.end, self.missing = hi, before+(hi-lo-known)/self.rate

    def prefix(self, time):
        origin = self.origin/self.rate
        if not math.isfinite(time) or not origin <= time <= self.end/self.rate:
            raise ValueError('prefix must lie within the acquired/notified physical clock')
        if time == origin:
            return {'lower': 0., 'upper': 0., 'history_lost': False}
        for offset in range(self.count):
            slot = (self.head-self.count+offset) % self.capacity
            at = slot*self.stride
            lo, hi, before, _ = struct.unpack_from('<QQdd', self.storage, at)
            left, right = lo/self.rate, hi/self.rate
            if time < left:
                if offset == 0 and self.evictions:
                    return {'lower': max(0., before-(left-time)),
                            'upper': min(before, time-origin), 'history_lost': True}
                value = max(0., before-(left-time))
                return {'lower': value, 'upper': value, 'history_lost': False}
            if time <= right:
                mask = memoryview(self.storage)[at+32:at+self.stride]
                position = max(0., min(hi-lo, (time-left)*self.rate))
                full = math.floor(position)
                known = sum(bool(mask[i//8] & (1 << (i % 8))) for i in range(min(full, self.hop)))
                if full < self.hop and mask[full//8] & (1 << (full % 8)):
                    known += position-full
                value = max(0., before+(position-known)/self.rate)
                return {'lower': value, 'upper': value, 'history_lost': False}
        return {'lower': self.missing, 'upper': self.missing, 'history_lost': False}


class InterferenceWindow:
    """Exact positive observed increments in (time-1,time], with explicit overflow.

Only positive known increments occupy rows; unsupported coarse results contribute
no observed rate. Overflow loses rate history, never actual retention increments.
Slot birth sequences prevent expiration from subtracting an evicted generation.
"""

    def __init__(self, slots=256, capacity=144):
        if type(slots) is not int or type(capacity) is not int or min(slots, capacity) < 1:
            raise ValueError('positive fixed rate-window dimensions required')
        self.slots, self.capacity, self.stride = slots, capacity, 16+slots*8
        self.storage = bytearray(capacity*self.stride)
        self.totals = array('d', [0.]*slots)
        self.head = self.count = self.losses = 0
        self.lost_until = -math.inf
        self.last_time = -math.inf
        self.invalid = False
        self.envelope_unverified = False

    def _drop(self, births):
        at = (self.head-self.count) % self.capacity*self.stride
        _, sequence = struct.unpack_from('<dQ', self.storage, at)
        for slot, born in enumerate(births):
            if born and sequence >= born:
                self.totals[slot] = max(0., self.totals[slot]-struct.unpack_from('<d', self.storage, at+16+slot*8)[0])
        self.count -= 1

    def observe(self, time, sequence, increments, births, r_max):
        if (not math.isfinite(time) or time < self.last_time
                or type(sequence) is not int or not 0 < sequence < 2**64
                or len(increments) != self.slots or len(births) != self.slots
                or not all(math.isfinite(x) and x >= 0 for x in increments)
                or not math.isfinite(r_max) or r_max <= 0):
            raise ValueError('ordered original times and finite slot-aligned observed increments required')
        while self.count:
            at = (self.head-self.count) % self.capacity*self.stride
            if struct.unpack_from('<d', self.storage, at)[0] > time-1:
                break
            self._drop(births)
        if any(increments):
            if self.count == self.capacity:
                at = (self.head-self.count) % self.capacity*self.stride
                lost = struct.unpack_from('<d', self.storage, at)[0]
                self.lost_until = max(self.lost_until, lost+1)
                self._drop(births)
                self.losses += 1
                self.envelope_unverified = True
            at = self.head*self.stride
            struct.pack_into('<dQ', self.storage, at, time, sequence)
            for slot, value in enumerate(increments):
                struct.pack_into('<d', self.storage, at+16+slot*8, value)
                self.totals[slot] += value
            self.head = (self.head+1) % self.capacity
            self.count += 1
        self.last_time = time
        self.invalid |= any(x > r_max for x in self.totals)
        return {'envelope_invalid': self.invalid, 'envelope_unverified': self.envelope_unverified,
                'rate_history_unknown': time < self.lost_until,
                'largest_retained_one_second_increment': max(self.totals), 'overflow_count': self.losses}


META_BYTES = 256
META_NAMES = ('handle', 'birth_sequence', 'strength', 'first_observed_end', 'last_observed_end',
              'interference_lower', 'interference_upper', 'gap_origin_lower', 'gap_origin_upper',
              'membership_total', 'first_committed_at', 'last_delivered_at', 'occurrence_id', 'support_id')


class EpisodeRetention:
    """Fixed metadata for retained episodes; descriptors/edges remain caller-owned.

New handles enter only through explicit positive-support admissions. Absent old
handles stay unretrieved. The caller owns globally fresh generation-qualified
handles and applies returned eviction decisions to its complete bank/graph.
"""

    def __init__(self, clock, tau_sec, kappa, strength_max, r_max, capacity=256, rate_capacity=144):
        if (not all(math.isfinite(x) and x > 0 for x in (tau_sec, kappa, strength_max, r_max))
                or strength_max < 1 or type(capacity) is not int or capacity < 1):
            raise ValueError('positive retention parameters, strength cap>=1 and fixed capacity required')
        self.clock = clock
        self.tau, self.kappa, self.cap, self.r_max = tau_sec, kappa, strength_max, r_max
        self.capacity = capacity
        self.storage = bytearray(capacity*META_BYTES)
        self.scratch = bytearray(capacity*META_BYTES)
        self.births = array('Q', [0]*capacity)
        self.increments = array('d', [0.]*capacity)
        self.window = InterferenceWindow(capacity, rate_capacity)
        self.sequence = 0
        self.last_digest = None
        self.last_end = clock.origin/clock.rate
        self.cut = -math.inf
        self.evictions = self.replays = 0

    def record(self, slot):
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError('retained metadata slot required')
        return dict(zip(META_NAMES, struct.unpack_from('<QQ10dQQ', self.storage, slot*META_BYTES)))

    def apply(self, write, observed_end, new_handles=()):
        if (write['epoch'] != self.clock.epoch or not math.isfinite(observed_end)
                or observed_end < max(self.cut, self.clock.cut)
                or not all(math.isfinite(write[k]) for k in ('start', 'support_end', 'committed_at', 'delivered_at'))
                or not self.clock.origin/self.clock.rate <= write['start'] < write['support_end'] <= write['committed_at']
                <= write['delivered_at'] <= observed_end
                or write['support_end'] > self.clock.end/self.clock.rate):
            raise ValueError('current-epoch sealed write with original support covered by the bus clock required')
        for key in ('sequence', 'occurrence_id', 'support_id'):
            if type(write[key]) is not int or not 0 <= write[key] < 2**64:
                raise ValueError('bounded stable ledger sequence and original identities required')
        support = write['support']
        assignments = support['episodes']
        if (any(type(h) is not int or not 0 < h < 2**64 or not math.isfinite(v) or v <= 0
                       for h, v in assignments.items())
                or not all(math.isfinite(support[k]) and 0 <= support[k] <= 1 for k in
                           ('available_support', 'unknown_episode_support', 'unassigned_available_support'))
                or abs(math.fsum(assignments.values())+support['unknown_episode_support']
                       +support['unassigned_available_support']-support['available_support']) > 1e-12):
            raise ValueError('conserved finite sealed assignment support required')
        snapshot = write['coarse_snapshot']
        receipt = None if snapshot is None else (
            snapshot['epoch'], snapshot['generation'], snapshot['query_id'], snapshot['occurrence_id'],
            snapshot['support_id'], snapshot['support_end'], snapshot['supporting_audio_end'],
            snapshot['available_end'], sorted(snapshot['entries'].items()))
        digest = hashlib.sha256(repr((write['sequence'], write['occurrence_id'], write['support_id'],
                                    write['start'], write['support_end'], write['committed_at'], write['delivered_at'],
                                    sorted(assignments.items()), support['unknown_episode_support'],
                                    support['unassigned_available_support'], receipt)).encode()).digest()
        if write['sequence'] <= self.sequence:
            if write['sequence'] == self.sequence and digest != self.last_digest:
                raise ValueError('conflicting repeated sealed write')
            self.replays += 1
            return False
        if write['sequence'] != self.sequence+1 or write['support_end'] < self.last_end:
            raise ValueError('complete chronological ledger sequence required; reorder upstream')
        new_handles = tuple(new_handles)
        current = {self.record(i)['handle']: i for i in range(self.capacity) if self.record(i)['handle']}
        if (len(set(new_handles)) != len(new_handles)
                or any(h in current or h not in assignments for h in new_handles)):
            raise ValueError('explicit fresh positive-support admissions required')
        free = [i for i in range(self.capacity) if not self.record(i)['handle']]
        if len(free) < len(new_handles):
            raise BufferError('bank capacity: explicitly evict and invalidate graph references before admission')
        prefix = self.clock.prefix(write['support_end'])
        start_prefix = self.clock.prefix(write['start'])
        duration = write['support_end']-write['start']
        known_fraction_max = 1-max(0., prefix['lower']-start_prefix['upper'])/duration
        if support['available_support'] > known_fraction_max+1e-12:
            raise ValueError('sealed acoustic support contradicts known bus acquisition loss')
        self.scratch[:] = self.storage
        for slot in range(self.capacity):
            self.increments[slot] = 0.
            record = self.record(slot)
            if not record['handle']:
                continue
            increment = occurrence_interference(write, record['handle'], record['first_observed_end'])
            recurrence = assignments.get(record['handle'], 0.)
            if recurrence:
                record['strength'] = min(self.cap, record['strength']+recurrence)
                record['last_observed_end'] = write['support_end']
                record['interference_lower'] = record['interference_upper'] = 0.
                record['gap_origin_lower'], record['gap_origin_upper'] = prefix['lower'], prefix['upper']
                record['membership_total'] += recurrence
                record['occurrence_id'], record['support_id'] = write['occurrence_id'], write['support_id']
            record['interference_lower'] += increment['lower']
            record['interference_upper'] += increment['upper']
            record['last_delivered_at'] = write['delivered_at']
            self.increments[slot] = increment['lower']
            struct.pack_into('<QQ10dQQ', self.scratch, slot*META_BYTES, *(record[k] for k in META_NAMES))
        for slot, handle in zip(free, new_handles):
            weight = assignments[handle]
            struct.pack_into('<QQ10dQQ', self.scratch, slot*META_BYTES,
                             handle, write['sequence'], weight, write['support_end'], write['support_end'],
                             0., 0., prefix['lower'], prefix['upper'], weight,
                             write['committed_at'], write['delivered_at'], write['occurrence_id'], write['support_id'])
        # All per-write validation completed before changing live metadata or the rate window.
        for slot, handle in zip(free, new_handles):
            self.births[slot] = write['sequence']
            self.window.totals[slot] = 0.
        self.window.observe(write['support_end'], write['sequence'], self.increments, self.births, self.r_max)
        self.storage[:] = self.scratch
        self.sequence, self.last_digest, self.last_end = write['sequence'], digest, write['support_end']
        self.cut = observed_end
        return True

    def availability(self, slot, time):
        r = self.record(slot)
        if not r['handle']:
            return None
        if not math.isfinite(time) or time < max(self.cut, self.clock.cut) or time < r['last_observed_end']:
            raise ValueError('live availability cannot be backdated before a delivered update')
        now = self.clock.prefix(time)
        missing_max = min(time-r['last_observed_end'], max(0., now['upper']-r['gap_origin_lower']))
        missing_min = max(0., now['lower']-r['gap_origin_upper'])
        uncertain_rate = self.window.invalid or self.window.envelope_unverified
        elapsed = time-r['last_observed_end']
        upper = episode_log_availability(r['strength'], elapsed, r['interference_lower'], self.tau, self.kappa)
        lower = (-math.inf if missing_max > 0 and uncertain_rate else
                 episode_log_availability(r['strength'], elapsed,
                                          r['interference_upper']+self.r_max*missing_max, self.tau, self.kappa))
        return {'handle': r['handle'], 'log_lower': lower, 'log_upper': upper,
                'elapsed_sec': elapsed, 'strength': r['strength'],
                'observed_interference': r['interference_lower'],
                'coarse_unknown_interference': r['interference_upper']-r['interference_lower'],
                'missing_seconds_lower': missing_min, 'missing_seconds_upper': missing_max,
                'gap_interference_upper': math.inf if missing_max and uncertain_rate else self.r_max*missing_max,
                'envelope_invalid': self.window.invalid, 'envelope_unverified': self.window.envelope_unverified,
                'rate_history_unknown': self.last_end < self.window.lost_until,
                'clock_history_lost': r['gap_origin_lower'] != r['gap_origin_upper'] or now['history_lost']}

    def recognition(self, match_scores, bias, time, epsilon_avail=1e-300):
        lower, upper, scores = [], [], []
        for slot in range(self.capacity):
            r = self.record(slot)
            if r['handle'] and r['handle'] in match_scores:
                bounds = self.availability(slot, time)
                lower.append(bounds['log_lower'])
                upper.append(bounds['log_upper'])
                scores.append(match_scores[r['handle']])
        # A fallback's zero availability is a true lower endpoint, not an eligible numerical floor.
        return {'lower': recognition_probability(lower, scores, bias, epsilon_avail),
                'upper': recognition_probability(upper, scores, bias, epsilon_avail),
                'supported_matches': len(scores)}

    def eviction_candidate(self, time):
        occupied = [i for i in range(self.capacity) if self.record(i)['handle']]
        if not occupied:
            return None
        return min(occupied, key=lambda i: (self.availability(i, time)['log_upper'],
                                           self.record(i)['first_committed_at'], self.record(i)['handle']))

    def assay_snapshot(self, target_start):
        """Freeze the pre-target bank; the query cannot reinforce its own past."""
        if not math.isfinite(target_start) or target_start < max(self.cut, self.clock.cut):
            raise ValueError('assay copy must be issued before hearing its target')
        payload = bytearray(32+self.capacity*24)
        count = 0
        for slot in range(self.capacity):
            r = self.record(slot)
            if r['handle'] and r['last_observed_end'] < target_start:
                bounds = self.availability(slot, target_start)
                struct.pack_into('<Qdd', payload, 32+count*24, r['handle'], bounds['log_lower'], bounds['log_upper'])
                count += 1
        struct.pack_into('<QQdd', payload, 0, self.clock.epoch, count, target_start, self.tau)
        return bytes(payload)

    def evict(self, slot):
        handle = self.record(slot)['handle']
        if not handle:
            return None
        self.storage[slot*META_BYTES:(slot+1)*META_BYTES] = bytes(META_BYTES)
        self.births[slot] = 0
        self.window.totals[slot] = 0.
        self.evictions += 1
        return handle

    def payload_bytes(self):
        return {'episode_metadata': len(self.storage), 'metadata_scratch': len(self.scratch),
                'gap_clock': len(self.clock.storage), 'rate_window': len(self.window.storage),
                'rate_totals': len(self.window.totals)*8,
                'slot_birth_and_increment_arrays': len(self.births)*8+len(self.increments)*8,
                'excludes': 'Python objects/counters, temporary validation/exports, descriptors/anchors/edges and full worker transport.'}


def assay_recognition(snapshot, epoch, match_scores, bias, query_end, epsilon_avail=1e-300):
    """Advance only elapsed time in a frozen identification-assay bank."""
    if len(snapshot) < 32:
        raise ValueError('packed pre-target assay snapshot required')
    saved_epoch, count, start, tau = struct.unpack_from('<QQdd', snapshot)
    if (saved_epoch != epoch or len(snapshot) < 32+count*24 or not math.isfinite(query_end)
            or query_end < start or not math.isfinite(tau) or tau <= 0):
        raise ValueError('same-epoch snapshot and causal heard query endpoint required')
    lower, upper, scores = [], [], []
    for i in range(count):
        handle, lo, hi = struct.unpack_from('<Qdd', snapshot, 32+i*24)
        if handle in match_scores:
            lower.append(lo-(query_end-start)/tau)
            upper.append(hi-(query_end-start)/tau)
            scores.append(match_scores[handle])
    return {'lower': recognition_probability(lower, scores, bias, epsilon_avail),
            'upper': recognition_probability(upper, scores, bias, epsilon_avail),
            'supported_matches': len(scores)}


def fit_gap_rate_envelope(development_windows):
    """Derive the registered rate assumption from complete development diagnostics."""
    count, maximum = 0, 0.
    for row in development_windows:
        value = row['largest_retained_one_second_increment']
        if (row['rate_history_unknown'] or row['envelope_unverified']
                or not math.isfinite(value) or value < 0):
            raise ValueError('complete finite development rate windows required')
        maximum = max(maximum, value)
        count += 1
    if count == 0:
        raise ValueError('no development rate observations to fit')
    return {'r_max': max(1., 2*maximum), 'maximum_one_second_increment': maximum,
            'development_windows': count, 'units': 'weighted occurrence units per second',
            'scope': 'Frozen operating-envelope assumption, not evidence about unheard content.'}
