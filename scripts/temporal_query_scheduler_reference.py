"""Bounded query dispatch and generation-qualified coarse snapshots.

One synchronous reference controller represents one bus and one outstanding
worker job. It does not launch threads, infer groups or implement episode-bank
identity allocation. Decoded worker/ledger exports are temporary allocations.
"""

from array import array
import hashlib
import math
import struct

from temporal_descriptor_reference import DescriptorKnot, PackedKnots
from temporal_matcher_reference import descriptor_query


SNAPSHOT_BYTES = 128 + 256 * 24
QUERY_HEADER_BYTES = 176


def _u64(value):
    return type(value) is int and 0 <= value < 2**64


def pack_coarse(result, bindings, destination, observed_end):
    """Pack the existing scan; bindings map (episode ID, generation) to (slot, handle).

Handle zero denotes an absent slot. Nonzero handles must name a unique episode
generation throughout the epoch. Their allocation/retirement belongs to the bank.
Validity and approximation occupy separate 256-bit arrays. No second similarity
scan runs here: the matcher already supplies exp(-cost).
"""
    keys = ('epoch', 'generation', 'query_id', 'occurrence_id', 'support_id')
    end, audio, completed = result['query_end'], result['supporting_audio_end'], result['completed_at']
    if (len(destination) != SNAPSHOT_BYTES or not all(_u64(result[k]) for k in keys)
            or not result['complete'] or result['superseded']
            or not all(math.isfinite(x) for x in (end, completed, observed_end, result['available_end']))
            or not 0 <= end <= result['available_end'] <= completed <= observed_end
            or audio is not None and (not math.isfinite(audio) or not 0 <= audio <= result['available_end'])
            or len(result['coarse_entries']) > 256 or len(bindings) > 256):
        raise ValueError('completed causal bounded coarse result required')
    slots, handles = set(), set()
    for key, (slot, handle) in bindings.items():
        if (len(key) != 2 or not all(_u64(x) for x in key) or type(slot) is not int
                or not 0 <= slot < 256 or not _u64(handle) or handle == 0
                or slot in slots or handle in handles):
            raise ValueError('distinct bounded bank slots and nonzero generation-qualified handles required')
        slots.add(slot)
        handles.add(handle)
    seen = set()
    for entry in result['coarse_entries']:
        key = entry['episode_id'], entry['episode_generation']
        cost, similarity = entry['cost'], entry['similarity']
        if (not all(_u64(x) for x in key) or key[0] in seen
                or (cost is None) != (similarity is None)
                or cost is not None and (not math.isfinite(cost) or cost < 0
                                        or not math.isfinite(similarity) or not 0 <= similarity <= 1)):
            raise ValueError('canonical episode entries with masked finite cost/similarity required')
        seen.add(key[0])
    destination[:] = bytes(SNAPSHOT_BYTES)
    # Receipt into this cache, not an earlier worker finish, makes it available.
    struct.pack_into('<5Q3d', destination, 0, *(result[k] for k in keys),
                     end, math.nan if audio is None else audio, observed_end)
    missing = 0
    for entry in result['coarse_entries']:
        key = entry['episode_id'], entry['episode_generation']
        if key not in bindings:
            missing += 1
            continue
        slot, handle = bindings[key]
        cost = entry['cost']
        struct.pack_into('<Qdd', destination, 128+slot*24, handle,
                         0. if cost is None else cost, 0. if cost is None else entry['similarity'])
        if cost is not None:
            destination[64+slot//8] |= 1 << (slot % 8)
        if entry['anchor_cap_loss'] or entry['bound_anchors'] or any(entry['unidentified']):
            destination[96+slot//8] |= 1 << (slot % 8)
    return missing


class CoarseCache:
    """Eight fixed records; endpoint/ID eviction cannot be undone by a replay."""

    def __init__(self, epoch, generation, capacity=8):
        if not _u64(epoch) or not _u64(generation) or type(capacity) is not int or capacity < 1:
            raise ValueError('unsigned epoch/generation and positive capacity required')
        self.epoch, self.generation, self.capacity = epoch, generation, capacity
        self.storage = bytearray(capacity * SNAPSHOT_BYTES)
        self.occupied = bytearray(capacity)
        self.highwater = -1
        self.evictions = self.stale = 0

    def clear(self, epoch, generation):
        count = sum(self.occupied)
        self.epoch, self.generation, self.highwater = epoch, generation, -1
        self.occupied[:] = bytes(self.capacity)
        return count

    def insert(self, packed):
        if len(packed) != SNAPSHOT_BYTES:
            raise ValueError('fixed coarse snapshot payload required')
        epoch, generation, query_id, _, _, end, _, available = struct.unpack_from('<5Q3d', packed)
        if epoch != self.epoch or generation != self.generation:
            self.stale += 1
            return False
        if not math.isfinite(end) or not math.isfinite(available) or available < end:
            raise ValueError('validated coarse payload required')
        oldest, oldest_key = None, None
        free = None
        for i, used in enumerate(self.occupied):
            if not used:
                free = i if free is None else free
                continue
            offset = i * SNAPSHOT_BYTES
            old_id = struct.unpack_from('<Q', self.storage, offset+16)[0]
            if old_id == query_id:
                if memoryview(self.storage)[offset:offset+SNAPSHOT_BYTES] != packed:
                    raise ValueError('conflicting repeated query snapshot')
                return False
            key = struct.unpack_from('<d', self.storage, offset+40)[0], old_id
            if oldest_key is None or key < oldest_key:
                oldest, oldest_key = i, key
        if query_id <= self.highwater:
            self.stale += 1
            return False
        self.highwater = query_id
        if free is None:
            self.evictions += 1
            if (end, query_id) < oldest_key:
                return False
        target = free if free is not None else oldest
        self.storage[target*SNAPSHOT_BYTES:(target+1)*SNAPSHOT_BYTES] = packed
        self.occupied[target] = 1
        return True

    def snapshots(self, current_handles, observed_end):
        """Export current bindings only; absent/retired entries remain unknown."""
        if (not math.isfinite(observed_end) or len(current_handles) > 256
                or any(type(i) is not int or not 0 <= i < 256 or not _u64(h) or h == 0
                       for i, h in current_handles.items())
                or len(set(current_handles.values())) != len(current_handles)):
            raise ValueError('finite cut and distinct live slot/handle bindings required')
        result = []
        for i, used in enumerate(self.occupied):
            if not used:
                continue
            block = memoryview(self.storage)[i*SNAPSHOT_BYTES:(i+1)*SNAPSHOT_BYTES]
            epoch, generation, query_id, occurrence, support, end, audio, available = struct.unpack_from('<5Q3d', block)
            if available > observed_end:
                continue
            entries, similarities, approximate = {}, {}, []
            for slot, current in current_handles.items():
                handle, cost, similarity = struct.unpack_from('<Qdd', block, 128+slot*24)
                if handle != current:
                    continue
                valid = bool(block[64+slot//8] & (1 << (slot % 8)))
                entries[handle] = cost if valid else None
                similarities[handle] = similarity if valid else None
                if block[96+slot//8] & (1 << (slot % 8)):
                    approximate.append(handle)
            result.append({'epoch': epoch, 'generation': generation, 'query_id': query_id,
                           'occurrence_id': occurrence, 'support_id': support,
                           'support_end': end, 'supporting_audio_end': None if math.isnan(audio) else audio,
                           'available_end': available, 'completed': True, 'superseded': False,
                           'entries': entries, 'similarities': similarities, 'approximate_handles': approximate})
        return sorted(result, key=lambda s: (s['support_end'], s['query_id']))


class QuerySlot:
    """A 176-byte header and at most128 original 320-byte knots, held in one buffer."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = bytearray(QUERY_HEADER_BYTES+capacity*320)

    def capture(self, span, observed_end, query_id, occurrence_id, support_id):
        if span.bank.capacity > self.capacity:
            raise ValueError('query descriptor exceeds registered capacity')
        q = descriptor_query(span, observed_end, query_id, occurrence_id, support_id)
        ids = tuple(q[k] for k in ('epoch', 'generation', 'query_id', 'occurrence_id', 'support_id'))
        if not all(_u64(x) for x in ids):
            raise ValueError('bounded unsigned query identities required')
        self.storage[:] = bytes(len(self.storage))
        struct.pack_into('<6Q16d', self.storage, 0, *ids, len(q['knots']),
                         q['start'], q['query_end'], math.nan if q['supporting_audio_end'] is None else q['supporting_audio_end'],
                         q['available_end'], q['issued_at'], q['reconstruction_error'], *q['scales'])
        self.storage[QUERY_HEADER_BYTES:QUERY_HEADER_BYTES+len(q['packed_knots'])] = q['packed_knots']

    def metadata(self):
        v = struct.unpack_from('<6Q16d', self.storage)
        result = dict(zip(('epoch', 'generation', 'query_id', 'occurrence_id', 'support_id', 'count',
                           'start', 'query_end', 'supporting_audio_end', 'available_end', 'issued_at',
                           'reconstruction_error'), v[:12]))
        if math.isnan(result['supporting_audio_end']):
            result['supporting_audio_end'] = None
        result.update(scales=list(v[12:]), complete=True, superseded=False)
        return result

    def export(self, packed=False):
        q = self.metadata()
        if packed:
            if not 0 <= q['count'] <= self.capacity:
                raise ValueError('bounded query knot count required')
            q['knots'] = PackedKnots(bytes(memoryview(self.storage)[QUERY_HEADER_BYTES:QUERY_HEADER_BYTES+q['count']*320]))
        else:
            q['knots'] = [DescriptorKnot(memoryview(self.storage)[QUERY_HEADER_BYTES+i*320:
                                     QUERY_HEADER_BYTES+(i+1)*320]).snapshot() for i in range(q['count'])]
        return q


class QueryScheduler:
    """Acquired-sample cadence, one latest pending query/group and one worker/bus.

Replacing pending work never invalidates a job already dispatched. Otherwise a
worker slower than the cadence could fail to publish any completed result. Group
retirement/rebinding and epoch restart invalidate that job but keep the worker
occupied until its completion/failure is acknowledged. Generations must increase
within each reused group slot; epochs must increase on restart.
"""

    def __init__(self, epoch, sample_rate=48000, hop_samples=512, cadence=.1,
                 groups=8, knots=128, snapshots=8):
        if (not _u64(epoch) or type(sample_rate) is not int or sample_rate <= 0
                or type(hop_samples) is not int or hop_samples <= 0
                or cadence not in (.05, .1, .2) or type(groups) is not int or not 1 <= groups <= 8
                or type(knots) is not int or not 3 <= knots <= 256):
            raise ValueError('registered epoch, sample clock, cadence and group/knot bounds required')
        self.epoch, self.rate, self.hop_samples = epoch, sample_rate, hop_samples
        self.cadence, self.period = cadence, math.ceil(cadence*sample_rate)
        self.groups = groups
        self.pending = [QuerySlot(knots) for _ in range(groups)]
        self.active = QuerySlot(knots)
        self.query_scratch = QuerySlot(knots)
        self.caches = [CoarseCache(epoch, 0, snapshots) for _ in range(groups)]
        self.coarse_scratch = bytearray(SNAPSHOT_BYTES)
        self.generation = array('Q', [0]*groups)
        self.ever_bound = bytearray(groups)
        self.bound = bytearray(groups)
        self.has_observation = bytearray(groups)
        self.last_sample = array('Q', [0]*groups)
        self.samples = array('Q', [0]*groups)
        self.next_due = array('Q', [self.period]*groups)
        self.pending_valid = bytearray(groups)
        self.query_seen = bytearray(groups)
        self.query_highwater = array('Q', [0]*groups)
        self.endpoint_highwater = array('d', [0.]*groups)
        self.ready_order = array('Q', [0]*groups)
        self.last_clock_digest = bytearray(groups*32)
        self.cut = -math.inf
        self.order = self.ticket = 0
        self.active_slot = None
        self.active_obsolete = False
        self.dispatched_at = None
        self.counts = {k: 0 for k in ('submitted', 'superseded_pending', 'older_query_loss',
                                     'stale_query', 'dispatched', 'completed', 'incomplete',
                                     'stale_completion', 'retired_pending', 'retired_snapshots',
                                     'invalidated_active', 'unbound_episode_entries')}

    def _validate_cut(self, observed_end, slot=None):
        if not math.isfinite(observed_end) or observed_end < self.cut:
            raise ValueError('finite monotone bus observation cut required')
        if slot is not None and (type(slot) is not int or not 0 <= slot < self.groups or not self.bound[slot]):
            raise ValueError('currently bound group slot required')

    def retire(self, slot, observed_end):
        self._validate_cut(observed_end, slot)
        self.counts['retired_pending'] += self.pending_valid[slot]
        self.counts['retired_snapshots'] += self.caches[slot].clear(self.epoch, self.generation[slot])
        if self.active_slot == slot and not self.active_obsolete:
            self.active_obsolete = True
            self.counts['invalidated_active'] += 1
        self.pending_valid[slot] = self.bound[slot] = 0
        self.cut = observed_end

    def bind(self, slot, generation, observed_end):
        self._validate_cut(observed_end)
        if (type(slot) is not int or not 0 <= slot < self.groups or not _u64(generation)
                or self.ever_bound[slot] and generation <= self.generation[slot]):
            raise ValueError('unused slot generation must increase; no ABA reuse')
        if self.bound[slot]:
            self.retire(slot, observed_end)
        self.generation[slot] = generation
        self.ever_bound[slot] = self.bound[slot] = 1
        self.has_observation[slot] = self.query_seen[slot] = 0
        self.samples[slot] = self.last_sample[slot] = self.query_highwater[slot] = 0
        self.endpoint_highwater[slot] = 0.
        self.next_due[slot] = self.period
        self.caches[slot].clear(self.epoch, generation)
        self.cut = observed_end

    def restart(self, epoch, sample_rate, hop_samples, observed_end):
        if (not _u64(epoch) or epoch <= self.epoch or not math.isfinite(observed_end)
                or type(sample_rate) is not int or sample_rate <= 0
                or type(hop_samples) is not int or hop_samples <= 0):
            raise ValueError('new epoch and positive sample clock required')
        for slot in range(self.groups):
            if self.bound[slot]:
                self.retire(slot, self.cut)
        self.epoch, self.rate, self.hop_samples = epoch, sample_rate, hop_samples
        self.period = math.ceil(self.cadence*sample_rate)
        self.ever_bound[:] = bytes(self.groups)
        self.cut = observed_end

    def observe(self, slot, raw, observed_end):
        """Count the union of canonical acquired samples; gaps and rereads add none."""
        self._validate_cut(observed_end, slot)
        lo, hi = raw['sample_start'], raw['sample_end']
        if (not _u64(lo) or not _u64(hi) or hi-lo != self.hop_samples
                or raw['sample_rate'] != self.rate or raw['epoch'] != self.epoch
                or raw['generation'] != self.generation[slot]
                or not all(math.isfinite(raw[k]) for k in ('raw_support_end', 'available_end'))
                or not hi/self.rate <= raw['raw_support_end'] <= raw['available_end'] <= observed_end):
            raise ValueError('current generation and causally acquired canonical hop required')
        intervals = raw.get('known_sample_intervals', [(lo, hi)] if raw['observed'] else [])
        if len(intervals) > self.hop_samples:
            raise ValueError('sample-interval input exceeds its hop bound')
        left, acquired = lo, 0
        canonical = []
        for a, b in sorted(intervals):
            if not _u64(a) or not _u64(b) or not lo <= a <= b <= hi:
                raise ValueError('known integer sample intervals must lie within the hop')
            acquired += max(0, b-max(left, a))
            left = max(left, b)
            if a < b:
                if canonical and a <= canonical[-1][1]:
                    canonical[-1] = (canonical[-1][0], max(b, canonical[-1][1]))
                else:
                    canonical.append((a, b))
        if bool(acquired) != bool(raw['observed']):
            raise ValueError('acquisition mask and sample intervals disagree')
        digest = hashlib.sha256(repr((lo, hi, canonical, raw['raw_support_end'], raw['available_end'])).encode()).digest()
        if self.has_observation[slot] and lo < self.last_sample[slot]:
            if hi == self.last_sample[slot] and digest == self.last_clock_digest[slot*32:(slot+1)*32]:
                self.cut = observed_end
                return False
            raise ValueError('stale, overlapping or conflicting observation; upstream ordering required')
        if not _u64(self.samples[slot]+acquired):
            raise ValueError('sample clock overflow requires epoch restart')
        self.samples[slot] += acquired
        self.last_sample[slot] = hi
        self.has_observation[slot] = 1
        self.last_clock_digest[slot*32:(slot+1)*32] = digest
        self.cut = observed_end
        return True

    def submit(self, slot, span, query_id, occurrence_id, support_id, observed_end):
        self._validate_cut(observed_end, slot)
        if (span.epoch != self.epoch or span.generation != self.generation[slot] or span.rate != self.rate
                or not all(_u64(x) for x in (query_id, occurrence_id, support_id))):
            raise ValueError('query must use current group generation and unsigned identities')
        if self.query_seen[slot] and query_id <= self.query_highwater[slot]:
            self.counts['stale_query'] += 1
            self.cut = observed_end
            return False
        self.query_scratch.capture(span, observed_end, query_id, occurrence_id, support_id)
        self.query_highwater[slot] = query_id
        self.query_seen[slot] = 1
        self.counts['submitted'] += 1
        self.cut = observed_end
        if span.cursor < self.endpoint_highwater[slot]:
            self.counts['older_query_loss'] += 1
            return False
        self.endpoint_highwater[slot] = span.cursor
        if self.pending_valid[slot]:
            self.counts['superseded_pending'] += 1
        else:
            self.order += 1
            self.ready_order[slot] = self.order
        self.pending[slot].storage[:] = self.query_scratch.storage
        self.pending_valid[slot] = 1
        return True

    def take(self, observed_end, packed=False):
        self._validate_cut(observed_end)
        self.cut = observed_end
        if self.active_slot is not None:
            return None
        eligible = [i for i in range(self.groups) if self.bound[i] and self.pending_valid[i]
                    and self.samples[i] >= self.next_due[i]]
        if not eligible:
            return None
        slot = min(eligible, key=lambda i: (self.ready_order[i], i))
        next_due = (self.samples[slot]//self.period+1)*self.period
        if not _u64(next_due) or not _u64(self.ticket+1):
            raise ValueError('scheduler clock/ticket exhausted')
        self.active.storage[:] = self.pending[slot].storage
        self.pending_valid[slot] = 0
        self.next_due[slot] = next_due
        self.active_slot, self.active_obsolete = slot, False
        self.dispatched_at = observed_end
        self.ticket += 1
        self.counts['dispatched'] += 1
        return {'ticket': self.ticket, 'group_slot': slot, 'query': self.active.export(packed=packed),
                'dispatched_at': observed_end}

    def finish(self, ticket, result, bindings, observed_end):
        """Acknowledge one job; stale receipts cannot release or replace a newer job."""
        self._validate_cut(observed_end)
        if self.active_slot is None or ticket != self.ticket:
            self.counts['stale_completion'] += 1
            self.cut = observed_end
            return False
        if self.active_obsolete:
            self.counts['stale_completion'] += 1
            self.active_slot = None
            self.cut = observed_end
            return False
        q = self.active.metadata()
        if result is not None:
            if (any(result[k] != q[k] for k in ('epoch', 'generation', 'query_id', 'occurrence_id',
                                               'support_id', 'query_end', 'supporting_audio_end', 'available_end'))
                    or not math.isfinite(result['completed_at'])
                    or not self.dispatched_at <= result['completed_at'] <= observed_end):
                raise ValueError('completion must name the dispatched frozen query and causal finish time')
        if result is None or not result['complete'] or result['superseded']:
            self.counts['incomplete'] += 1
            accepted = False
        else:
            missing = pack_coarse(result, bindings, self.coarse_scratch, observed_end)
            accepted = self.caches[self.active_slot].insert(self.coarse_scratch)
            self.counts['unbound_episode_entries'] += missing
            self.counts['completed'] += 1
        self.active_slot = None
        self.cut = observed_end
        return accepted

    def status(self, observed_end):
        self._validate_cut(observed_end)
        rows = []
        for slot in range(self.groups):
            q = self.pending[slot].metadata() if self.pending_valid[slot] else None
            rows.append({'slot': slot, 'bound': bool(self.bound[slot]), 'generation': self.generation[slot],
                         'acquired_samples': self.samples[slot], 'next_due_samples': self.next_due[slot],
                         'pending_query': None if q is None else q['query_id'],
                         'pending_age': None if q is None else observed_end-q['issued_at'],
                         'snapshots': sum(self.caches[slot].occupied)})
        q = self.active.metadata() if self.active_slot is not None else None
        return {'epoch': self.epoch, 'groups': rows, 'counts': dict(self.counts),
                'active_slot': self.active_slot, 'active_obsolete': self.active_obsolete,
                'active_query': None if q is None else q['query_id'],
                'worker_age': None if q is None or self.active_obsolete else observed_end-self.dispatched_at,
                'support_age': None if q is None or self.active_obsolete or q['supporting_audio_end'] is None
                else observed_end-q['supporting_audio_end'],
                'snapshot_evictions': sum(c.evictions for c in self.caches),
                'stale_snapshots': sum(c.stale for c in self.caches)}

    def payload_bytes(self):
        buffers = sum(len(q.storage) for q in [*self.pending, self.active, self.query_scratch])
        caches = sum(len(c.storage) for c in self.caches)
        controls = sum(len(v)*v.itemsize if isinstance(v, array) else len(v)
                       for v in vars(self).values() if isinstance(v, (array, bytearray))
                       and v is not self.coarse_scratch)
        return {'query_buffers_including_active_and_scratch': buffers,
                'coarse_snapshots': caches, 'coarse_scratch': len(self.coarse_scratch),
                'control_arrays': controls+sum(len(c.occupied) for c in self.caches),
                'total_fixed_buffers': buffers+caches+len(self.coarse_scratch)+controls
                +sum(len(c.occupied) for c in self.caches),
                'excludes': 'Python objects/scalar counters, temporary capture/exports, matcher/bank/ledger and runtime.'}
