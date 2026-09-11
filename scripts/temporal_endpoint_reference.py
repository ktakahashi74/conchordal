"""Bounded endpoint queue; payload pools and inferred ledger writes are separate.

The producer supplies stable original occurrence/support IDs and immutable,
generation-qualified payload handles. The queue retains no hidden payload objects.
Taking an endpoint freezes its header; acknowledgment releases the slot only after
the caller commits or explicitly reports unsealed computational loss.
"""

from array import array
import math
import struct


TOMBSTONE = 2**32-1
INTEGER_FIELDS = ('ticket', 'occurrence_id', 'support_id', 'generation', 'descriptor_ref',
                  'activity_ref', 'joint_ref', 'lineage_ref', 'revision', 'flags')
TIME_FIELDS = ('start', 'support_end', 'deadline', 'first_received_at', 'updated_at', 'source_available_end')


class EndpointQueue:
    """128-byte headers, fixed min-heap/free list and two bounded ID indexes."""

    def __init__(self, epoch, capacity=65536, lag_sec=.5, writes_per_cycle=10240):
        if (type(epoch) is not int or not 0 <= epoch < 2**64
                or type(capacity) is not int or not 0 < capacity < TOMBSTONE
                or not math.isfinite(lag_sec) or lag_sec <= 0
                or type(writes_per_cycle) is not int or writes_per_cycle < 1):
            raise ValueError('unsigned epoch, bounded positive capacity, lag and write budget required')
        self.epoch, self.capacity, self.lag, self.limit = epoch, capacity, lag_sec, writes_per_cycle
        self.storage = bytearray(capacity*128)
        self.heap = array('I', [0]*capacity)
        self.free = array('I', reversed(range(capacity)))
        self.table_size = 1 << (2*capacity-1).bit_length()
        self.occurrences = array('I', [0]*self.table_size)
        self.supports = array('I', [0]*self.table_size)
        self.heap_count = 0
        self.free_count = capacity
        self.next_ticket = 1
        self.active = None
        self.active_delivery = None
        self.observed_cut = self.sealed_cut = -math.inf
        self.cycle_id = -1
        self.cycle_start = -math.inf
        self.cycle_takes = self.committed = self.dropped = 0
        self.capacity_losses = self.late_losses = self.stale_references = 0
        self.index_probes = self.heap_comparisons = self.maximum_probe = 0

    def _lookup(self, table, identity, field):
        at = (identity*11400714819323198485) & (self.table_size-1)
        reusable = None
        for count in range(1, self.table_size+1):
            value = table[at]
            self.index_probes += 1
            self.maximum_probe = max(self.maximum_probe, count)
            if value == 0:
                return None, at if reusable is None else reusable
            if value == TOMBSTONE:
                if reusable is None:
                    reusable = at
            elif struct.unpack_from('<Q', self.storage, (value-1)*128+field*8)[0] == identity:
                return value-1, at
            at = (at+1) & (self.table_size-1)
        if reusable is None:
            raise RuntimeError('fixed ID index occupancy invariant violated')
        return None, reusable

    def _key(self, slot):
        self.heap_comparisons += 1
        end = struct.unpack_from('<d', self.storage, slot*128+88)[0]
        start = struct.unpack_from('<d', self.storage, slot*128+80)[0]
        identity = struct.unpack_from('<Q', self.storage, slot*128+8)[0]
        return end, start, identity

    def record(self, slot):
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError('bounded endpoint slot required')
        values = struct.unpack_from('<10Q6d', self.storage, slot*128)
        return dict(zip(INTEGER_FIELDS+TIME_FIELDS, values))

    def offer(self, original, observed_end):
        """Register an ending before sealing; aliases retain its original identity."""
        if original['epoch'] != self.epoch:
            self.stale_references += 1
            return None
        keys = INTEGER_FIELDS[1:8]+('flags',)
        if (any(type(original[k]) is not int or not 0 <= original[k] < 2**64 for k in keys)
                or any(original[k] == 0 for k in ('descriptor_ref', 'activity_ref', 'joint_ref', 'lineage_ref'))
                or not all(math.isfinite(v) for v in
                           (original['start'], original['support_end'], original['source_available_end'], observed_end))
                or not 0 <= original['start'] < original['support_end'] <= original['source_available_end'] <= observed_end
                or observed_end < self.observed_cut):
            raise ValueError('stable original IDs, nonzero payload handles and causal ending receipt required')
        slot, occurrence_at = self._lookup(self.occurrences, original['occurrence_id'], 1)
        if slot is not None:
            prior = self.record(slot)
            immutable = ('occurrence_id', 'support_id', 'generation', 'descriptor_ref', 'lineage_ref',
                         'flags', 'start', 'support_end')
            if any(prior[k] != original[k] for k in immutable):
                raise ValueError('an occurrence alias cannot change original support/provenance')
            self.observed_cut = observed_end
            return prior['ticket']
        owner, support_at = self._lookup(self.supports, original['support_id'], 2)
        if owner is not None:
            raise ValueError('distinct occurrences cannot reclaim one pending support identity')
        deadline = original['support_end']+self.lag
        if not math.isfinite(deadline):
            raise ValueError('finite commitment deadline required')
        if deadline <= self.sealed_cut or observed_end > deadline:
            self.late_losses += 1
            self.observed_cut = observed_end
            return None
        if not self.free_count:
            self.capacity_losses += 1
            self.observed_cut = observed_end
            return None
        if self.next_ticket >= 2**64:
            raise OverflowError('ticket exhaustion requires a new epoch, never wrap')
        slot = self.free[self.free_count-1]
        self.free_count -= 1
        ticket = self.next_ticket
        self.next_ticket += 1
        struct.pack_into('<10Q6d', self.storage, slot*128, ticket,
                         *(original[k] for k in INTEGER_FIELDS[1:8]), 0, original['flags'],
                         original['start'], original['support_end'], deadline, observed_end,
                         observed_end, original['source_available_end'])
        self.occurrences[occurrence_at] = self.supports[support_at] = slot+1
        position = self.heap_count
        self.heap_count += 1
        while position:
            parent = (position-1)//2
            if self._key(self.heap[parent]) <= self._key(slot):
                break
            self.heap[position] = self.heap[parent]
            position = parent
        self.heap[position] = slot
        self.observed_cut = observed_end
        return ticket

    def revise(self, epoch, occurrence_id, ticket, revision, activity_ref, joint_ref, source_available_end, observed_end):
        """Replace provisional payload handles without moving the original deadline."""
        if epoch != self.epoch:
            self.stale_references += 1
            return False
        if (type(occurrence_id) is not int or not 0 <= occurrence_id < 2**64
                or type(ticket) is not int or not 0 < ticket < 2**64):
            raise ValueError('unsigned original identity and nonzero ticket required')
        slot, _ = self._lookup(self.occurrences, occurrence_id, 1)
        if slot is None or self.record(slot)['ticket'] != ticket:
            self.stale_references += 1
            return False
        prior = self.record(slot)
        if (any(type(v) is not int or not 0 < v < 2**64 for v in (revision, activity_ref, joint_ref))
                or not all(math.isfinite(v) for v in (source_available_end, observed_end))
                or observed_end < self.observed_cut or source_available_end < prior['source_available_end']
                or not source_available_end <= observed_end <= prior['deadline']):
            raise ValueError('causal provisional revision before original deadline required')
        if self.active == slot:
            raise ValueError('claimed endpoint is frozen; later interpretation is separate metadata')
        if revision <= prior['revision']:
            if revision == prior['revision'] and (activity_ref, joint_ref, source_available_end) != (
                    prior['activity_ref'], prior['joint_ref'], prior['source_available_end']):
                raise ValueError('conflicting payload for one revision')
            return False
        struct.pack_into('<QQ', self.storage, slot*128+40, activity_ref, joint_ref)
        struct.pack_into('<Q', self.storage, slot*128+64, revision)
        struct.pack_into('<dd', self.storage, slot*128+112, observed_end, source_available_end)
        self.observed_cut = observed_end
        return {'changed': True, 'release_refs': {
            key: prior[key] for key, replacement in (('activity_ref', activity_ref), ('joint_ref', joint_ref))
            if prior[key] != replacement}}

    def take(self, cycle_id, observed_end):
        """Claim the oldest due endpoint, holding its slot until acknowledgment."""
        if (type(cycle_id) is not int or not 0 <= cycle_id < 2**64 or cycle_id < self.cycle_id
                or not math.isfinite(observed_end) or observed_end < self.observed_cut):
            raise ValueError('monotone processing cycle and observation cut required')
        if cycle_id != self.cycle_id:
            if observed_end <= self.cycle_start:
                raise ValueError('a new processing cycle must advance its observation cut')
            self.cycle_id, self.cycle_start = cycle_id, observed_end
            self.cycle_takes = int(self.active is not None)
        self.observed_cut = observed_end
        self.sealed_cut = max(self.sealed_cut, observed_end)
        if self.active is not None:
            slot = self.active
        else:
            if not self.heap_count or self.cycle_takes >= self.limit:
                return None
            slot = self.heap[0]
            if self.record(slot)['deadline'] > observed_end:
                return None
            self.heap_count -= 1
            if self.heap_count:
                tail = self.heap[self.heap_count]
                position = 0
                while 2*position+1 < self.heap_count:
                    child = 2*position+1
                    if child+1 < self.heap_count and self._key(self.heap[child+1]) < self._key(self.heap[child]):
                        child += 1
                    if self._key(tail) <= self._key(self.heap[child]):
                        break
                    self.heap[position] = self.heap[child]
                    position = child
                self.heap[position] = tail
            self.active, self.active_delivery = slot, observed_end
            self.cycle_takes += 1
        return {'epoch': self.epoch, 'ticket': self.record(slot)['ticket'],
                'sequence': self.committed+1, 'delivered_at': self.active_delivery,
                'packed': bytes(self.storage[slot*128:(slot+1)*128])}

    def finish(self, epoch, ticket, committed, observed_end):
        """Release only the acknowledged active ticket; failed sealing is explicit loss."""
        if type(ticket) is not int or not 0 < ticket < 2**64:
            raise ValueError('unsigned nonzero acknowledgment ticket required')
        if epoch != self.epoch or self.active is None or self.record(self.active)['ticket'] != ticket:
            self.stale_references += 1
            return None
        if type(committed) is not bool or not math.isfinite(observed_end) or observed_end < self.observed_cut:
            raise ValueError('explicit commit/loss acknowledgment at a current cut required')
        slot = self.active
        record = self.record(slot)
        for table, identity, field in ((self.occurrences, record['occurrence_id'], 1),
                                       (self.supports, record['support_id'], 2)):
            found, at = self._lookup(table, identity, field)
            if found != slot:
                raise RuntimeError('active endpoint ID index invariant violated')
            table[at] = TOMBSTONE
        self.storage[slot*128:(slot+1)*128] = bytes(128)
        self.free[self.free_count] = slot
        self.free_count += 1
        self.active = self.active_delivery = None
        self.committed += committed
        self.dropped += not committed
        self.observed_cut = observed_end
        return {'record': record, 'committed': committed,
                'reason': None if committed else 'unsealed_computational_loss',
                'release_refs': tuple(record[k] for k in ('descriptor_ref', 'activity_ref', 'joint_ref', 'lineage_ref'))}

    def payload_bytes(self):
        return {'endpoint_headers': len(self.storage), 'heap': len(self.heap)*4,
                'free_slots': len(self.free)*4, 'id_indexes': (len(self.occurrences)+len(self.supports))*4,
                'total_fixed_buffers': len(self.storage)+4*(len(self.heap)+len(self.free)+len(self.occurrences)+len(self.supports)),
                'excludes': 'Scalar/object overhead,128-byte claim copy, referenced descriptor/activity/joint/lineage pools, receipt ledger, inferred writes and actual worker transport.'}
