"""Fixed immutable payload storage and endpoint ownership for the M0 reference.

Sharing is explicit: retain an existing handle rather than copy its bytes. This
does not infer beam aliases, intern equal descriptors or share interior knots.
The pool stores bytes, not hidden Python payload objects or inferred ledger state.
"""

from array import array
import struct

from temporal_endpoint_reference import EndpointQueue, INTEGER_FIELDS, TIME_FIELDS


KINDS = {'descriptor_ref': 1, 'activity_ref': 2, 'joint_ref': 3, 'lineage_ref': 4}
EMPTY = 2**32-1


class PayloadPool:
    """Fixed 320-byte blocks, 24-byte object metadata and versioned u64 handles."""

    def __init__(self, epoch, slots=65536, blocks=262144, max_payload=65536):
        if (type(epoch) is not int or not 0 <= epoch < 2**64
                or any(type(v) is not int or not 0 < v < EMPTY for v in (slots, blocks, max_payload))):
            raise ValueError('unsigned epoch and positive bounded pool capacities required')
        self.epoch, self.slots, self.blocks, self.max_payload = epoch, slots, blocks, max_payload
        self.storage = bytearray(blocks*320)
        self.metadata = bytearray(slots*24)
        self.next_block = array('I', [EMPTY])*blocks
        self.free_blocks = array('I', reversed(range(blocks)))
        self.free_slots = array('I', reversed(range(slots)))
        if self.next_block.itemsize != 4:
            raise RuntimeError('u32 pool indexes required')
        self.available_blocks, self.available_slots = blocks, slots
        self.serial = 0
        self.capacity_losses = self.allocations = self.releases = self.copied_bytes = 0
        self.peak_blocks = self.peak_slots = 0

    def _record(self, epoch, handle, kind=None):
        if (epoch != self.epoch or type(handle) is not int or not 0 < handle < 2**64):
            raise ValueError('live epoch-qualified payload handle required')
        slot = (handle-1) % self.slots
        record = struct.unpack_from('<Q4I', self.metadata, slot*24)
        if record[0] != handle or record[3] == 0 or (kind is not None and record[4] != kind):
            raise ValueError('retired or wrong-kind payload handle')
        return slot, record

    def create(self, kind, payload):
        """Create one owner; capacity rejection leaves all existing objects intact."""
        if type(kind) is not int or kind not in KINDS.values() or not isinstance(payload, (bytes, bytearray)):
            raise ValueError('known payload kind and contiguous byte input required')
        length = len(payload)
        if not 0 < length <= self.max_payload:
            raise ValueError('positive payload length within the registered bound required')
        count = (length+319)//320
        if not self.available_slots or count > self.available_blocks:
            self.capacity_losses += 1
            return None
        slot = self.free_slots[self.available_slots-1]
        handle = self.serial*self.slots+slot+1
        if handle >= 2**64:
            raise OverflowError('payload identity exhaustion requires a new epoch')
        first = EMPTY
        previous = EMPTY
        for offset in range(0, length, 320):
            self.available_blocks -= 1
            block = self.free_blocks[self.available_blocks]
            self.next_block[block] = EMPTY
            if previous == EMPTY:
                first = block
            else:
                self.next_block[previous] = block
            size = min(320, length-offset)
            self.storage[block*320:block*320+size] = payload[offset:offset+size]
            previous = block
        self.available_slots -= 1
        struct.pack_into('<Q4I', self.metadata, slot*24, handle, length, first, 1, kind)
        self.serial += 1
        self.allocations += 1
        self.copied_bytes += length
        self.peak_slots = max(self.peak_slots, self.slots-self.available_slots)
        self.peak_blocks = max(self.peak_blocks, self.blocks-self.available_blocks)
        return handle

    def retain(self, epoch, handle, kind=None):
        slot, record = self._record(epoch, handle, kind)
        if record[3] == EMPTY:
            raise OverflowError('payload reference count cannot wrap')
        struct.pack_into('<I', self.metadata, slot*24+16, record[3]+1)
        return handle

    def create_bundle(self, payloads):
        """Acquire all four producer-owned payloads or release the partial bundle."""
        if set(payloads) != set(KINDS):
            raise ValueError('exact descriptor/activity/joint/lineage payload inventory required')
        acquired = {}
        complete = False
        try:
            for key, kind in KINDS.items():
                handle = self.create(kind, payloads[key])
                if handle is None:
                    return None
                acquired[key] = handle
            complete = True
            return acquired
        finally:
            if not complete:
                for handle in acquired.values():
                    self.release(self.epoch, handle)

    def release(self, epoch, handle, kind=None):
        slot, record = self._record(epoch, handle, kind)
        if record[3] > 1:
            struct.pack_into('<I', self.metadata, slot*24+16, record[3]-1)
            return False
        remaining, block = record[1], record[2]
        while remaining:
            following = self.next_block[block]
            self.storage[block*320:(block+1)*320] = bytes(320)
            self.next_block[block] = EMPTY
            self.free_blocks[self.available_blocks] = block
            self.available_blocks += 1
            remaining -= min(320, remaining)
            block = following
        self.metadata[slot*24:(slot+1)*24] = bytes(24)
        self.free_slots[self.available_slots] = slot
        self.available_slots += 1
        self.releases += 1
        return True

    def read(self, epoch, handle, kind=None):
        """Return a detached bounded copy; callers cannot mutate retained bytes."""
        _, record = self._record(epoch, handle, kind)
        result = bytearray(record[1])
        block = record[2]
        for offset in range(0, len(result), 320):
            size = min(320, len(result)-offset)
            result[offset:offset+size] = self.storage[block*320:block*320+size]
            block = self.next_block[block]
        self.copied_bytes += len(result)
        return bytes(result)

    def owners(self, epoch, handle):
        return self._record(epoch, handle)[1][3]

    def payload_bytes(self):
        return {'blocks': len(self.storage), 'object_metadata': len(self.metadata),
                'block_links': self.next_block.itemsize*len(self.next_block),
                'free_blocks': self.free_blocks.itemsize*len(self.free_blocks),
                'free_objects': self.free_slots.itemsize*len(self.free_slots),
                'total_fixed_buffers': len(self.storage)+len(self.metadata)+4*(2*self.blocks+self.slots),
                'excludes': 'Python objects/scalars, caller input, detached read copies, block-sized scratch, all other worker state.'}


class OccurrenceIds:
    """One nonreused pair per original span; reinterpretations retain that pair."""

    def __init__(self, epoch):
        if type(epoch) is not int or not 0 <= epoch < 2**64:
            raise ValueError('unsigned epoch required')
        self.epoch, self.next_id = epoch, 1

    def allocate(self):
        if self.next_id >= 2**64:
            raise OverflowError('original identity exhaustion requires a new epoch')
        identity = self.next_id
        self.next_id += 1
        return {'epoch': self.epoch, 'occurrence_id': identity, 'support_id': identity}


class OwnedEndpointQueue:
    """Pin four typed pool objects through revision, active claims and release.

    Producer ownership is separate: an accepted offer retains its own references.
    Successful acknowledgment still requires a real consumer commit upstream.
    """

    def __init__(self, pool, capacity=65536, lag_sec=.5, writes_per_cycle=10240):
        self.pool = pool
        self.queue = EndpointQueue(pool.epoch, capacity, lag_sec, writes_per_cycle)
        self.closed = False

    def _open(self):
        if self.closed:
            raise ValueError('retired endpoint owner cannot reopen within its epoch')

    def offer(self, original, observed_end):
        self._open()
        q, p = self.queue, self.pool
        if original['epoch'] != q.epoch:
            return q.offer(original, observed_end)
        identity = original['occurrence_id']
        if type(identity) is not int or not 0 <= identity < 2**64:
            raise ValueError('unsigned original occurrence identity required')
        slot, _ = q._lookup(q.occurrences, identity, 1)
        if slot is not None:
            # Old mutable references can already be retired after a revision.
            return q.offer(original, observed_end)
        pinned = []
        try:
            for key, kind in KINDS.items():
                p.retain(q.epoch, original[key], kind)
                pinned.append((original[key], kind))
            ticket = q.offer(original, observed_end)
        except Exception:
            for handle, kind in pinned:
                p.release(q.epoch, handle, kind)
            raise
        if ticket is None:
            for handle, kind in pinned:
                p.release(q.epoch, handle, kind)
        return ticket

    def revise(self, epoch, occurrence_id, ticket, revision, activity_ref, joint_ref, source_available_end, observed_end):
        self._open()
        q, p = self.queue, self.pool
        args = (epoch, occurrence_id, ticket, revision, activity_ref, joint_ref, source_available_end, observed_end)
        if epoch != q.epoch or type(occurrence_id) is not int or not 0 <= occurrence_id < 2**64:
            return q.revise(*args)
        slot, _ = q._lookup(q.occurrences, occurrence_id, 1)
        prior = None if slot is None else q.record(slot)
        if prior is None or prior['ticket'] != ticket or type(revision) is not int or revision <= prior['revision']:
            return q.revise(*args)
        pinned = []
        try:
            for key, handle in (('activity_ref', activity_ref), ('joint_ref', joint_ref)):
                if prior[key] != handle:
                    p.retain(epoch, handle, KINDS[key])
                    pinned.append((handle, KINDS[key]))
            changed = q.revise(*args)
        except Exception:
            for handle, kind in pinned:
                p.release(epoch, handle, kind)
            raise
        if changed:
            for key, handle in changed['release_refs'].items():
                p.release(epoch, handle, KINDS[key])
        else:
            for handle, kind in pinned:
                p.release(epoch, handle, kind)
        return changed

    def take(self, cycle_id, observed_end):
        self._open()
        return self.queue.take(cycle_id, observed_end)

    def payloads(self, claim):
        self._open()
        q = self.queue
        if (claim['epoch'] != q.epoch or q.active is None or q.record(q.active)['ticket'] != claim['ticket']
                or claim['packed'] != bytes(q.storage[q.active*128:(q.active+1)*128])):
            raise ValueError('current unmodified active claim required')
        record = dict(zip(INTEGER_FIELDS+TIME_FIELDS, struct.unpack('<10Q6d', claim['packed'])))
        return {key: self.pool.read(q.epoch, record[key], kind) for key, kind in KINDS.items()}

    def finish(self, epoch, ticket, committed, observed_end):
        self._open()
        result = self.queue.finish(epoch, ticket, committed, observed_end)
        if result is not None:
            for key, kind in KINDS.items():
                self.pool.release(epoch, result['record'][key], kind)
        return result

    def close(self):
        """Retire pending/active ownership as loss without fabricated acknowledgments."""
        if self.closed:
            return 0
        q = self.queue
        lost = 0
        for entry in q.occurrences:
            if entry in (0, EMPTY):
                continue
            slot = entry-1
            record = q.record(slot)
            for key, kind in KINDS.items():
                self.pool.release(q.epoch, record[key], kind)
            q.storage[slot*128:(slot+1)*128] = bytes(128)
            q.free[q.free_count] = slot
            q.free_count += 1
            lost += 1
        for index in range(q.table_size):
            q.occurrences[index] = q.supports[index] = 0
        q.active = q.active_delivery = None
        q.heap_count = 0
        q.dropped += lost
        self.closed = True
        return lost
