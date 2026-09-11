"""Immutable descriptor versions sharing exact320-byte knots in bounded storage.

This representation preserves already constructed packed descriptors. It neither
infers common musical identity nor compresses their values further. Hash matches
always require full byte equality, including original timestamps and generations.
"""

from array import array
from hashlib import blake2b
import struct

from temporal_payload_reference import EMPTY, KINDS


class SharedDescriptorPool:
    """Fixed root vectors and interned immutable leaves, owned by one thread."""

    def __init__(self, epoch, roots=65536, knots=262144, max_knots=128):
        if (type(epoch) is not int or not 0 <= epoch < 2**64
                or any(type(n) is not int or not 0 < n < EMPTY for n in (roots, knots, max_knots))):
            raise ValueError('unsigned epoch and bounded positive descriptor capacities required')
        self.epoch, self.roots, self.knots, self.max_knots = epoch, roots, knots, max_knots
        self.root_metadata = bytearray(24*roots)
        self.root_knots = array('I', [0])*(roots*max_knots)
        self.free_roots = array('I', reversed(range(roots)))
        self.knot_data = bytearray(320*knots)
        self.knot_metadata = bytearray(24*knots)
        self.free_knots = array('I', reversed(range(knots)))
        self.table_size = 1 << (2*knots-1).bit_length()
        self.index = array('I', [0])*self.table_size
        self.scratch = array('I', [0])*max_knots
        if self.index.itemsize != 4:
            raise RuntimeError('u32 descriptor indexes required')
        self.available_roots, self.available_knots = roots, knots
        self.serial = 0
        self.capacity_losses = self.index_probes = self.maximum_probe = 0
        self.full_byte_comparisons = self.shared_leaf_hits = 0
        self.root_allocations = self.root_releases = 0
        self.knot_allocations = self.knot_releases = 0
        self.data_copy_bytes = self.vector_writes = self.leaf_owner_updates = 0
        self.peak_roots = self.peak_knots = 0

    def _root(self, epoch, handle, kind=1):
        if epoch != self.epoch or kind != 1 or type(handle) is not int or not 0 < handle < 2**64:
            raise ValueError('live epoch-qualified descriptor handle required')
        slot = (handle-1) % self.roots
        record = struct.unpack_from('<Q4I', self.root_metadata, slot*24)
        if record[0] != handle or record[2] == 0:
            raise ValueError('retired descriptor handle')
        return slot, record

    def create(self, packed):
        """Acquire an ordered descriptor version or roll back every acquired leaf."""
        if (not isinstance(packed, (bytes, bytearray)) or len(packed) % 320
                or not 0 < len(packed) <= self.max_knots*320):
            raise ValueError('one through max_knots complete320-byte records required')
        if not self.available_roots:
            self.capacity_losses += 1
            return None
        root = self.free_roots[self.available_roots-1]
        handle = self.serial*self.roots+root+1
        if handle >= 2**64:
            raise OverflowError('descriptor version exhaustion requires a new epoch')
        count, acquired, complete = len(packed)//320, 0, False
        try:
            for i in range(count):
                data = memoryview(packed)[i*320:(i+1)*320]
                digest = blake2b(data, digest_size=16).digest()
                at = int.from_bytes(digest[:8], 'little') & (self.table_size-1)
                reusable = slot = None
                for probe in range(1, self.table_size+1):
                    self.index_probes += 1
                    self.maximum_probe = max(self.maximum_probe, probe)
                    entry = self.index[at]
                    if entry == 0:
                        break
                    if entry == EMPTY:
                        if reusable is None:
                            reusable = at
                    else:
                        candidate = entry-1
                        if self.knot_metadata[candidate*24:candidate*24+16] == digest:
                            self.full_byte_comparisons += 1
                            if memoryview(self.knot_data)[candidate*320:(candidate+1)*320] == data:
                                slot = candidate
                                break
                    at = (at+1) & (self.table_size-1)
                else:
                    if reusable is None:
                        raise RuntimeError('descriptor hash-index occupancy invariant violated')
                if slot is None and reusable is not None:
                    at = reusable
                if slot is None:
                    if not self.available_knots:
                        self.capacity_losses += 1
                        return None
                    self.available_knots -= 1
                    slot = self.free_knots[self.available_knots]
                    self.knot_data[slot*320:(slot+1)*320] = data
                    struct.pack_into('<16s2I', self.knot_metadata, slot*24, digest, 1, at)
                    self.index[at] = slot+1
                    self.knot_allocations += 1
                    self.data_copy_bytes += 320
                    self.peak_knots = max(self.peak_knots, self.knots-self.available_knots)
                else:
                    owners = struct.unpack_from('<I', self.knot_metadata, slot*24+16)[0]
                    if owners == EMPTY:
                        raise OverflowError('shared-knot reference count cannot wrap')
                    struct.pack_into('<I', self.knot_metadata, slot*24+16, owners+1)
                    self.shared_leaf_hits += 1
                self.leaf_owner_updates += 1
                self.scratch[acquired] = slot
                acquired += 1
            self.root_knots[root*self.max_knots:root*self.max_knots+count] = self.scratch[:count]
            struct.pack_into('<Q4I', self.root_metadata, root*24, handle, count, 1, 0, 0)
            self.available_roots -= 1
            self.serial += 1
            self.root_allocations += 1
            self.vector_writes += count
            self.peak_roots = max(self.peak_roots, self.roots-self.available_roots)
            self.peak_knots = max(self.peak_knots, self.knots-self.available_knots)
            complete = True
            return handle
        finally:
            if not complete:
                for i in range(acquired):
                    slot = self.scratch[i]
                    owners, at = struct.unpack_from('<2I', self.knot_metadata, slot*24+16)
                    self.leaf_owner_updates += 1
                    if owners > 1:
                        struct.pack_into('<I', self.knot_metadata, slot*24+16, owners-1)
                    else:
                        self.index[at] = EMPTY
                        self.knot_data[slot*320:(slot+1)*320] = bytes(320)
                        self.knot_metadata[slot*24:(slot+1)*24] = bytes(24)
                        self.free_knots[self.available_knots] = slot
                        self.available_knots += 1
                        self.knot_releases += 1

    def retain(self, epoch, handle, kind=1):
        slot, record = self._root(epoch, handle, kind)
        if record[2] == EMPTY:
            raise OverflowError('descriptor owner count cannot wrap')
        struct.pack_into('<I', self.root_metadata, slot*24+12, record[2]+1)
        return handle

    def release(self, epoch, handle, kind=1):
        slot, record = self._root(epoch, handle, kind)
        if record[2] > 1:
            struct.pack_into('<I', self.root_metadata, slot*24+12, record[2]-1)
            return False
        for i in range(record[1]):
            leaf = self.root_knots[slot*self.max_knots+i]
            owners, at = struct.unpack_from('<2I', self.knot_metadata, leaf*24+16)
            if owners == 0:
                raise RuntimeError('live descriptor cannot reference a retired knot')
            self.leaf_owner_updates += 1
            if owners > 1:
                struct.pack_into('<I', self.knot_metadata, leaf*24+16, owners-1)
            else:
                self.index[at] = EMPTY
                self.knot_data[leaf*320:(leaf+1)*320] = bytes(320)
                self.knot_metadata[leaf*24:(leaf+1)*24] = bytes(24)
                self.free_knots[self.available_knots] = leaf
                self.available_knots += 1
                self.knot_releases += 1
            self.root_knots[slot*self.max_knots+i] = 0
        self.root_metadata[slot*24:(slot+1)*24] = bytes(24)
        self.free_roots[self.available_roots] = slot
        self.available_roots += 1
        self.root_releases += 1
        return True

    def read(self, epoch, handle, kind=1):
        slot, record = self._root(epoch, handle, kind)
        result = bytearray(record[1]*320)
        for i in range(record[1]):
            leaf = self.root_knots[slot*self.max_knots+i]
            result[i*320:(i+1)*320] = memoryview(self.knot_data)[leaf*320:(leaf+1)*320]
        self.data_copy_bytes += len(result)
        return bytes(result)

    def owners(self, epoch, handle):
        return self._root(epoch, handle)[1][2]

    def payload_bytes(self):
        parts = {'root_metadata': len(self.root_metadata), 'root_vectors': 4*len(self.root_knots),
                 'free_roots': 4*len(self.free_roots), 'knot_data': len(self.knot_data),
                 'knot_metadata': len(self.knot_metadata), 'free_knots': 4*len(self.free_knots),
                 'hash_index': 4*len(self.index), 'insertion_scratch': 4*len(self.scratch)}
        return {**parts, 'total_fixed_buffers': sum(parts.values()),
                'excludes': 'Python objects/scalars, hash temporaries, max_knots index-slice scratch, producer input, detached read copies, auxiliary payloads and other worker state.'}


class SplitPayloadPool:
    """Route descriptor ownership to shared knots and other kinds to byte storage."""

    def __init__(self, descriptors, auxiliary):
        if descriptors.epoch != auxiliary.epoch:
            raise ValueError('descriptor and auxiliary pool epochs must match')
        self.epoch = descriptors.epoch
        self.descriptors, self.auxiliary = descriptors, auxiliary

    def _pool(self, kind):
        if type(kind) is not int or kind not in KINDS.values():
            raise ValueError('explicit payload kind required for disjoint handle namespaces')
        return self.descriptors if kind == 1 else self.auxiliary

    def create_bundle(self, payloads):
        if set(payloads) != set(KINDS):
            raise ValueError('exact descriptor/activity/joint/lineage inventory required')
        acquired, complete = {}, False
        try:
            for key, kind in KINDS.items():
                h = (self.descriptors.create(payloads[key]) if kind == 1
                     else self.auxiliary.create(kind, payloads[key]))
                if h is None:
                    return None
                acquired[key] = h
            complete = True
            return acquired
        finally:
            if not complete:
                for key, h in acquired.items():
                    self.release(self.epoch, h, KINDS[key])

    def retain(self, epoch, handle, kind):
        return self._pool(kind).retain(epoch, handle, kind)

    def release(self, epoch, handle, kind):
        return self._pool(kind).release(epoch, handle, kind)

    def read(self, epoch, handle, kind):
        return self._pool(kind).read(epoch, handle, kind)

    def owners(self, epoch, handle, kind):
        pool = self._pool(kind)
        if kind != 1:
            pool._record(epoch, handle, kind)
        return pool.owners(epoch, handle)

    def payload_bytes(self):
        descriptors, auxiliary = self.descriptors.payload_bytes(), self.auxiliary.payload_bytes()
        return {'descriptors': descriptors, 'auxiliary': auxiliary,
                'total_fixed_buffers': descriptors['total_fixed_buffers']+auxiliary['total_fixed_buffers'],
                'excludes': 'Both component exclusions, adapter object and actual endpoint queue/consumer/worker state.'}
