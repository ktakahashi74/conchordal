"""Byte-exact sharing, independent finite ownership and real descriptor fixtures."""

from pathlib import Path
import random
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import temporal_shared_descriptor_reference as ref
from temporal_descriptor_reference import BoundedDescriptor, DescriptorKnot
from temporal_payload_reference import PayloadPool, OwnedEndpointQueue, KINDS
from temporal_matcher_reference import descriptor_query
from test_evaluate_temporal_descriptor_reference import raw
from test_evaluate_temporal_endpoint_reference import endpoint
from test_evaluate_temporal_query_scheduler_reference import span


def leaf(identity):
    return struct.pack('<Q', identity)+bytes(312)


class SharedDescriptorTests(unittest.TestCase):
    def test_default_fixed_layout_charges_vectors_metadata_indexes_and_scratch(self):
        p = ref.SharedDescriptorPool(1)
        sizes = p.payload_bytes()
        self.assertEqual(sizes['total_fixed_buffers'], 128713216)
        self.assertEqual(sizes['root_vectors'], 65536*128*4)
        self.assertEqual(sizes['knot_data'], 262144*320)
        self.assertEqual(sizes['hash_index'], 524288*4)

    def test_shifted_order_shares_identical_knots_without_changing_the_sequence(self):
        p = ref.SharedDescriptorPool(1, 3, 6, 4)
        sequences = ((1, 2, 3, 4), (1, 5, 3, 4), (5, 3, 4, 6))
        handles = [p.create(b''.join(leaf(i) for i in s)) for s in sequences]
        self.assertEqual(p.available_knots, 0)
        for h, s in zip(handles, sequences):
            self.assertEqual(p.read(1, h), b''.join(leaf(i) for i in s))
        p.release(1, handles[1])
        self.assertEqual(p.available_knots, 0)
        p.release(1, handles[0])
        self.assertEqual(p.available_knots, 2)
        self.assertEqual(p.read(1, handles[2]), b''.join(leaf(i) for i in sequences[2]))
        p.release(1, handles[2])
        self.assertEqual((p.available_roots, p.available_knots), (3, 6))

    def test_input_and_export_are_detached_and_preserve_every_byte(self):
        p = ref.SharedDescriptorPool(1, 4, 4, 1)
        original = bytearray(leaf(5))
        h = p.create(original)
        original[-1] = 1
        distinct = p.create(original)
        self.assertEqual(p.available_knots, 2)
        self.assertNotEqual(p.read(1, h), p.read(1, distinct))
        exported = bytearray(p.read(1, h))
        exported[0] = 9
        self.assertEqual(p.read(1, h), leaf(5))

    def test_multiple_root_owners_do_not_multiply_child_ownership(self):
        p = ref.SharedDescriptorPool(1, 2, 2, 3)
        h = p.create(leaf(1)*3)
        p.retain(1, h)
        other = p.create(leaf(1))
        self.assertEqual(p.available_knots, 1)
        self.assertFalse(p.release(1, h))
        self.assertEqual(p.read(1, h), leaf(1)*3)
        self.assertTrue(p.release(1, h))
        self.assertEqual(p.read(1, other), leaf(1))
        self.assertEqual(p.available_knots, 1)
        p.release(1, other)
        self.assertEqual(p.available_knots, 2)

    def test_hash_collisions_require_full_equality_and_survive_deletion(self):
        class ConstantHash:
            def digest(self):
                return bytes(16)
        p = ref.SharedDescriptorPool(1, 5, 5, 1)
        with patch.object(ref, 'blake2b', return_value=ConstantHash()):
            held = [p.create(leaf(i)) for i in range(4)]
            p.release(1, held[1])
            another = p.create(leaf(3))
            fresh = p.create(leaf(5))
            self.assertEqual(p.available_knots, 1)
            for h, i in ((held[0], 0), (held[2], 2), (held[3], 3), (another, 3), (fresh, 5)):
                self.assertEqual(p.read(1, h), leaf(i))
        self.assertGreater(p.maximum_probe, 1)

    def test_all_tombstone_index_remains_bounded_and_reusable(self):
        class SelectedHash:
            def __init__(self, at):
                self.at = at
            def digest(self):
                return self.at.to_bytes(16, 'little')
        p = ref.SharedDescriptorPool(1, 1, 2, 1)
        for at in range(p.table_size):
            with patch.object(ref, 'blake2b', return_value=SelectedHash(at)):
                h = p.create(leaf(at))
                p.release(1, h)
        self.assertTrue(all(v == ref.EMPTY for v in p.index))
        h = p.create(leaf(100))
        self.assertEqual(p.read(1, h), leaf(100))
        self.assertEqual(p.maximum_probe, p.table_size)

    def test_leaf_exhaustion_rolls_back_new_and_shared_prefix_without_losing_old_version(self):
        p = ref.SharedDescriptorPool(1, 3, 3, 4)
        original = p.create(leaf(1)+leaf(2))
        self.assertIsNone(p.create(leaf(1)+leaf(3)+leaf(4)))
        self.assertEqual((p.available_roots, p.available_knots), (2, 1))
        self.assertEqual(p.read(1, original), leaf(1)+leaf(2))
        self.assertEqual(p.peak_knots, 3)
        p.release(1, original)
        self.assertEqual((p.available_roots, p.available_knots), (3, 3))
        self.assertEqual(p.knot_allocations, p.knot_releases)

    def test_root_exhaustion_and_invalid_length_do_not_acquire_leaves(self):
        p = ref.SharedDescriptorPool(1, 1, 3, 2)
        h = p.create(leaf(1))
        before = p.leaf_owner_updates
        self.assertIsNone(p.create(leaf(2)))
        for bad in (b'', bytes(319), bytes(321), bytes(960)):
            with self.assertRaises(ValueError):
                p.create(bad)
        self.assertEqual(p.leaf_owner_updates, before)
        self.assertEqual(p.read(1, h), leaf(1))

    def test_root_reuse_rejects_retired_or_foreign_handles(self):
        p = ref.SharedDescriptorPool(1, 1, 1, 1)
        old = p.create(leaf(1))
        p.release(1, old)
        new = p.create(leaf(2))
        self.assertNotEqual(old, new)
        for fn in (p.read, p.retain, p.release):
            for args in ((1, old), (2, new), (1, new, 2)):
                with self.assertRaises(ValueError):
                    fn(*args)
        self.assertEqual(p.read(1, new), leaf(2))

    def test_handle_and_reference_count_exhaustion_do_not_wrap_or_leak(self):
        p = ref.SharedDescriptorPool(1, 2, 2, 2)
        p.serial = 2**64
        with self.assertRaises(OverflowError):
            p.create(leaf(1))
        self.assertEqual((p.available_roots, p.available_knots), (2, 2))
        p.serial = 0
        h = p.create(leaf(1))
        slot = (h-1) % p.roots
        struct.pack_into('<I', p.root_metadata, slot*24+12, 2**32-1)
        with self.assertRaises(OverflowError):
            p.retain(1, h)
        struct.pack_into('<I', p.root_metadata, slot*24+12, 1)
        k = p.root_knots[slot*p.max_knots]
        struct.pack_into('<I', p.knot_metadata, k*24+16, 2**32-1)
        with self.assertRaises(OverflowError):
            p.create(leaf(2)+leaf(1))
        self.assertEqual(p.available_knots, 1)
        struct.pack_into('<I', p.knot_metadata, k*24+16, 1)
        p.release(1, h)
        self.assertEqual(p.available_knots, 2)

    def test_independent_randomized_byte_union_and_owner_oracle(self):
        rng = random.Random(47021)
        for run in range(50):
            p = ref.SharedDescriptorPool(1, 6, 20, 8)
            live, retired = {}, set()
            for step in range(200):
                op = rng.choice(('new', 'retain', 'release'))
                if not live or op == 'new':
                    values = tuple(rng.randrange(30) for _ in range(rng.randrange(1, 9)))
                    union = {v for row, _ in live.values() for v in row} | set(values)
                    allowed = len(live) < 6 and len(union) <= 20
                    h = p.create(b''.join(leaf(v) for v in values))
                    self.assertEqual(h is not None, allowed, (run, step))
                    if h is not None:
                        self.assertNotIn(h, retired)
                        live[h] = [values, 1]
                else:
                    h = rng.choice(list(live))
                    if op == 'retain':
                        p.retain(1, h)
                        live[h][1] += 1
                    else:
                        live[h][1] -= 1
                        self.assertEqual(p.release(1, h), live[h][1] == 0)
                        if live[h][1] == 0:
                            retired.add(h)
                            del live[h]
                union = {v for values, _ in live.values() for v in values}
                self.assertEqual((p.available_roots, p.available_knots), (6-len(live), 20-len(union)))
                for h, (values, owners) in live.items():
                    self.assertEqual(p.owners(1, h), owners)
                    self.assertEqual(p.read(1, h), b''.join(leaf(v) for v in values))

    def test_actual_compressed_versions_preserve_moments_times_masks_and_generation(self):
        bank = BoundedDescriptor([1.]*10)
        pool = ref.SharedDescriptorPool(1, 80, 2048)
        snapshots = []
        for i in range(640):
            values = [None if (i+j) % 31 == 0 else float((i+3*j) % 19) for j in range(10)]
            bank.append(DescriptorKnot.from_raw(raw(i, values)))
            if (i+1) % 8 == 0:
                packed = bytes(bank.storage[:bank.count*320])
                snapshots.append((pool.create(packed), packed))
        self.assertGreater(bank.merges, 0)
        self.assertTrue(all(h is not None for h, _ in snapshots))
        self.assertLess(pool.knot_allocations, sum(len(data)//320 for _, data in snapshots))
        for h, packed in reversed(snapshots):
            self.assertEqual(pool.read(1, h), packed)
            pool.release(1, h)
        self.assertEqual((pool.available_roots, pool.available_knots), (80, 2048))

    def test_split_pool_delivers_original_packed_descriptor_through_owned_queue(self):
        d = ref.SharedDescriptorPool(1, 2, 8)
        a = PayloadPool(1, 6, 10)
        p = ref.SplitPayloadPool(d, a)
        packed = descriptor_query(span(), .1, 1, 1, 1001)['packed_knots']
        payloads = dict(descriptor_ref=packed, activity_ref=bytes(640), joint_ref=bytes(32), lineage_ref=bytes(8))
        refs = p.create_bundle(payloads)
        self.assertEqual(refs['descriptor_ref'], refs['activity_ref'])
        q = OwnedEndpointQueue(p, 2)
        ticket = q.offer(endpoint(1, **refs), .1)
        for key, h in refs.items():
            p.release(1, h, KINDS[key])
        self.assertEqual(q.payloads(q.take(0, .6)), payloads)
        q.finish(1, ticket, True, .6)
        self.assertEqual((d.available_roots, d.available_knots, a.available_slots, a.available_blocks), (2, 8, 6, 10))

    def test_split_bundle_failure_releases_only_its_new_owners(self):
        d = ref.SharedDescriptorPool(1, 3, 8)
        original = d.create(leaf(1))
        a = PayloadPool(1, 1, 10)
        p = ref.SplitPayloadPool(d, a)
        payloads = dict(descriptor_ref=leaf(1), activity_ref=bytes(640), joint_ref=bytes(32), lineage_ref=bytes(8))
        self.assertIsNone(p.create_bundle(payloads))
        self.assertEqual((d.available_roots, d.available_knots, a.available_slots), (2, 7, 1))
        self.assertEqual(d.read(1, original), leaf(1))
        d.release(1, original)
        self.assertEqual(d.available_knots, 8)

    def test_split_owner_retirement_releases_shared_roots_and_all_auxiliary_owners(self):
        d = ref.SharedDescriptorPool(1, 3, 8)
        a = PayloadPool(1, 9, 12)
        p = ref.SplitPayloadPool(d, a)
        q = OwnedEndpointQueue(p, 3)
        for i in range(3):
            payloads = dict(descriptor_ref=leaf(1)+leaf(i+2), activity_ref=bytes(640), joint_ref=bytes(32), lineage_ref=bytes(8))
            refs = p.create_bundle(payloads)
            q.offer(endpoint(i, **refs), .1)
            for key, h in refs.items():
                p.release(1, h, KINDS[key])
        q.take(0, .6)
        self.assertEqual(q.close(), 3)
        self.assertEqual((d.available_roots, d.available_knots, a.available_slots, a.available_blocks), (3, 8, 9, 12))


if __name__ == '__main__':
    unittest.main()
