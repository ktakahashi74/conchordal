"""Independent byte/ownership oracles and immutable endpoint lifecycle checks."""

from pathlib import Path
import random
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import temporal_payload_reference as ref
from temporal_matcher_reference import descriptor_query
from temporal_section_reference import SectionRecord
from test_evaluate_temporal_endpoint_reference import endpoint
from test_evaluate_temporal_query_scheduler_reference import span


def bundle(pool, **payloads):
    return {key: pool.create(kind, payloads.get(key, bytes([kind])*kind*100))
            for key, kind in ref.KINDS.items()}


class PayloadTests(unittest.TestCase):
    def test_default_layout_includes_all_fixed_indexes_and_metadata(self):
        p = ref.PayloadPool(1)
        sizes = p.payload_bytes()
        self.assertEqual(sizes['total_fixed_buffers'], 87818240)
        self.assertEqual(sizes['object_metadata'], 65536*24)
        self.assertEqual(sizes['blocks'], 262144*320)

    def test_block_boundaries_copy_input_and_detach_reads(self):
        for length in (1, 319, 320, 321, 40960, 65536):
            p = ref.PayloadPool(1, 1, 205)
            original = bytearray(i % 251 for i in range(length))
            expected = bytes(original)
            h = p.create(1, original)
            original[:] = bytes(length)
            self.assertEqual(p.read(1, h, 1), expected)
            self.assertEqual(p.available_blocks, 205-(length+319)//320)
            self.assertTrue(p.release(1, h))
            self.assertEqual((p.available_blocks, p.available_slots), (205, 1))
            self.assertFalse(any(p.storage))

    def test_fragmented_multiblock_payload_survives_unrelated_slot_reuse(self):
        p = ref.PayloadPool(1, 6, 6)
        held = [p.create(1, bytes([i+1])*320) for i in range(6)]
        for i in (0, 2, 4):
            p.release(1, held[i])
        new = p.create(2, b'abcd'*240)
        self.assertEqual(p.read(1, new), b'abcd'*240)
        for i in (1, 3, 5):
            self.assertEqual(p.read(1, held[i]), bytes([i+1])*320)
        for i in (0, 2, 4):
            with self.assertRaises(ValueError):
                p.read(1, held[i])

    def test_last_owner_releases_storage_and_stale_handle_cannot_touch_replacement(self):
        p = ref.PayloadPool(1, 1, 2)
        first = p.create(1, b'old')
        p.retain(1, first)
        self.assertFalse(p.release(1, first))
        self.assertEqual(p.read(1, first), b'old')
        self.assertTrue(p.release(1, first))
        fresh = p.create(1, b'new')
        self.assertNotEqual(first, fresh)
        for fn in (p.read, p.release, p.retain):
            with self.assertRaises(ValueError):
                fn(1, first)
            with self.assertRaises(ValueError):
                fn(2, fresh)
            with self.assertRaises(ValueError):
                fn(1, fresh, 2)
        self.assertEqual(p.read(1, fresh), b'new')
        self.assertEqual(p.owners(1, fresh), 1)

    def test_capacity_and_oversize_rejection_leave_live_bytes_and_ownership_unchanged(self):
        for slots, blocks in ((1, 10), (10, 1)):
            p = ref.PayloadPool(1, slots, blocks, max_payload=321)
            h = p.create(1, b'old')
            before = (bytes(p.storage), bytes(p.metadata), p.serial)
            self.assertIsNone(p.create(1, b'new'*107))
            self.assertEqual((bytes(p.storage), bytes(p.metadata), p.serial), before)
            for payload in (b'', bytes(322)):
                with self.assertRaises(ValueError):
                    p.create(1, payload)
            self.assertEqual(p.owners(1, h), 1)
            self.assertEqual(p.capacity_losses, 1)

    def test_identity_and_reference_count_exhaustion_do_not_wrap(self):
        p = ref.PayloadPool(1, 1, 1)
        p.serial = 2**64-1
        with self.assertRaises(OverflowError):
            p.create(1, b'x')
        self.assertEqual((p.available_slots, p.available_blocks), (1, 1))
        p.serial = 0
        h = p.create(1, b'x')
        struct.pack_into('<I', p.metadata, 16, 2**32-1)
        with self.assertRaises(OverflowError):
            p.retain(1, h)
        self.assertEqual(p.owners(1, h), 2**32-1)

    def test_original_identity_pairs_survive_aliases_and_require_epoch_on_exhaustion(self):
        ids = ref.OccurrenceIds(1)
        first = ids.allocate()
        alias = dict(first)
        second = ids.allocate()
        self.assertEqual(first, alias)
        self.assertEqual((first['occurrence_id'], first['support_id']), (1, 1))
        self.assertEqual((second['occurrence_id'], second['support_id']), (2, 2))
        ids.next_id = 2**64
        with self.assertRaises(OverflowError):
            ids.allocate()
        self.assertNotEqual(ref.OccurrenceIds(2).allocate(), first)

    def test_randomized_finite_byte_and_owner_oracle(self):
        rng = random.Random(47020)
        for run in range(50):
            p = ref.PayloadPool(1, 8, 19, 1600)
            objects, stale = {}, []
            for step in range(200):
                op = rng.choice(('new', 'retain', 'release', 'read'))
                if not objects or op == 'new':
                    value = rng.randbytes(rng.randrange(1, 1601))
                    kind = rng.randrange(1, 5)
                    used = sum((len(v[0])+319)//320 for v in objects.values())
                    allowed = len(objects) < 8 and used+(len(value)+319)//320 <= 19
                    h = p.create(kind, value)
                    self.assertEqual(h is not None, allowed, (run, step))
                    if h is not None:
                        self.assertNotIn(h, stale)
                        objects[h] = [value, 1, kind]
                else:
                    h = rng.choice(list(objects))
                    if op == 'retain':
                        p.retain(1, h)
                        objects[h][1] += 1
                    elif op == 'release':
                        objects[h][1] -= 1
                        self.assertEqual(p.release(1, h), objects[h][1] == 0)
                        if objects[h][1] == 0:
                            del objects[h]
                            stale.append(h)
                for h, (value, owners, kind) in objects.items():
                    self.assertEqual(p.read(1, h, kind), value)
                    self.assertEqual(p.owners(1, h), owners)
                self.assertEqual(p.available_slots, 8-len(objects))
                self.assertEqual(p.available_blocks, 19-sum((len(v[0])+319)//320 for v in objects.values()))

    def test_two_endpoints_share_bytes_and_hold_after_producer_releases(self):
        p = ref.PayloadPool(1, 4, 10)
        refs = bundle(p)
        expected = {key: p.read(1, h) for key, h in refs.items()}
        q = ref.OwnedEndpointQueue(p, 2)
        for identity in (1, 2):
            q.offer(endpoint(identity, **refs), .1)
        self.assertEqual(p.allocations, 4)
        for h in refs.values():
            self.assertEqual(p.owners(1, h), 3)
            p.release(1, h)
        first = q.take(0, .6)
        self.assertEqual(q.payloads(first), expected)
        q.finish(1, first['ticket'], True, .6)
        self.assertEqual(p.available_slots, 0)
        second = q.take(0, .6)
        self.assertEqual(q.payloads(second), expected)
        q.finish(1, second['ticket'], False, .6)
        self.assertEqual((p.available_slots, p.available_blocks), (4, 10))
        self.assertEqual((q.queue.committed, q.queue.dropped), (1, 1))

    def test_failed_offer_and_wrong_kind_roll_back_all_pins(self):
        p = ref.PayloadPool(1, 4, 10)
        refs = bundle(p)
        q = ref.OwnedEndpointQueue(p, 1)
        original = endpoint(1, **refs)
        wrong = dict(original, lineage_ref=refs['joint_ref'])
        with self.assertRaises(ValueError):
            q.offer(wrong, .1)
        self.assertTrue(all(p.owners(1, h) == 1 for h in refs.values()))
        q.offer(original, .1)
        self.assertIsNone(q.offer(endpoint(2, **refs), .1))
        with self.assertRaises(ValueError):
            q.offer(endpoint(3, support_id=original['support_id'], **refs), .1)
        self.assertTrue(all(p.owners(1, h) == 2 for h in refs.values()))

    def test_revision_releases_old_versions_but_original_alias_never_repins_them(self):
        p = ref.PayloadPool(1, 8, 20)
        refs = bundle(p)
        q = ref.OwnedEndpointQueue(p, 1)
        original = endpoint(1, **refs)
        ticket = q.offer(original, .1)
        for h in refs.values():
            p.release(1, h)
        activity = p.create(2, b'updated activity')
        joint = p.create(3, b'updated joint')
        q.revise(1, 1, ticket, 3, activity, joint, .2, .3)
        p.release(1, activity)
        p.release(1, joint)
        for key in ('activity_ref', 'joint_ref'):
            with self.assertRaises(ValueError):
                p.read(1, refs[key])
        self.assertEqual(q.offer(original, .3), ticket)
        self.assertEqual(p.owners(1, activity), 1)
        self.assertFalse(q.revise(1, 1, ticket, 1, refs['activity_ref'], refs['joint_ref'], .2, .3))
        claim = q.take(0, .6)
        self.assertEqual(q.payloads(claim)['activity_ref'], b'updated activity')
        q.finish(1, ticket, True, .6)
        self.assertEqual(p.available_slots, 8)

    def test_rejected_revision_rolls_back_pins_and_keeps_claim_immutable(self):
        p = ref.PayloadPool(1, 8, 20)
        refs = bundle(p)
        q = ref.OwnedEndpointQueue(p, 1)
        ticket = q.offer(endpoint(1, **refs), .1)
        a, j = p.create(2, b'a'), p.create(3, b'j')
        for args in ((1, 1, ticket, 1, a, j, .2, .7), (1, 1, ticket, 1, a, a, .2, .3)):
            with self.assertRaises(ValueError):
                q.revise(*args)
            self.assertEqual((p.owners(1, a), p.owners(1, j)), (1, 1))
        claim = q.take(0, .6)
        with self.assertRaises(ValueError):
            q.revise(1, 1, ticket, 1, a, j, .3, .6)
        self.assertEqual((p.owners(1, a), p.owners(1, j)), (1, 1))
        self.assertEqual(q.take(0, .6), claim)

    def test_stale_ack_and_detached_claim_cannot_release_or_read_new_active_payload(self):
        p = ref.PayloadPool(1, 4, 10)
        refs = bundle(p)
        q = ref.OwnedEndpointQueue(p, 1)
        first = q.offer(endpoint(1, **refs), .1)
        old = q.take(0, .6)
        q.finish(1, first, True, .6)
        second = q.offer(endpoint(2, .6, .7, **refs), .7)
        fresh = q.take(1, 1.2)
        self.assertIsNone(q.finish(1, first, True, 1.2))
        self.assertIsNone(q.finish(2, second, True, 1.2))
        with self.assertRaises(ValueError):
            q.payloads(old)
        with self.assertRaises(ValueError):
            q.payloads(dict(fresh, packed=bytes(128)))
        self.assertTrue(all(p.owners(1, h) == 2 for h in refs.values()))

    def test_actual_frozen_descriptor_and_section_record_round_trip(self):
        s = span()
        query = descriptor_query(s, .1, 1, 1, 1001)
        activity = SectionRecord()
        activity.add('assignment_seconds', [.1])
        activity.add('physical_window_seconds', [.1])
        activity.ending([1., None, .3, .4, .5, .6])
        p = ref.PayloadPool(1, 4, 140)
        refs = bundle(p, descriptor_ref=query['packed_knots'], activity_ref=activity.buffer,
                      joint_ref=struct.pack('<3Qd', 1, 10, 20, .75), lineage_ref=struct.pack('<Q', 4))
        q = ref.OwnedEndpointQueue(p, 1)
        ticket = q.offer(endpoint(1, **refs), .1)
        for h in refs.values():
            p.release(1, h)
        activity.ending([None]*6)
        out = q.payloads(q.take(0, 1.))
        self.assertEqual(out['descriptor_ref'], query['packed_knots'])
        restored = SectionRecord(out['activity_ref'])
        self.assertEqual(restored.ending(), [1., None, .3, .4, .5, .6])
        self.assertEqual(restored.block('assignment_seconds'), [.1])
        self.assertEqual(struct.unpack('<3Qd', out['joint_ref']), (1, 10, 20, .75))
        q.finish(1, ticket, True, 1.)
        self.assertEqual(p.available_blocks, 140)

    def test_epoch_retirement_releases_active_and_pending_without_consumer_commits(self):
        p = ref.PayloadPool(1, 4, 10)
        refs = bundle(p)
        q = ref.OwnedEndpointQueue(p, 3)
        for identity in range(3):
            q.offer(endpoint(identity, **refs), .1)
        for h in refs.values():
            p.release(1, h)
        active = q.take(0, .6)
        self.assertEqual(q.close(), 3)
        self.assertEqual(q.close(), 0)
        self.assertEqual((p.available_slots, p.available_blocks), (4, 10))
        self.assertEqual((q.queue.committed, q.queue.dropped, q.queue.free_count), (0, 3, 3))
        with self.assertRaises(ValueError):
            q.payloads(active)
        with self.assertRaises(ValueError):
            q.finish(1, active['ticket'], True, .6)
        with self.assertRaises(ValueError):
            q.take(1, .7)

    def test_partial_bundle_failure_rolls_back_and_never_reuses_consumed_identities(self):
        payloads = {key: bytes([kind])*320 for key, kind in ref.KINDS.items()}
        for slots, blocks in ((3, 8), (8, 3)):
            p = ref.PayloadPool(1, slots, blocks)
            self.assertIsNone(p.create_bundle(payloads))
            self.assertEqual((p.available_slots, p.available_blocks), (slots, blocks))
            self.assertEqual((p.allocations, p.releases, p.capacity_losses), (3, 3, 1))
            self.assertGreater(p.create(1, b'new'), slots*2)
        p = ref.PayloadPool(1, 4, 4)
        with self.assertRaises(ValueError):
            p.create_bundle(dict(payloads, lineage_ref=b''))
        self.assertEqual((p.available_slots, p.available_blocks), (4, 4))
        refs = p.create_bundle(payloads)
        self.assertEqual({key: p.read(1, h) for key, h in refs.items()}, payloads)


if __name__ == '__main__':
    unittest.main()
