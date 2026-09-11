"""Independent sorted endpoints, fixed-index churn and finite-ledger comparisons."""

import copy
from pathlib import Path
import random
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import temporal_endpoint_reference as ref
from temporal_occurrence_reference import OccurrenceLedger
from test_evaluate_temporal_occurrence_reference import record, paths, hop


def endpoint(identity, start=0., end=.1, **extra):
    return {'epoch': 1, 'occurrence_id': identity, 'support_id': identity+1000,
            'generation': 4, 'descriptor_ref': identity+2001, 'activity_ref': identity+3001,
            'joint_ref': identity+4001, 'lineage_ref': 5001, 'flags': 0,
            'start': start, 'support_end': end, 'source_available_end': end, **extra}


def unpack(claim):
    return dict(zip(ref.INTEGER_FIELDS+ref.TIME_FIELDS, struct.unpack('<10Q6d', claim['packed'])))


class EndpointTests(unittest.TestCase):
    def test_default_fixed_layout_charges_heap_free_list_and_both_indexes(self):
        q = ref.EndpointQueue(1)
        self.assertEqual(q.payload_bytes()['endpoint_headers'], 8388608)
        self.assertEqual(q.payload_bytes()['heap'], 262144)
        self.assertEqual(q.payload_bytes()['free_slots'], 262144)
        self.assertEqual(q.payload_bytes()['id_indexes'], 1048576)
        self.assertEqual(q.payload_bytes()['total_fixed_buffers'], 9961472)

    def test_aliases_share_one_ticket_and_cannot_change_original_support(self):
        q = ref.EndpointQueue(1, 2)
        original = endpoint(3)
        ticket = q.offer(original, .1)
        self.assertEqual(q.offer(copy.deepcopy(original), .2), ticket)
        self.assertEqual((q.heap_count, q.free_count), (1, 1))
        for data in [{**original, 'support_end': .11, 'source_available_end': .11},
                     {**original, 'descriptor_ref': 999}, endpoint(4, support_id=1003)]:
            with self.assertRaises(ValueError):
                q.offer(data, .2)
        self.assertEqual(q.heap_count, 1)

    def test_deadline_and_actual_delivery_are_distinct_and_claim_is_frozen(self):
        q = ref.EndpointQueue(1, 2)
        ticket = q.offer(endpoint(1), .2)
        self.assertIsNone(q.take(0, .5))
        claim = q.take(0, 1.)
        self.assertEqual(unpack(claim)['deadline'], .6)
        self.assertEqual(claim['delivered_at'], 1.)
        self.assertEqual(q.take(1, 1.1), claim)
        self.assertEqual(q.free_count, 1)
        done = q.finish(1, ticket, True, 1.1)
        self.assertEqual(done['release_refs'], (2002, 3002, 4002, 5001))
        self.assertEqual(q.free_count, 2)
        self.assertEqual(q.committed, 1)
        self.assertEqual(unpack(claim)['occurrence_id'], 1)

    def test_revision_changes_only_mutable_refs_and_keeps_original_deadline(self):
        q = ref.EndpointQueue(1, 2)
        original = endpoint(1)
        ticket = q.offer(original, .1)
        changed = q.revise(1, 1, ticket, 1, 6001, 7001, .3, .4)
        self.assertEqual(changed['release_refs'], {'activity_ref': 3002, 'joint_ref': 4002})
        self.assertEqual(q.offer(original, .4), ticket)
        self.assertFalse(q.revise(1, 1, ticket, 1, 6001, 7001, .3, .4))
        with self.assertRaises(ValueError):
            q.revise(1, 1, ticket, 1, 6002, 7001, .3, .4)
        claim = q.take(0, .6)
        r = unpack(claim)
        self.assertEqual((r['revision'], r['activity_ref'], r['joint_ref']), (1, 6001, 7001))
        self.assertEqual((r['start'], r['support_end'], r['deadline']), (0., .1, .6))
        with self.assertRaises(ValueError):
            q.revise(1, 1, ticket, 2, 6002, 7002, .3, .6)
        self.assertEqual(q.take(0, .6), claim)

    def test_expired_or_future_evidence_cannot_revise_a_pending_endpoint(self):
        q = ref.EndpointQueue(1, 2)
        ticket = q.offer(endpoint(1), .1)
        before = bytes(q.storage)
        for source, receipt in ((.5, .4), (.3, .7), (.05, .2)):
            with self.assertRaises(ValueError):
                q.revise(1, 1, ticket, 1, 9, 10, source, receipt)
            self.assertEqual(bytes(q.storage), before)

    def test_capacity_loss_does_not_evict_pending_or_active_endpoints(self):
        q = ref.EndpointQueue(1, 1)
        ticket = q.offer(endpoint(1), .1)
        self.assertIsNone(q.offer(endpoint(2, .1, .2), .2))
        self.assertEqual(q.capacity_losses, 1)
        claim = q.take(0, .6)
        self.assertIsNone(q.offer(endpoint(3, .2, .3), .6))
        self.assertEqual(q.take(0, .6), claim)
        q.finish(1, ticket, False, .6)
        self.assertEqual((q.committed, q.dropped), (0, 1))
        self.assertEqual(q.free_count, 1)

    def test_seal_cut_rejects_replayed_endings_after_slot_release(self):
        q = ref.EndpointQueue(1, 1)
        original = endpoint(1)
        ticket = q.offer(original, .1)
        q.take(0, .6); q.finish(1, ticket, True, .6)
        self.assertIsNone(q.offer(original, .7))
        self.assertIsNone(q.offer(endpoint(2, .01, .15), .7))
        self.assertEqual(q.late_losses, 2)
        self.assertEqual(q.free_count, 1)

    def test_slot_reuse_and_new_epoch_reject_stale_tickets(self):
        q = ref.EndpointQueue(1, 1)
        a = q.offer(endpoint(1), .1); q.take(0, .6); q.finish(1, a, True, .6)
        b = q.offer(endpoint(2, .6, .7), .7)
        self.assertGreater(b, a)
        self.assertFalse(q.revise(1, 2, a, 1, 9, 10, .7, .7))
        claim = q.take(1, 1.2)
        self.assertIsNone(q.finish(1, a, True, 1.2))
        self.assertEqual(q.take(1, 1.2), claim)
        fresh = ref.EndpointQueue(2, 1)
        self.assertIsNone(fresh.offer(endpoint(1), .1))
        self.assertFalse(fresh.revise(1, 1, a, 1, 9, 10, .1, .1))
        self.assertIsNone(fresh.finish(1, a, True, .1))

    def test_failed_commit_does_not_skip_downstream_sequence(self):
        q = ref.EndpointQueue(1, 2)
        q.offer(endpoint(1), .1); q.offer(endpoint(2, .1, .2), .2)
        first = q.take(0, 1.)
        self.assertEqual(first['sequence'], 1)
        failure = q.finish(1, first['ticket'], False, 1.)
        self.assertEqual(failure['reason'], 'unsealed_computational_loss')
        second = q.take(0, 1.)
        self.assertEqual(second['sequence'], 1)
        q.finish(1, second['ticket'], True, 1.)
        self.assertEqual((q.committed, q.dropped), (1, 1))

    def test_write_budget_counts_carried_active_claim_and_leaves_backlog(self):
        q = ref.EndpointQueue(1, 4, writes_per_cycle=2)
        for identity in range(4):
            q.offer(endpoint(identity), .1)
        first = q.take(0, .6)
        self.assertEqual(q.take(1, .7), first)
        q.finish(1, first['ticket'], True, .7)
        second = q.take(1, .7); q.finish(1, second['ticket'], True, .7)
        self.assertIsNone(q.take(1, .7))
        self.assertEqual(q.heap_count, 2)
        with self.assertRaises(ValueError):
            q.take(2, .7)
        self.assertIsNotNone(q.take(2, .8))

    def test_full_tombstone_indexes_remain_bounded_and_reusable(self):
        q = ref.EndpointQueue(1, 2)
        for identity in range(20):
            end = identity+1.
            ticket = q.offer(endpoint(identity, end-.1, end), end)
            claim = q.take(identity, end+.5)
            self.assertEqual(claim['ticket'], ticket)
            q.finish(1, ticket, True, end+.5)
        self.assertEqual(q.committed, 20)
        self.assertLessEqual(q.maximum_probe, q.table_size)
        self.assertEqual(q.free_count, 2)

    def test_sorted_order_matches_independent_complete_history_with_ties_and_collisions(self):
        rng = random.Random(47019)
        for capacity in (4, 8, 32):
            for _ in range(20):
                q = ref.EndpointQueue(1, capacity)
                rows = []
                for identity in rng.sample(range(100), capacity):
                    end = rng.choice([.3, .4, .5])
                    start = rng.choice([0., .1, .2])
                    data = endpoint(identity*q.table_size, start, end)
                    rows.append(data)
                    q.offer(data, .5)
                expected = sorted(rows, key=lambda r: (r['support_end'], r['start'], r['occurrence_id']))
                actual = []
                while (claim := q.take(0, 1.)) is not None:
                    actual.append(unpack(claim)['occurrence_id'])
                    q.finish(1, claim['ticket'], True, 1.)
                self.assertEqual(actual, [r['occurrence_id'] for r in expected])
                self.assertEqual(q.free_count, capacity)
                self.assertLessEqual(q.maximum_probe, q.table_size)

    def test_queue_matches_finite_ledger_commit_order_deadline_and_delivery(self):
        q = ref.EndpointQueue(1, 4)
        ledger = OccurrenceLedger(1)
        inputs = [(3, .1, .4), (2, .1, .3), (1, 0., .3)]
        for identity, start, end in inputs:
            original = record(identity, start, end, 1.)
            ledger.stage(original, paths((10, 10, 1.)), .5)
            q.offer(endpoint(identity, start, end, support_id=original['support_id']), .5)
        writes = ledger.advance(1.2, [hop(.3, 1.2)])
        for write in writes:
            claim = q.take(0, 1.2); saved = unpack(claim)
            self.assertEqual(saved['occurrence_id'], write['occurrence_id'])
            self.assertEqual(saved['support_id'], write['support_id'])
            self.assertEqual(saved['deadline'], write['committed_at'])
            self.assertEqual(claim['sequence'], write['sequence'])
            self.assertEqual(claim['delivered_at'], write['delivered_at'])
            q.finish(1, claim['ticket'], True, 1.2)
        self.assertIsNone(q.take(0, 1.2))


if __name__ == '__main__':
    unittest.main()
