"""Real bank/section side effects, interrupted receipts and independent sums."""

import copy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import temporal_consumer_reference as ref
from temporal_consumer_packet_reference import ConsumerPacketBuilder, pack_knots
from temporal_occurrence_reference import OccurrenceLedger
from temporal_section_reference import SectionHistory, SectionRecord, section_activity
from temporal_payload_reference import PayloadPool, OwnedEndpointQueue, KINDS
from temporal_matcher_reference import descriptor_query
from test_evaluate_temporal_bank_reference import bank, admissions
from test_evaluate_temporal_memory_reference import clock, raw
from test_evaluate_temporal_occurrence_reference import record, paths, hop
from test_evaluate_temporal_query_scheduler_reference import span
from test_evaluate_temporal_endpoint_reference import endpoint


class InterruptedSection(SectionHistory):
    def _apply_prepared(self, *args):
        self.calls += 1
        if self.failure == 'before':
            self.failure = None
            raise RuntimeError('interrupted before section write')
        if self.failure == 'false_success':
            return True
        changed = super()._apply_prepared(*args)
        if self.failure == 'after':
            self.failure = None
            raise RuntimeError('section acknowledgment lost after application')
        return changed


def setup(weights=(.5, .5)):
    b = bank(2, c=clock(6))
    h = b.reserve()
    histories = {}
    for i in range(len(weights)):
        history = InterruptedSection(1, 4, i+100, 0.)
        history.failure, history.calls = None, 0
        history.observe(section_activity([hop(0., .6)], [], 1, {4}, 0., .6), 1, 4)
        histories[i+100] = history
    pool = PayloadPool(1, 8, 300)
    queue = OwnedEndpointQueue(pool, 4)
    consumer = ref.EndpointConsumer(queue, b, histories, capacity=max(1, len(histories)))
    return consumer, pool, h


def packet(consumer, pool, handle, identity=1, index=0, weights=(.5, .5), admission=True):
    start, end = index/10, (index+1)/10
    original = record(identity, start, end, 1.)
    rows = paths(*[(handle, handle, w) for w in weights]) if weights else paths((handle, handle, 1.))
    ledger = OccurrenceLedger(1)
    ledger.stage(original, rows, end)
    packed = descriptor_query(span(index), end, identity, identity, identity+1000)['packed_knots']
    activity = SectionRecord()
    for key in ('numerators', 'denominators', 'physical_valid_seconds'):
        activity.add(key, original['activity'][key])
    activity.add('assignment_seconds', [original['activity']['assignment_seconds']])
    activity.add('physical_window_seconds', [original['activity']['physical_window_seconds']])
    activity.ending(original['ending_descriptor'])
    # Joint decoding is caller-owned in this interface; the byte codec is separate.
    refs = pool.create_bundle(dict(descriptor_ref=packed, activity_ref=activity.buffer,
                                   joint_ref=b'caller-owned decoded joint', lineage_ref=b'generation4'))
    consumer.queue.offer(endpoint(identity, start, end, **refs), end)
    for key, value in refs.items():
        pool.release(1, value, KINDS[key])
    claim = consumer.queue.take(identity, end+.5)
    w = ledger.advance(end+.5, [hop(end, end+.5)])[0]
    w['sequence'] = claim['sequence']
    projections = []
    for i, target in enumerate(consumer.sections):
        p = ledger.section_record(identity, i)
        p['sequence'] = w['sequence']
        projections.append(dict(target=target, path_id=i, **p))
    return claim, {'write': w, 'admissions': admissions(w, [handle]) if admission else {},
                   'relations': (), 'sections': projections}


def freeze(claim, batch):
    builder = ConsumerPacketBuilder(claim, {'write':batch['write']},
                                    len(batch['sections']), len(batch['admissions']), len(batch['relations']))
    for row in batch['relations']:
        builder.append_relation(row)
    for handle, episode in batch['admissions'].items():
        builder.append_admission(handle, {k:v for k,v in episode.items() if k != 'knots'},
                                 pack_knots(episode['knots']))
    for projection in batch['sections']:
        builder.append(projection)
    return builder.finish()


def prepare(consumer, claim, batch, cut=None, phase=ref.SECTIONS):
    cut = claim['delivered_at'] if cut is None else cut
    packet = freeze(claim, batch)
    consumer.begin(claim, packet, cut)
    while consumer._state()['phase'] != phase:
        consumer.advance(claim, packet, cut)
    return packet


class ConsumerTests(unittest.TestCase):
    def test_observations_between_preflight_and_each_delivery_survive_the_barrier(self):
        c, pool, h = setup()
        claim, native = packet(c, pool, h)
        sealed = freeze(claim, native)
        serial = copy.deepcopy(c.sections)
        self.assertFalse(c.advance(claim, sealed, .6, target_budget=1))
        accent = dict(epoch=1, generation=4, event_interval=[.6, .65], time=.65,
                      raw_support_end=.67, available_end=.7, weight=.625)
        for lo, hi in ((.6, .8), (.8, 1.)):
            for i in range(round(lo*10), round(hi*10)):
                c.bank.memory.clock.observe(raw(i), (i+1)/10)
            delta = section_activity([hop(lo, hi)], [], 1, {4}, lo, hi)
            delivery = dict(sequence=1, accent=accent, weight=.625, delivered_at=(lo+hi)/2)
            for history in (*c.sections.values(), *serial.values()):
                history.observe(delta, 1, 4, [delivery])
            if lo == .6:
                while c._state()['phase'] != ref.SECTIONS:
                    c.advance(claim, sealed, hi, target_budget=1, bank_budget=1)
                self.assertFalse(c.advance(claim, sealed, hi, target_budget=1))
                self.assertFalse(c.publishable())
        self.assertTrue(c.advance(claim, sealed, 1., target_budget=1))
        for projection in native['sections']:
            target = projection['target']
            serial[target].commit(projection['record'], projection['activity'],
                                  projection['sequence'], projection['adjacency_coverage'])
            self.assertEqual(c.sections[target].cumulative.buffer, serial[target].cumulative.buffer)
            self.assertEqual([r.buffer for r in c.sections[target].ring], [r.buffer for r in serial[target].ring])
            self.assertEqual(c.sections[target].cumulative.block('numerators')[4], .625)
            self.assertEqual(c.sections[target].calls, 1)
        self.assertEqual(pool.available_slots, 8)

    def test_each_section_is_decoded_and_prepared_once_across_receipt_recovery(self):
        from unittest.mock import patch
        from temporal_consumer_packet_reference import ConsumerPacket
        c, pool, h = setup()
        claim, native = packet(c, pool, h)
        sealed = freeze(claim, native)
        decode, prepare_commit = ConsumerPacket.section, SectionHistory._prepare_commit
        decoded, prepared = [], []
        def counted_decode(packet, index):
            decoded.append(index)
            return decode(packet, index)
        def counted_prepare(history, *args):
            prepared.append(history.cumulative.integer(2))
            return prepare_commit(history, *args)
        c.sections[101].failure = 'after'
        with patch.object(ConsumerPacket, 'section', counted_decode), \
                patch.object(SectionHistory, '_prepare_commit', counted_prepare):
            for _ in range(16):
                try:
                    done = c.advance(claim, sealed, .6, target_budget=1, bank_budget=1)
                except RuntimeError as error:
                    self.assertIn('acknowledgment lost', str(error))
                    continue
                if done:
                    break
        self.assertTrue(done)
        self.assertEqual(decoded, [0, 1])
        self.assertEqual(prepared, [100, 101])
        self.assertEqual([h.calls for h in c.sections.values()], [1, 1])
        self.assertEqual(c.section_receipt_recoveries, 1)
        self.assertEqual(pool.available_slots, 8)

    def test_terminal_preparation_cannot_leak_into_a_following_zero_weight_target(self):
        for abort_first in (False, True):
            with self.subTest(abort_first=abort_first):
                c, pool, h = setup()
                claim, native = packet(c, pool, h)
                sealed = prepare(c, claim, native, phase=ref.BANK)
                self.assertTrue(any(c.prepared_sections))
                if abort_first:
                    c.abort(claim, sealed, .6)
                else:
                    while not c.advance(claim, sealed, .6):
                        pass
                for i in range(6, 12):
                    c.bank.memory.clock.observe(raw(i), (i+1)/10)
                for history in c.sections.values():
                    history.observe(section_activity([hop(.6, 1.2)], [], 1, {4}, .6, 1.2), 1, 4)
                unused = c.sections[101]
                before = bytes(unused.cumulative.buffer), [bytes(r.buffer) for r in unused.ring], unused.calls
                following, native = packet(c, pool, h, identity=2, index=6, weights=(1., 0.), admission=abort_first)
                sealed = freeze(following, native)
                while not c.advance(following, sealed, 1.2):
                    pass
                self.assertEqual((bytes(unused.cumulative.buffer), [bytes(r.buffer) for r in unused.ring], unused.calls), before)
                self.assertEqual(c.queue.queue.committed, 1 if abort_first else 2)
                self.assertEqual(pool.available_slots, 8)

    def test_last_section_nonfinite_statistic_is_found_before_bank_effects(self):
        c, pool, h = setup()
        claim, native = packet(c, pool, h)
        native['sections'][-1]['activity']['numerators'][4] = float('inf')
        sealed = freeze(claim, native)
        before = [bytes(v.cumulative.buffer) for v in c.sections.values()]
        self.assertFalse(c.advance(claim, sealed, .6, target_budget=1))
        with self.assertRaises(ValueError):
            c.advance(claim, sealed, .6, target_budget=1)
        self.assertEqual(c.bank.memory.sequence, 0)
        self.assertEqual([bytes(v.cumulative.buffer) for v in c.sections.values()], before)
        self.assertEqual(sum(v.calls for v in c.sections.values()), 0)
        self.assertTrue(c.abort(claim, sealed, .6))
        self.assertTrue(c.publishable())

    def test_default_fixed_receipt_layout_and_distinct_consumer_ownership(self):
        c, pool, h = setup()
        default = ref.EndpointConsumer(c.queue, c.bank, c.sections)
        self.assertEqual(default.payload_bytes()['total_fixed_buffers'], 1474816)
        self.assertTrue(c.publishable())
        with self.assertRaises(ValueError):
            ref.EndpointConsumer(c.queue, c.bank, {1: c.sections[100], 2: c.sections[100]})

    def test_real_bank_and_sections_apply_once_before_queue_release(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch = freeze(claim, batch)
        c.begin(claim, batch, .6)
        self.assertFalse(c.publishable())
        self.assertEqual(c.bank.memory.sequence, 0)
        for i in range(2):
            self.assertFalse(c.advance(claim, batch, .6, target_budget=1))
            self.assertEqual(c.preflight_targets, i+1)
            self.assertEqual(c.bank.memory.sequence, 0)
            self.assertEqual([v.calls for v in c.sections.values()], [0, 0])
        self.assertFalse(c.advance(claim, batch, .6, target_budget=1))
        self.assertEqual([v.calls for v in c.sections.values()], [0, 0])
        self.assertFalse(c.advance(claim, batch, .6, target_budget=1))
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)
        self.assertEqual([v.cumulative.integer(13) for v in c.sections.values()], [1, 0])
        self.assertEqual(pool.available_slots, 4)
        self.assertEqual(c.queue.queue.committed, 0)
        self.assertTrue(c.advance(claim, batch, .7, target_budget=1))
        self.assertTrue(c.publishable())
        self.assertEqual(c.queue.queue.committed, 1)
        self.assertEqual(pool.available_slots, 8)
        self.assertTrue(c.advance(claim, batch, .7))
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)
        self.assertEqual([v.calls for v in c.sections.values()], [1, 1])

    def test_invalid_projection_is_rejected_before_any_bank_or_section_effect(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch['sections'][1]['adjacency_coverage'] = -1.
        before = [bytes(v.cumulative.buffer) for v in c.sections.values()]
        batch = freeze(claim, batch)
        c.begin(claim, batch, .6)
        self.assertFalse(c.advance(claim, batch, .6, target_budget=1))
        with self.assertRaises(ValueError):
            c.advance(claim, batch, .6, target_budget=1)
        self.assertEqual(c.bank.memory.sequence, 0)
        self.assertEqual([bytes(v.cumulative.buffer) for v in c.sections.values()], before)
        self.assertFalse(c.publishable())
        self.assertEqual(c.queue.queue.committed, 0)
        self.assertTrue(c.abort(claim, batch, .6))
        self.assertTrue(c.publishable())

    def test_changed_packet_cannot_replace_an_inflight_or_completed_write(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        changed = copy.deepcopy(batch)
        changed['sections'][1]['record']['membership'] = .25
        batch = prepare(c, claim, batch)
        changed = freeze(claim, changed)
        c.advance(claim, batch, .6, target_budget=1)
        with self.assertRaises(ValueError):
            c.advance(claim, changed, .6)
        self.assertEqual(c.sections[101].calls, 0)
        c.advance(claim, batch, .6)
        with self.assertRaises(ValueError):
            c.advance(claim, changed, .6)
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)

    def test_lost_bank_acknowledgment_recovers_without_reinforcement_or_backdating(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch = prepare(c, claim, batch, phase=ref.BANK)
        apply = c.bank.advance_packet
        def interrupted(*args):
            c.bank.advance_packet = apply
            apply(*args)
            raise RuntimeError('bank acknowledgment lost')
        c.bank.advance_packet = interrupted
        with self.assertRaises(RuntimeError):
            c.advance(claim, batch, .8)
        self.assertFalse(c.publishable())
        self.assertEqual(c.queue.queue.committed, 0)
        self.assertEqual(c.bank.extra(0)['available_end'], .8)
        with self.assertRaises(ValueError):
            c.abort(claim, batch, .8)
        self.assertFalse(c.advance(claim, batch, .9))
        self.assertTrue(c.advance(claim, batch, .9))
        self.assertEqual(c.bank_receipt_recoveries, 1)
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)
        self.assertEqual(c.bank.extra(0)['available_end'], .8)

    def test_section_failure_before_mutation_resumes_only_unfinished_target(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch = prepare(c, claim, batch)
        c.sections[101].failure = 'before'
        with self.assertRaises(RuntimeError):
            c.advance(claim, batch, .6)
        self.assertEqual([v.calls for v in c.sections.values()], [1, 1])
        self.assertFalse(c.publishable())
        self.assertTrue(c.advance(claim, batch, .7))
        self.assertEqual([v.calls for v in c.sections.values()], [1, 2])
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)

    def test_lost_section_acknowledgment_recovers_even_when_record_is_outside_recent_ring(self):
        c, pool, h = setup()
        # A context starting inside the old occurrence need not put it in its ring.
        c.sections[101].cumulative.time(4, .05)
        claim, batch = packet(c, pool, h)
        batch = prepare(c, claim, batch)
        c.sections[101].failure = 'after'
        with self.assertRaises(RuntimeError):
            c.advance(claim, batch, .6)
        self.assertEqual(c.sections[101].ring, [])
        self.assertEqual(c.sections[101].cumulative.integer(13), 1)
        self.assertTrue(c.advance(claim, batch, .7))
        self.assertEqual(c.section_receipt_recoveries, 1)
        self.assertEqual([v.calls for v in c.sections.values()], [1, 1])
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)

    def test_lost_final_queue_acknowledgment_recovers_after_payload_release(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch = prepare(c, claim, batch)
        finish = c.queue.finish
        def interrupted(*args):
            c.queue.finish = finish
            finish(*args)
            raise RuntimeError('final acknowledgment lost')
        c.queue.finish = interrupted
        with self.assertRaises(RuntimeError):
            c.advance(claim, batch, .6)
        self.assertEqual(c.queue.queue.committed, 1)
        self.assertEqual(pool.available_slots, 8)
        self.assertFalse(c.publishable())
        self.assertTrue(c.advance(claim, batch, .7))
        self.assertEqual(c.ack_receipt_recoveries, 1)
        self.assertEqual([v.calls for v in c.sections.values()], [1, 1])

    def test_failed_bank_validation_can_abort_without_a_committed_sequence_gap(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch['admissions'][h]['scales'][0] = 2.
        batch = prepare(c, claim, batch, phase=ref.BANK)
        with self.assertRaises(ValueError):
            c.advance(claim, batch, .6)
        self.assertTrue(c.abort(claim, batch, .6))
        self.assertFalse(c.abort(claim, batch, .6))
        self.assertEqual((c.bank.memory.sequence, c.queue.queue.committed, c.queue.queue.dropped), (0, 0, 1))
        self.assertEqual(pool.available_slots, 8)
        self.assertTrue(c.publishable())
        with self.assertRaises(ValueError):
            c.advance(claim, batch, .6)

        for i in range(6, 12):
            c.bank.memory.clock.observe(raw(i), (i+1)/10)
        for history in c.sections.values():
            history.observe(section_activity([hop(.6, 1.2)], [], 1, {4}, .6, 1.2), 1, 4)
        following, valid = packet(c, pool, h, identity=2, index=6)
        self.assertEqual(following['sequence'], 1)
        valid = prepare(c, following, valid, 1.2)
        self.assertTrue(c.advance(following, valid, 1.2))
        self.assertEqual((c.bank.memory.sequence, c.queue.queue.committed, c.queue.queue.dropped), (1, 1, 1))
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)

    def test_abort_acknowledgment_loss_is_idempotent(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch = prepare(c, claim, batch, phase=ref.BANK)
        finish = c.queue.finish
        def interrupted(*args):
            c.queue.finish = finish
            finish(*args)
            raise RuntimeError('loss acknowledgment lost')
        c.queue.finish = interrupted
        with self.assertRaises(RuntimeError):
            c.abort(claim, batch, .6)
        self.assertTrue(c.abort(claim, batch, .7))
        self.assertEqual(c.queue.queue.dropped, 1)
        self.assertTrue(c.publishable())

    def test_boolean_success_without_applied_state_is_not_a_receipt(self):
        c, pool, h = setup()
        claim, batch = packet(c, pool, h)
        batch = prepare(c, claim, batch, phase=ref.BANK)
        apply = c.bank.advance_packet
        c.bank.advance_packet = lambda *args: True
        with self.assertRaises(RuntimeError):
            c.advance(claim, batch, .6)
        self.assertEqual(c.queue.queue.committed, 0)
        c.bank.advance_packet = apply
        self.assertFalse(c.advance(claim, batch, .6))
        c.sections[100].failure = 'false_success'
        with self.assertRaises(RuntimeError):
            c.advance(claim, batch, .6)
        self.assertEqual(c.queue.queue.committed, 0)
        self.assertFalse(c.publishable())
        c.sections[100].failure = None
        self.assertTrue(c.advance(claim, batch, .7))

    def test_zero_weight_target_is_an_explicit_no_effect_receipt(self):
        c, pool, h = setup((1., 0.))
        claim, batch = packet(c, pool, h, weights=(1., 0.))
        batch = prepare(c, claim, batch)
        self.assertTrue(c.advance(claim, batch, .6))
        self.assertEqual(c.sections[101].cumulative.integer(13), 0)
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)
        self.assertTrue(c.publishable())

    def test_queue_boolean_success_without_release_cannot_publish_or_abort(self):
        for committed in (True, False):
            c, pool, h = setup()
            claim, batch = packet(c, pool, h)
            batch = prepare(c, claim, batch, phase=ref.SECTIONS if committed else ref.BANK)
            finish = c.queue.finish
            c.queue.finish = lambda *args: {'committed': committed}
            method = c.advance if committed else c.abort
            with self.assertRaises(RuntimeError):
                method(claim, batch, .6)
            self.assertFalse(c.publishable())
            self.assertEqual(pool.available_slots, 4)
            self.assertEqual((c.queue.queue.committed, c.queue.queue.dropped), (0, 0))
            c.queue.finish = finish
            self.assertTrue(method(claim, batch, .7))
            self.assertTrue(c.publishable())
            self.assertEqual(pool.available_slots, 8)
            self.assertEqual((c.queue.queue.committed, c.queue.queue.dropped), (int(committed), int(not committed)))

    def test_full_1024_target_fanout_retains_each_joint_weight_and_bounded_delivery(self):
        weights = (1/1024,)*1024
        c, pool, h = setup(weights)
        claim, batch = packet(c, pool, h, weights=weights)
        batch = freeze(claim, batch)
        for i in range(64):
            self.assertFalse(c.advance(claim, batch, .6, target_budget=16))
            self.assertEqual(c.preflight_targets, (i+1)*16)
            self.assertEqual(c.bank.memory.sequence, 0)
            self.assertEqual(sum(v.calls for v in c.sections.values()), 0)
        self.assertFalse(c.advance(claim, batch, .6))
        self.assertEqual(sum(v.calls for v in c.sections.values()), 0)
        c.sections[1123].failure = 'after'
        for i in range(63):
            self.assertFalse(c.advance(claim, batch, .6, target_budget=16))
            self.assertEqual(sum(v.calls for v in c.sections.values()), (i+1)*16)
            self.assertFalse(c.publishable())
            self.assertEqual(pool.available_slots, 4)
        with self.assertRaises(RuntimeError):
            c.advance(claim, batch, .6, target_budget=16)
        self.assertFalse(c.publishable())
        self.assertEqual(c.queue.queue.committed, 0)
        self.assertTrue(c.advance(claim, batch, .7, target_budget=16))
        self.assertEqual(c.section_receipt_recoveries, 1)
        self.assertEqual(sum(v.calls for v in c.sections.values()), 1024)
        self.assertEqual(c.preflight_targets, 1024)
        self.assertEqual(c.payload_bytes()['total_fixed_buffers'], 1474816)
        self.assertEqual(c.bank.memory.record(0)['strength'], 1.)
        self.assertEqual(c.bank.memory.record(0)['membership_total'], 1.)
        for history in c.sections.values():
            self.assertAlmostEqual(history.snapshot()['cumulative']['edge_denominator'], .1/1024)
        self.assertEqual(c.queue.queue.committed, 1)
        self.assertEqual(pool.available_slots, 8)

    def test_stale_epoch_changed_generation_and_foreign_progress_cannot_acknowledge(self):
        for fault in ('epoch', 'generation', 'progress'):
            c, pool, h = setup()
            claim, batch = packet(c, pool, h)
            batch = prepare(c, claim, batch)
            if fault == 'epoch':
                claim = dict(claim, epoch=2)
            elif fault == 'generation':
                c.sections[100].cumulative.integer(1, 99)
            else:
                c.sections[100].cumulative.integer(13, 99)
            with self.assertRaises((ValueError, RuntimeError)):
                c.advance(claim, batch, .6)
            self.assertEqual(c.queue.queue.committed, 0)

    def test_duplicate_target_and_wrong_joint_path_are_rejected_before_effects(self):
        for fault in ('target', 'path', 'membership', 'sequence'):
            c, pool, h = setup()
            claim, batch = packet(c, pool, h)
            p = batch['sections'][1]
            if fault == 'target':
                p['target'] = 100
            elif fault == 'path':
                p['path_id'] = 99
            elif fault == 'membership':
                p['record']['membership'] = .1
            else:
                p['sequence'] = 2
            batch = freeze(claim, batch)
            with self.assertRaises(ValueError):
                c.advance(claim, batch, .6)
            self.assertEqual(c.bank.memory.sequence, 0)
            self.assertFalse(c.publishable())
            self.assertTrue(c.abort(claim, batch, .6))

    def test_independent_stream_sums_survive_repeated_interrupted_section_receipts(self):
        rng = random.Random(47022)
        for run in range(30):
            c, pool, h = setup()
            expected_total, expected_sections = 0., [0., 0.]
            for step in range(20):
                index = step*6
                end, cut = (index+1)/10, (index+6)/10
                if step:
                    for i in range(index, index+6):
                        c.bank.memory.clock.observe(raw(i), (i+1)/10)
                    delta = section_activity([hop(index/10, cut)], [], 1, {4}, index/10, cut)
                    for history in c.sections.values():
                        history.observe(delta, 1, 4)
                weights = (rng.random()*.5, rng.random()*.5)
                claim, batch = packet(c, pool, h, step+1, index, weights, admission=step == 0)
                batch = prepare(c, claim, batch, cut)
                selected = rng.randrange(2)
                c.sections[100+selected].failure = 'after'
                with self.assertRaises(RuntimeError):
                    c.advance(claim, batch, cut)
                self.assertTrue(c.advance(claim, batch, cut))
                self.assertTrue(c.advance(claim, batch, cut))
                expected_total += sum(weights)
                for i, weight in enumerate(weights):
                    expected_sections[i] += weight*(end-index/10)
                self.assertAlmostEqual(c.bank.memory.record(0)['membership_total'], expected_total)
                self.assertAlmostEqual(c.bank.memory.record(0)['strength'], min(3., expected_total))
                for i, expected in enumerate(expected_sections):
                    self.assertAlmostEqual(c.sections[100+i].snapshot()['cumulative']['edge_denominator'], expected)
                self.assertEqual(c.queue.queue.committed, step+1)
                self.assertEqual(pool.available_slots, 8)

    def test_abort_during_bank_preparation_releases_unpublished_buffers_and_queue(self):
        c, pool, h = setup()
        claim, native = packet(c, pool, h)
        sealed = prepare(c, claim, native, phase=ref.BANK)
        for _ in range(2):
            self.assertFalse(c.advance(claim, sealed, .6, bank_budget=1))
            self.assertEqual(c.bank.memory.sequence, 0)
            self.assertEqual(sum(v.calls for v in c.sections.values()), 0)
        self.assertTrue(any(c.bank.knots))
        self.assertFalse(c.publishable())
        self.assertTrue(c.abort(claim, sealed, .7))
        self.assertFalse(any(c.bank.knots))
        self.assertFalse(any(c.bank.anchors))
        self.assertIsNone(c.bank._pending)
        self.assertEqual((c.queue.queue.committed, c.queue.queue.dropped, pool.available_slots), (0, 1, 8))
        self.assertTrue(c.publishable())

    def test_bank_budget_and_section_budget_have_separate_phase_boundaries(self):
        c, pool, h = setup()
        claim, native = packet(c, pool, h)
        sealed = prepare(c, claim, native, phase=ref.BANK)
        for i in range(4):
            self.assertFalse(c.advance(claim, sealed, .6+i/10, target_budget=1, bank_budget=1))
            self.assertEqual(c.bank.preparation_units, i+1)
            self.assertEqual(sum(v.calls for v in c.sections.values()), 0)
            self.assertFalse(c.publishable())
        self.assertEqual(c.bank.extra(0)['available_end'], .6+3/10)
        self.assertFalse(c.advance(claim, sealed, 1., target_budget=1, bank_budget=1))
        self.assertEqual(sum(v.calls for v in c.sections.values()), 1)
        self.assertTrue(c.advance(claim, sealed, 1., target_budget=1, bank_budget=1))
        self.assertEqual(c.bank.memory.record(0)['membership_total'], 1.)
        self.assertEqual(pool.available_slots, 8)

    def test_consumer_and_bank_share_one_header_decode_and_release_it(self):
        from unittest.mock import patch
        import temporal_consumer_packet_reference as packets
        c, pool, h = setup()
        claim, native = packet(c, pool, h)
        sealed = freeze(claim, native)
        original, count = packets.decode, 0
        def decode(data):
            nonlocal count
            count += data is sealed.bank_bytes
            return original(data)
        with patch.object(packets, 'decode', decode):
            for _ in range(8):
                done = c.advance(claim, sealed, .6, target_budget=1, bank_budget=1)
                self.assertEqual(count, 1)
                if c.bank._pending:
                    self.assertIs(c.bank._pending['input'], c._bank_input)
                    self.assertIs(c._bank_batch, c._bank_input._header)
                if done:
                    break
        self.assertTrue(done)
        self.assertEqual(c.bank.memory.record(0)['membership_total'], 1.)
        self.assertIsNone(c._bank_input)
        self.assertIsNone(c._bank_batch)
        self.assertIsNone(c.bank._pending)
        self.assertEqual(pool.available_slots, 8)

    def test_invalid_lazy_relation_after_section_preflight_remains_abortable(self):
        from dataclasses import replace
        c, pool, h = setup()
        claim, native = packet(c, pool, h)
        sealed = replace(freeze(claim, native), relation_bytes=(b'n',))
        self.assertFalse(c.advance(claim, sealed, .6))
        self.assertEqual(c.preflight_targets, 2)
        with self.assertRaises(ValueError):
            c.advance(claim, sealed, .6)
        self.assertEqual(sum(v.calls for v in c.sections.values()), 0)
        self.assertEqual(c.bank.memory.sequence, 0)
        self.assertTrue(c.abort(claim, sealed, .6))
        self.assertIsNone(c._bank_input)
        self.assertIsNone(c._bank_batch)
        self.assertEqual(pool.available_slots, 8)


if __name__ == '__main__':
    unittest.main()
