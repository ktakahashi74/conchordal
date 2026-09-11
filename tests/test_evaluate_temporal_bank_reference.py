"""Finite full-history membership, eviction and cache/bank integration oracles."""

import copy
import math
from pathlib import Path
import random
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import temporal_bank_reference as ref
from temporal_memory_reference import EpisodeRetention, GapClock
from temporal_consumer_packet_reference import ConsumerPacketBuilder, pack_knots
from temporal_matcher_reference import committed_episode, match_query
from temporal_occurrence_reference import OccurrenceLedger, joint_occurrence_support
from temporal_query_scheduler_reference import CoarseCache, pack_coarse
from test_evaluate_temporal_memory_reference import clock, write
from test_evaluate_temporal_query_scheduler_reference import raw, span, scheduler
from test_evaluate_temporal_occurrence_reference import record, paths, hop


def bank(capacity=4, edges=16, c=None):
    return ref.EpisodeBank(EpisodeRetention(clock(30) if c is None else c, 20., 4., 3., 100.,
                                           capacity=capacity), [1.]*10, edge_capacity=edges)


def event(sequence, end, cells, **extra):
    support = joint_occurrence_support([{'path_id': i, 'episode_handle': e,
                                        'context_handle': c, 'weight': weight}
                                       for i, (e, c, weight) in enumerate(cells)], 1.)
    w = write(sequence, end, support['episodes'], support['unknown_episode_support'],
              support['unassigned_available_support'], **extra)
    w['support'] = support
    return w


def admissions(w, handles):
    index = round(w['start']*10)
    return {h: committed_episode(span(index), w, h, h) for h in handles}


def relation(w, source, target, support=.5, **values):
    return {'source': source, 'target': target, 'support': support,
            'occurrence_id': w['occurrence_id'], 'support_id': w['support_id'],
            'values': {key: values.get(key) for key in ref.RELATION_VALUES}}


def bank_packet(w, episodes, relations=()):
    claim = dict(epoch=w['epoch'], ticket=1, sequence=w['sequence'],
                 delivered_at=w['delivered_at'], packed=bytes(128))
    builder = ConsumerPacketBuilder(claim, dict(write=w), 0, len(episodes), len(relations))
    for row in relations:
        builder.append_relation(row)
    for handle, episode in episodes.items():
        builder.append_admission(handle, {k:v for k,v in episode.items() if k != 'knots'},
                                 pack_knots(episode['knots']))
    return builder.finish()


class BankTests(unittest.TestCase):
    def test_mixed_packed_and_native_admissions_preserve_bytes_in_either_order(self):
        for packed_first in (True, False):
            with self.subTest(packed_first=packed_first):
                mixed, native = bank(2), bank(2)
                mixed.next_handle = native.next_handle = 2**53+7
                handles = [mixed.reserve(), mixed.reserve()]
                self.assertEqual(handles, [native.reserve(), native.reserve()])
                w = event(1, .1, [(h, handles[0], .5) for h in handles])
                supplied = admissions(w, handles)
                for h in handles:
                    row = supplied[h]['knots'][0]
                    weight, _, error = row['moments'][0]
                    row['moments'][0] = (weight, -0., error)
                    row['values'][0] = -0.
                expected = copy.deepcopy(supplied)
                packed_handle = handles[0 if packed_first else 1]
                packed = pack_knots(supplied[packed_handle]['knots'])
                supplied[packed_handle] = {k:v for k,v in supplied[packed_handle].items() if k != 'knots'}
                supplied[packed_handle]['packed_knots'] = packed
                mixed.apply(w, 3., supplied)
                native.apply(w, 3., expected)
                for attribute in ('knots','anchors','edges','reservations'):
                    self.assertEqual(getattr(mixed, attribute), getattr(native, attribute))
                self.assertEqual(mixed.memory.storage, native.memory.storage)
                slot = 0 if packed_first else 1
                self.assertEqual(bytes(mixed.knots[slot*128*320:slot*128*320+320]), packed)
                self.assertEqual(struct.unpack_from('<d', packed, 8)[0], 0.)
                self.assertEqual(packed[8:16], struct.pack('<d', -0.))

    def test_native_admission_still_rejects_contradictory_derived_views(self):
        for field in ('coverage','gap','gap_location_lost','values'):
            with self.subTest(field=field):
                b = bank(1); h = b.reserve()
                w = event(1, .1, [(h, None, 1.)])
                supplied = admissions(w, [h])
                row = supplied[h]['knots'][0]
                if field in ('coverage','values'): row[field][0] += .25
                else: row[field] = not row[field]
                before = bytes(b.knots), bytes(b.anchors), bytes(b.edges), bytes(b.memory.storage)
                with self.assertRaises(ValueError):
                    b.apply(w, 3., supplied)
                self.assertEqual((bytes(b.knots),bytes(b.anchors),bytes(b.edges),bytes(b.memory.storage)), before)
                self.assertIn(h, b.reservations)

    def test_packed_late_hidden_values_and_provenance_are_validated_before_publication(self):
        for offset, value, hidden in ((8, float('nan'), True), (16, -1., True),
                                       (296, float('nan'), False), (288, 4., False), (312, 5, False)):
            with self.subTest(offset=offset, hidden=hidden):
                b = bank(2); hs = [b.reserve(),b.reserve()]
                w = event(1, .1, [(h, None, .5) for h in hs])
                supplied = admissions(w, hs)
                claim = dict(epoch=1,ticket=1,sequence=1,delivered_at=w['delivered_at'],packed=bytes(128))
                builder = ConsumerPacketBuilder(claim, dict(write=w), 0, 2, 0)
                for i,h in enumerate(hs):
                    packed = bytearray(pack_knots(supplied[h]['knots']))
                    if i == 1:
                        if hidden: struct.pack_into('<d', packed, 0, 0.)
                        struct.pack_into('<Q' if offset == 312 else '<d', packed, offset, value)
                    builder.append_admission(h, {k:v for k,v in supplied[h].items() if k != 'knots'}, bytes(packed))
                packet = builder.finish()
                self.assertFalse(b.advance_packet(packet, 3., 2))
                self.assertTrue(any(b.knots))
                with self.assertRaises(ValueError):
                    b.advance_packet(packet, 3., 1)
                for storage in (b.knots,b.anchors,b.edges,b.memory.storage): self.assertFalse(any(storage))
                self.assertEqual(b.memory.sequence, 0)
                self.assertIsNone(b._pending)
                self.assertTrue(all(h in b.reservations for h in hs))

    def test_fixed_bank_layout_includes_metadata_and_shared_edge_cap(self):
        b = ref.EpisodeBank(EpisodeRetention(GapClock(1), 20., 4., 3., 10.), [1.]*10)
        size = b.payload_bytes()
        self.assertEqual(size['bank_including_retention_metadata'], 11862016)
        self.assertEqual(size['knots'], 256*128*320)
        self.assertEqual(size['anchors'], 256*32*96)
        self.assertEqual(size['edges'], 256*16*128)
        self.assertEqual(size['edge_scratch'], 256*16*128)
        self.assertEqual(size['retention_additional'], 381184)
        self.assertEqual(size['scales_and_reservations'], 2128)

    def test_reservation_cancellation_and_capacity_never_reuse_id(self):
        b = bank(2)
        first, second = b.reserve(), b.reserve()
        self.assertEqual((first, second), (1, 2))
        with self.assertRaises(BufferError):
            b.reserve()
        self.assertTrue(b.cancel(first))
        self.assertFalse(b.cancel(first))
        self.assertEqual(b.reserve(), 3)
        self.assertEqual(b.episodes(3.), [])
        b.next_handle = 2**64
        with self.assertRaises(OverflowError):
            b.reserve()

    def test_admission_freezes_original_descriptor_and_anchor_indices(self):
        b = bank()
        handle = b.reserve()
        w = event(1, .1, [(handle, handle, 1.)])
        supplied = admissions(w, [handle])
        original = copy.deepcopy(supplied[handle])
        original['available_end'] = 3.
        b.apply(w, 3., supplied)
        supplied[handle]['knots'][0]['values'][0] = 999.
        out = b.episodes(3.)
        self.assertEqual(out, [original])
        out[0]['knots'][0]['values'][0] = 888.
        self.assertEqual(b.episodes(3.), [original])
        anchor = struct.unpack_from('<8Q4d', b.anchors)
        self.assertEqual(anchor[:8], (0,)+(2**64-1,)*7)
        self.assertEqual(anchor[8:], (1., .1, .1, 0.))
        self.assertEqual(b.bindings(), {(handle, handle): (0, handle)})
        self.assertEqual(b.memory.record(0)['strength'], 1.)
        self.assertEqual(b.extra(0)['ledger_delivered_at'], .6)

    def test_recurrence_updates_membership_but_preserves_first_descriptor(self):
        b = bank()
        h = b.reserve()
        first = event(1, .1, [(h, h, .5), (h, None, .5)])
        b.apply(first, 3., admissions(first, [h]))
        old = b.episodes(3.)
        b.apply(event(2, .2, [(h, h, 1.)]), 3.)
        self.assertEqual(b.episodes(3.), old)
        self.assertEqual(b.memory.record(0)['strength'], 2.)
        self.assertEqual(b.memory.record(0)['membership_total'], 2.)
        f = b.focus([{'context_handle': h, 'weight': 1.}])['episodes'][h]
        self.assertEqual(f['compatibility'], .75)
        self.assertEqual(f['unresolved_membership'], .25)

    def test_split_pruning_and_return_do_not_reassign_context_edges(self):
        b = bank()
        h = b.reserve()
        w = event(1, .1, [(h, h, 1.)])
        b.apply(w, 3., admissions(w, [h]))
        before = bytes(b.edges), bytes(b.memory.storage)
        split = b.focus([{'context_handle': h, 'weight': .25}, {'context_handle': h, 'weight': .5},
                         {'context_handle': None, 'weight': .25}])
        self.assertEqual(split['episodes'][h]['compatibility'], .75)
        self.assertEqual(split['unknown_focus_mass'], .25)
        self.assertEqual(b.focus([{'context_handle': None, 'weight': 1.}])['episodes'][h]['compatibility'], 0.)
        self.assertEqual(b.focus([{'context_handle': h, 'weight': 1.}])['episodes'][h]['compatibility'], 1.)
        self.assertEqual((bytes(b.edges), bytes(b.memory.storage)), before)

    def test_evicted_representative_keeps_unretrieved_link_and_unknown_mass(self):
        b = bank(2)
        a, c = b.reserve(), b.reserve()
        w = event(1, .1, [(a, a, .1), (c, a, .9)])
        b.apply(w, 3., admissions(w, [a, c]))
        losses = b.make_room(1, 3.)
        self.assertEqual(losses[0]['handle'], a)
        self.assertEqual(losses[0]['affected_links'], [(c, 1, a)])
        edge = b.edge_records(1)[0]
        self.assertTrue(edge['unretrieved'])
        self.assertEqual(edge['support'], .9)
        self.assertEqual(b.memory.record(1)['membership_total'], .9)
        self.assertEqual(b.focus([{'context_handle': a, 'weight': 1.}])['unknown_focus_mass'], 1.)
        self.assertEqual(b.focus([])['episodes'][c]['unresolved_membership'], 1.)
        replacement = b.reserve()
        self.assertNotEqual(replacement, a)
        w2 = event(2, .2, [(replacement, replacement, 1.)])
        b.apply(w2, 3., admissions(w2, [replacement]))
        self.assertEqual(b.current_handles()[0], replacement)
        self.assertTrue(b.edge_records(1)[0]['unretrieved'])

    def test_old_cache_cannot_bind_evicted_episode_to_reused_slot(self):
        b = bank(1)
        h = b.reserve(); w = event(1, .1, [(h, h, 1.)])
        b.apply(w, 3., admissions(w, [h]))
        from test_evaluate_temporal_query_scheduler_reference import result, entry
        cache = CoarseCache(1, 4)
        packed = bytearray(6272)
        pack_coarse(result(coarse_entries=[entry(h, h)]), b.bindings(), packed, 3.)
        cache.insert(packed)
        self.assertEqual(cache.snapshots(b.current_handles(), 3.)[0]['entries'], {h: 0.})
        b.make_room(1, 3.)
        h2 = b.reserve(); w2 = event(2, .2, [(h2, h2, 1.)])
        b.apply(w2, 3., admissions(w2, [h2]))
        self.assertEqual(cache.snapshots(b.current_handles(), 3.)[0]['entries'], {})

    def test_context_and_correspondence_compete_for_same_edge_cap(self):
        b = bank(2, edges=1, c=clock(6))
        a, c = b.reserve(), b.reserve()
        first = event(1, .1, [(c, c, 1.)])
        b.apply(first, .6, admissions(first, [c]))
        b.memory.clock.observe(raw(6), .7)
        second = event(2, .2, [(a, c, .1), (a, None, .4), (None, None, .5)])
        proposal = relation(second, a, c, .5, transpose_log2=.5, source_time=.2, target_time=.1)
        b.apply(second, .7, admissions(second, [a]), [proposal])
        edge = b.edge_records(1)[0]
        self.assertEqual(edge['kind'], 2)
        self.assertEqual(edge['values']['transpose_log2'], .5)
        self.assertIsNone(edge['values']['tempo_log2'])
        self.assertEqual(b.extra(1)['dropped_membership'], .1)
        self.assertEqual(b.focus([])['episodes'][a]['unresolved_membership'], 1.)

    def test_correspondence_cannot_use_unheard_target_or_shift_original_times(self):
        b = bank(2, c=clock(6))
        a, c = b.reserve(), b.reserve()
        first = event(1, .1, [(a, a, .5), (c, c, .5)])
        with self.assertRaises(ValueError):
            b.apply(first, .6, admissions(first, [a, c]), [relation(first, a, c)])
        b.apply(first, .6, admissions(first, [a, c]))
        b.memory.clock.observe(raw(6), .7)
        second = event(2, .2, [(a, c, 1.)])
        for values in ({'source_time': .3}, {'target_time': .2}, {'pitch_residual': -.1}):
            before = bytes(b.edges), bytes(b.memory.storage)
            with self.assertRaises(ValueError):
                b.apply(second, .7, relations=[relation(second, a, c, **values)])
            self.assertEqual((bytes(b.edges), bytes(b.memory.storage)), before)

    def test_bank_receipt_cannot_backdate_target_availability_to_ledger_delivery(self):
        b = bank(2); a, c = b.reserve(), b.reserve()
        first = event(1, .1, [(c, c, 1.)])
        b.apply(first, 3., admissions(first, [c]))
        self.assertEqual(b.episodes(3.)[0]['available_end'], 3.)
        self.assertEqual(b.extra(0)['ledger_delivered_at'], .6)
        second = event(2, .2, [(a, c, 1.)])
        with self.assertRaises(ValueError):
            b.apply(second, 3., admissions(second, [a]), [relation(second, a, c)])
        self.assertEqual(b.memory.sequence, 1)

    def test_compressed_full_descriptor_preserves_moments_and_anchor_members(self):
        b = bank(c=clock(270)); h = b.reserve()
        s = span()
        for index in range(1, 260):
            s.push(raw(index, values=[float(index % 7)]*10), (index+1)/10)
        w = event(1, 26., [(h, h, 1.)], start=0., delivered_at=27.)
        episode = committed_episode(s, w, h, h)
        b.apply(w, 27., {h: episode})
        exported = b.episodes(27.)[0]
        self.assertEqual(exported, episode)
        self.assertEqual(len(exported['knots']), 128)
        self.assertGreater(exported['compression_error'], 0.)
        for anchor in range(32):
            indices = struct.unpack_from('<8Q', b.anchors, anchor*96)
            expected = tuple(range(anchor*4, min(anchor*4+8, 128)))
            self.assertEqual(indices[:len(expected)], expected)
            self.assertEqual(indices[len(expected):], (2**64-1,)*(8-len(expected)))

    def test_dropped_context_gets_only_new_support_on_later_readmission(self):
        b = bank(2, edges=1)
        a, c = b.reserve(), b.reserve()
        first = event(1, .1, [(a, a, .3), (a, c, .2), (c, c, .5)])
        b.apply(first, 3., admissions(first, [a, c]))
        self.assertEqual(b.extra(0)['dropped_membership'], .2)
        b.apply(event(2, .2, [(a, c, 1.)]), 3.)
        edge = b.edge_records(0)[0]
        self.assertEqual((edge['target'], edge['support']), (c, 1.))
        self.assertAlmostEqual(b.extra(0)['dropped_membership'], .5)
        self.assertAlmostEqual(b.focus([{'context_handle': c, 'weight': 1.}])['episodes'][a]['compatibility'], 1/1.5)

    def test_repeat_is_noop_and_changed_joint_metadata_is_not_reinforcement(self):
        b = bank(); h = b.reserve(); w = event(1, .1, [(h, h, 1.)]); supplied = admissions(w, [h])
        b.apply(w, 3., supplied)
        before = bytes(b.edges), bytes(b.memory.storage), bytes(b.knots)
        self.assertFalse(b.apply(copy.deepcopy(w), 3., copy.deepcopy(supplied)))
        changed = copy.deepcopy(w)
        changed['support']['joint'] = {(h, None): 1.}
        with self.assertRaises(ValueError):
            b.apply(changed, 3.)
        self.assertEqual((bytes(b.edges), bytes(b.memory.storage), bytes(b.knots)), before)

    def test_failed_descriptor_or_memory_validation_leaves_live_state_unchanged(self):
        b = bank(); a, c = b.reserve(), b.reserve()
        w = event(1, .1, [(a, a, 1.)]); b.apply(w, 3., admissions(w, [a]))
        w2 = event(2, .2, [(c, c, 1.)], costs={a: -1.})
        for mutation in ('moment', 'source_generation_type', 'memory'):
            supplied = admissions(w2, [c])
            if mutation == 'moment':
                supplied[c]['knots'][0]['moments'][0] = (1., 999., 0.)
            if mutation == 'source_generation_type':
                supplied[c]['source_generation'] = 4.
            before = bytes(b.knots), bytes(b.anchors), bytes(b.edges), bytes(b.memory.storage)
            with self.assertRaises(ValueError):
                b.apply(w2, 3., supplied)
            self.assertEqual((bytes(b.knots), bytes(b.anchors), bytes(b.edges), bytes(b.memory.storage)), before)
            self.assertIn(c, b.reservations)
            self.assertEqual(b.memory.sequence, 1)

    def test_full_capacity_never_evicts_as_side_effect_of_apply(self):
        b = bank(1); a = b.reserve(); w = event(1, .1, [(a, a, 1.)])
        b.apply(w, 3., admissions(w, [a]))
        c = b.reserve(); w2 = event(2, .2, [(c, c, 1.)])
        before = b.episodes(3.)
        with self.assertRaises(BufferError):
            b.apply(w2, 3., admissions(w2, [c]))
        self.assertEqual(b.episodes(3.), before)
        self.assertEqual(b.memory.evictions, 0)

    def test_original_epoch_and_issued_handle_marginals_are_required(self):
        b = bank(); h = b.reserve(); w = event(1, .1, [(h, h, 1.)])
        for bad in [event(1, .1, [(h, 99, 1.)]), {**w, 'epoch': 2}]:
            with self.assertRaises(ValueError):
                b.apply(bad, 3., admissions(w, [h]))
            self.assertEqual(b.memory.sequence, 0)
        fresh = ref.EpisodeBank(EpisodeRetention(GapClock(2), 20., 4., 3., 10.), [1.]*10)
        self.assertEqual(fresh.bindings(), {})
        self.assertEqual(fresh.reserve(), 1)
        with self.assertRaises(ValueError):
            fresh.apply(w, 3., admissions(w, [h]))

    def test_membership_matches_independent_unbounded_joint_sums(self):
        rng = random.Random(47018)
        for _ in range(30):
            b = bank(4)
            hs = [b.reserve() for _ in range(4)]
            sums = {h: 0. for h in hs}; joint = {}
            for seq in range(1, 21):
                cells = [(h, rng.choice(hs+[None]), .25) for h in hs]
                w = event(seq, seq/10, cells)
                b.apply(w, 3., admissions(w, hs) if seq == 1 else {})
                for h, target, value in cells:
                    sums[h] += value
                    if target is not None:
                        joint[h, target] = joint.get((h, target), 0.)+value
                focus = b.focus([{'context_handle': h, 'weight': .25} for h in hs])
                for h in hs:
                    total = math.fsum(value for (source, _), value in joint.items() if source == h)
                    self.assertAlmostEqual(focus['episodes'][h]['resolved_membership'], total/sums[h])
                    self.assertAlmostEqual(focus['episodes'][h]['compatibility'], total/sums[h]*.25)
            self.assertTrue(all(b.extra(i)['edge_drops'] == 0 for i in range(4)))

    def test_real_ledger_bank_scheduler_matcher_and_retention_integration(self):
        c = clock(6); b = bank(2, c=c); h = b.reserve()
        ledger = OccurrenceLedger(1)
        ledger.stage(record(1, 0., .1, 1.), paths((h, h, 1.)), .1)
        first = ledger.advance(.6, [hop(.1, .6)])[0]
        b.apply(first, .6, admissions(first, [h]))
        for index in range(6, 15):
            c.observe(raw(index), (index+1)/10)
        s = scheduler(); s.observe(0, raw(9), 1.)
        s.submit(0, span(9), 1, 2, 1002, 1.)
        job = s.take(1.)
        result = match_query(job['query'], b.episodes(1.5), 1.5)
        s.finish(job['ticket'], result, b.bindings(), 1.5)
        ledger.stage(record(2, .9, 1., 1.), paths((h, h, 1.)), 1.)
        second = ledger.advance(1.5, [hop(1., 1.5)], s.caches[0].snapshots(b.current_handles(), 1.5))[0]
        b.apply(second, 1.5)
        self.assertEqual(b.memory.record(0)['strength'], 2.)
        self.assertEqual(b.memory.record(0)['last_observed_end'], 1.)
        self.assertEqual(b.episodes(1.5)[0]['first_observed_end'], .1)
        self.assertEqual(b.edge_records(0)[0]['support'], 2.)
        self.assertEqual(ledger.snapshot()['context_edges'], {(h, h): 1.})

    def test_staged_admission_and_edges_publish_once_at_actual_final_cut(self):
        b, oracle = bank(), bank()
        hs = [b.reserve() for _ in range(3)]
        self.assertEqual(hs, [oracle.reserve() for _ in hs])
        w = event(1, .1, [(hs[0], hs[1], .25), (hs[1], hs[2], .5), (hs[2], None, .25)])
        native = admissions(w, hs)
        packet = bank_packet(w, native)
        before = bytes(b.memory.storage), bytes(b.edges), bytes(b.memory.window.storage)
        # Header, three admissions and three edge rows are all unpublished work.
        for step in range(7):
            self.assertFalse(b.advance_packet(packet, 3.+step/10, work_budget=1))
            self.assertEqual(b.preparation_units, step+1)
            self.assertEqual((bytes(b.memory.storage), bytes(b.edges), bytes(b.memory.window.storage)), before)
            self.assertEqual(b.episodes(3.+step/10), [])
        self.assertTrue(b.advance_packet(packet, 4., work_budget=1))
        oracle.apply(w, 4., native)
        self.assertEqual(b.episodes(4.), oracle.episodes(4.))
        self.assertEqual((b.memory.storage, b.edges, b.knots, b.anchors),
                         (oracle.memory.storage, oracle.edges, oracle.knots, oracle.anchors))
        self.assertTrue(all(b.extra(i)['available_end'] == 4. for i in range(3)))
        self.assertTrue(all(b.extra(i)['ledger_delivered_at'] == .6 for i in range(3)))
        self.assertTrue(b.advance_packet(packet, 5., 1))
        self.assertEqual(b.preparation_units, 8)
        self.assertEqual(b.memory.sequence, 1)
        self.assertIsNone(b._pending)

    def test_staged_abort_clears_unused_slots_without_touching_existing_memory(self):
        for stop in (1, 2, 3, 4):
            b = bank(3, c=clock(6))
            old, new = b.reserve(), b.reserve()
            first = event(1, .1, [(old, old, 1.)])
            b.apply(first, .6, admissions(first, [old]))
            w = event(2, .2, [(new, old, 1.)])
            packet = bank_packet(w, admissions(w, [new]), [relation(w, new, old)])
            before = (bytes(b.memory.storage), bytes(b.knots), bytes(b.anchors), bytes(b.edges))
            self.assertFalse(b.advance_packet(packet, 3., stop))
            with self.assertRaises(ValueError):
                b.cancel(new)
            with self.assertRaises(ValueError):
                b.make_room(3, 3.)
            with self.assertRaises(ValueError):
                b.apply(w, 3., admissions(w, [new]))
            self.assertTrue(b.abort_packet(packet))
            self.assertFalse(b.abort_packet(packet))
            self.assertEqual((bytes(b.memory.storage), bytes(b.knots), bytes(b.anchors), bytes(b.edges)), before)
            self.assertEqual(b.memory.sequence, 1)
            self.assertTrue(b.cancel(new))
            self.assertGreater(b.reserve(), new)

    def test_late_invalid_descriptor_and_final_memory_failure_release_preparation(self):
        for fault in ('scales', 'support'):
            b = bank(2)
            hs = [b.reserve(), b.reserve()]
            w = event(1, .1, [(h, None, .5) for h in hs])
            native = admissions(w, hs)
            if fault == 'scales':
                native[hs[1]]['scales'][0] = 2.
            else:
                w['support']['available_support'] = .9
            packet = bank_packet(w, native)
            self.assertFalse(b.advance_packet(packet, 3., 2))
            self.assertTrue(any(b.knots))
            with self.assertRaises(ValueError):
                while not b.advance_packet(packet, 3., 1):
                    pass
            self.assertFalse(any(b.memory.storage))
            self.assertFalse(any(b.knots))
            self.assertFalse(any(b.anchors))
            self.assertFalse(any(b.edges))
            self.assertEqual(b.memory.sequence, 0)
            self.assertIsNone(b._pending)
            self.assertTrue(all(h in b.reservations for h in hs))
            fixed = event(1, .1, [(h, None, .5) for h in hs])
            self.assertTrue(b.advance_packet(bank_packet(fixed, admissions(fixed, hs)), 3., 8))

    def test_partial_preparation_rejects_conflicts_and_backdated_cut_without_losing_owner(self):
        b = bank(2)
        h = b.reserve()
        w = event(1, .1, [(h, None, 1.)])
        packet = bank_packet(w, admissions(w, [h]))
        self.assertFalse(b.advance_packet(packet, 4., 2))
        changed = admissions(w, [h])
        changed[h]['compression_error'] = 1.
        other = bank_packet(w, changed)
        for bad, cut in ((other, 4.), (packet, 3.)):
            with self.assertRaises(ValueError):
                b.advance_packet(bad, cut, 1)
        with self.assertRaises(ValueError):
            b.abort_packet(other)
        for budget in (0, -1, 1025, True):
            with self.assertRaises(ValueError):
                b.advance_packet(packet, 4., budget)
        self.assertTrue(b.advance_packet(packet, 4., 2))
        self.assertEqual(b.memory.record(0)['membership_total'], 1.)
        with self.assertRaises(ValueError):
            b.advance_packet(other, 4.)

    def test_full_256_admissions_are_prepared_in_bounded_calls_with_exact_bytes(self):
        b = bank(256)
        hs = [b.reserve() for _ in range(256)]
        w = event(1, .1, [(h, hs[0], 1/256) for h in hs])
        packet = bank_packet(w, admissions(w, hs))
        for step in range(64):
            self.assertFalse(b.advance_packet(packet, 3., 8))
            self.assertEqual(b.preparation_units, (step+1)*8)
            self.assertEqual(b.memory.sequence, 0)
            self.assertEqual(b.bindings(), {})
        self.assertTrue(b.advance_packet(packet, 3.5, 8))
        self.assertEqual(b.preparation_units, 518)
        for slot in range(256):
            self.assertEqual(b.memory.record(slot)['membership_total'], 1/256)
            self.assertEqual(b.extra(slot)['available_end'], 3.5)
            self.assertEqual(bytes(b.knots[slot*128*320:slot*128*320+320]), packet.descriptor_bytes[0])
            self.assertEqual(b.edge_records(slot)[0]['support'], 1/256)
        self.assertTrue(all(h == 0 for h in b.reservations))

    def test_corrupt_compact_descriptor_fails_before_any_published_effect(self):
        from dataclasses import replace
        for offset, value in ((264, 0.), (0, float('nan')), (0, -1.)):
            b = bank(1)
            h = b.reserve()
            w = event(1, .1, [(h, None, 1.)])
            packet = bank_packet(w, admissions(w, [h]))
            malformed = bytearray(packet.descriptor_bytes[0])
            struct.pack_into('<d', malformed, offset, value)
            packet = replace(packet, descriptor_bytes=(bytes(malformed),))
            with self.assertRaises(ValueError):
                b.advance_packet(packet, 3.)
            self.assertEqual(b.memory.sequence, 0)
            self.assertFalse(any(b.knots))
            self.assertFalse(any(b.anchors))
            self.assertIsNone(b._pending)

    def test_joint_validation_can_abort_or_fail_late_without_any_preparation_effect(self):
        for abort in (False, True):
            b = bank(128)
            hs = [b.reserve() for _ in range(128)]
            w = event(1, .1, [(h, None, 1/128) for h in hs])
            if not abort:
                w['support']['joint'][(hs[-1], None)] = float('nan')
            packet = bank_packet(w, admissions(w, hs))
            self.assertFalse(b.advance_packet(packet, 3., 1))
            self.assertEqual(b.preparation_units, 1)
            if abort:
                self.assertTrue(b.abort_packet(packet))
            else:
                with self.assertRaises(ValueError):
                    b.advance_packet(packet, 3., 1)
            self.assertEqual(b.memory.sequence, 0)
            self.assertFalse(any(b.knots))
            self.assertFalse(any(b.anchors))
            self.assertFalse(any(b.edges))
            self.assertIsNone(b._pending)

    def test_relation_decode_budget_late_failure_and_abort_preserve_existing_state(self):
        from dataclasses import replace
        from unittest.mock import patch
        import temporal_consumer_packet_reference as packets
        for fault in ('wire', 'domain', 'abort'):
            b = bank(16, c=clock(6))
            hs = [b.reserve() for _ in range(16)]
            first = event(1, .1, [(h, None, 1/16) for h in hs])
            b.apply(first, .6, admissions(first, hs))
            for index in range(6, 12):
                b.memory.clock.observe(raw(index), (index+1)/10)
            w = event(2, .7, [(h, None, 1/16) for h in hs], start=.6, delivered_at=1.2)
            rows = [relation(w, h, target, 1/256) for h in hs for target in hs]
            packet = bank_packet(w, {}, rows)
            if fault == 'wire':
                packet = replace(packet, relation_bytes=packet.relation_bytes[:-1]+(b'n',))
            elif fault == 'domain':
                rows[-1]['values']['pitch_residual'] = float('nan')
                packet = bank_packet(w, {}, rows)
            before = bytes(b.memory.storage), bytes(b.edges), bytes(b.knots), bytes(b.anchors)
            decoded, method = [], packets.ConsumerPacket.relation
            def count(owner, index):
                decoded.append(index)
                return method(owner, index)
            with patch.object(packets.ConsumerPacket, 'relation', count):
                for step in range(3):
                    self.assertFalse(b.advance_packet(packet, 1.2, 1))
                    self.assertEqual(decoded, list(range((step+1)*64)))
                    self.assertEqual((bytes(b.memory.storage), bytes(b.edges), bytes(b.knots), bytes(b.anchors)), before)
                if fault == 'abort':
                    self.assertTrue(b.abort_packet(packet))
                else:
                    with self.assertRaises(ValueError):
                        b.advance_packet(packet, 1.2, 1)
                    self.assertEqual(decoded, list(range(256)))
            self.assertEqual(b.memory.sequence, 1)
            self.assertIsNone(b._pending)
            self.assertEqual((bytes(b.memory.storage), bytes(b.edges), bytes(b.knots), bytes(b.anchors)), before)

    def test_bank_rejects_foreign_header_owner_and_never_accepts_caller_decoded_data(self):
        from temporal_consumer_packet_reference import ConsumerBankInput
        b = bank(1)
        h = b.reserve()
        w = event(1, .1, [(h, None, 1.)])
        packet = bank_packet(w, admissions(w, [h]))
        owner = ConsumerBankInput(packet)
        changed = admissions(w, [h]);changed[h]['compression_error'] = 1.
        other = ConsumerBankInput(bank_packet(w, changed))
        for invalid in (other, owner._header):
            with self.assertRaises(ValueError):
                b.advance_packet(packet, 3., 1, invalid)
        self.assertFalse(b.advance_packet(packet, 3., 1, owner))
        self.assertIs(b._pending['input'], owner)
        self.assertTrue(b.advance_packet(packet, 3., 8, owner))
        self.assertEqual(b.memory.record(0)['membership_total'], 1.)


if __name__ == '__main__':
    unittest.main()
