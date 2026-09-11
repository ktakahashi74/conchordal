"""Finite-history cache oracles and causal single-worker scheduling fixtures."""

import copy
import math
from pathlib import Path
import random
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import temporal_descriptor_reference as descriptor
import temporal_matcher_reference as matcher
import temporal_query_scheduler_reference as ref
from test_evaluate_temporal_occurrence_reference import record, paths, hop
import temporal_occurrence_reference as occurrence


def raw(index, generation=4, epoch=1, intervals=None, **extra):
    lo, hi = index*10, (index+1)*10
    intervals = [(lo, hi)] if intervals is None else intervals
    return {'epoch': epoch, 'generation': generation, 'sample_rate': 100,
            'sample_start': lo, 'sample_end': hi, 'start': lo/100, 'end': hi/100,
            'time': hi/100, 'raw_support_start': lo/100, 'raw_support_end': hi/100,
            'available_end': hi/100, 'known_sample_intervals': intervals,
            'observed': bool(intervals), 'gap': not intervals,
            'values': [1.]*10 if intervals else [None]*10, **extra}


def span(index=0, generation=4, epoch=1, **extra):
    s = descriptor.SpanDescriptor(epoch, generation, 100, index*10, 10, index/10, [1.]*10)
    r = raw(index, generation, epoch, **extra)
    s.push(r, max(r['available_end'], r['end']))
    return s


def scheduler(groups=1, cadence=.1):
    s = ref.QueryScheduler(1, 100, 10, groups=groups, cadence=cadence)
    for slot in range(groups):
        s.bind(slot, 4+slot, 0.)
    return s


def entry(identity=10, generation=2, cost=0., **extra):
    return {'episode_id': identity, 'episode_generation': generation, 'cost': cost,
            'similarity': None if cost is None else math.exp(-cost),
            'anchor_cap_loss': False, 'bound_anchors': 0, 'unidentified': [False, False], **extra}


def result(query_id=1, end=1., generation=4, **extra):
    return {'epoch': 1, 'generation': generation, 'query_id': query_id,
            'occurrence_id': 1, 'support_id': 1001, 'query_end': end,
            'supporting_audio_end': end, 'available_end': end, 'completed_at': end,
            'complete': True, 'superseded': False, 'coarse_entries': [entry()], **extra}


def packed(out, bindings=None):
    data = bytearray(ref.SNAPSHOT_BYTES)
    ref.pack_coarse(out, {(10, 2): (0, 1010)} if bindings is None else bindings, data, out['completed_at'])
    return data


def finish(s, job, at, **extra):
    q = job['query']
    out = {k: q[k] for k in ('epoch', 'generation', 'query_id', 'occurrence_id', 'support_id',
                            'query_end', 'supporting_audio_end', 'available_end')}
    out.update(completed_at=at, complete=True, superseded=False, coarse_entries=[entry()])
    out.update(extra)
    return s.finish(job['ticket'], out, {(10, 2): (0, 1010)}, at)


class CoarseCacheTests(unittest.TestCase):
    def test_exact_layout_masks_and_generation_handle_at_last_slot(self):
        out = result(coarse_entries=[entry(cost=math.log(2)), entry(11, 3, None, anchor_cap_loss=True)])
        p = packed(out, {(10, 2): (255, 1010), (11, 3): (0, 1111)})
        self.assertEqual(len(p), 6272)
        self.assertEqual(struct.unpack_from('<5Q3d', p), (1, 4, 1, 1, 1001, 1., 1., 1.))
        self.assertEqual(struct.unpack_from('<Qdd', p, 128+255*24), (1010, math.log(2), .5))
        self.assertEqual(p[95], 128)
        self.assertEqual(p[96], 1)
        c = ref.CoarseCache(1, 4)
        c.insert(p)
        row = c.snapshots({255: 1010, 0: 1111}, 1.)[0]
        self.assertEqual(row['entries'], {1010: math.log(2), 1111: None})
        self.assertEqual(row['approximate_handles'], [1111])
        self.assertEqual(c.snapshots({255: 2010}, 1.)[0]['entries'], {})

    def test_missing_binding_remains_absent_and_none_audio_stays_unknown(self):
        p = bytearray(ref.SNAPSHOT_BYTES)
        self.assertEqual(ref.pack_coarse(result(supporting_audio_end=None), {}, p, 1.), 1)
        c = ref.CoarseCache(1, 4)
        c.insert(p)
        row = c.snapshots({0: 1010}, 1.)[0]
        self.assertIsNone(row['supporting_audio_end'])
        self.assertEqual(row['entries'], {})

    def test_oldest_endpoint_then_id_eviction_and_stale_replay(self):
        c = ref.CoarseCache(1, 4)
        for i in range(1, 10):
            c.insert(packed(result(i, 1.)))
        self.assertEqual([r['query_id'] for r in c.snapshots({}, 1.)], list(range(2, 10)))
        self.assertEqual(c.evictions, 1)
        old = bytes(c.storage)
        self.assertFalse(c.insert(packed(result(1, 1.))))
        self.assertEqual(bytes(c.storage), old)
        self.assertFalse(c.insert(packed(result(10, .5))))
        self.assertEqual([r['query_id'] for r in c.snapshots({}, 1.)], list(range(2, 10)))
        self.assertEqual((c.stale, c.evictions), (1, 2))

    def test_exact_replay_noop_conflict_atomic_and_original_cut(self):
        c = ref.CoarseCache(1, 4)
        p = packed(result(completed_at=2.))
        c.insert(p)
        self.assertEqual(c.snapshots({}, 1.5), [])
        self.assertFalse(c.insert(p))
        before = bytes(c.storage)
        with self.assertRaises(ValueError):
            c.insert(packed(result(completed_at=2.1)))
        self.assertEqual(bytes(c.storage), before)
        frozen = c.snapshots({0: 1010}, 2.)[0]
        c.clear(1, 5)
        self.assertFalse(c.insert(p))
        self.assertEqual(c.snapshots({0: 1010}, 2.), [])
        self.assertEqual(frozen['entries'], {1010: 0.})

    def test_packing_rejects_aliases_future_unsupported_or_nonfinite_before_write(self):
        data = bytearray([19]*ref.SNAPSHOT_BYTES)
        for out, binding in [
            (result(complete=False), {(10, 2): (0, 1010)}),
            (result(superseded=True), {}), (result(completed_at=.9), {}),
            (result(supporting_audio_end=1.1), {}), (result(query_id=2**64), {}),
            (result(coarse_entries=[entry(), entry()]), {}),
            (result(coarse_entries=[entry(cost=math.inf)]), {}),
            (result(), {(10, 2): (0, 0)}), (result(), {(10, 2): (256, 1)}),
            (result(), {(10, 2): (0, 1), (11, 2): (0, 2)}),
            (result(), {(10, 2): (0, 1), (11, 2): (1, 1)}),
        ]:
            with self.subTest(out=out, binding=binding), self.assertRaises(ValueError):
                ref.pack_coarse(out, binding, data, 1.)
            self.assertEqual(data, bytearray([19]*ref.SNAPSHOT_BYTES))

    def test_random_stream_matches_independent_full_history_top_eight(self):
        rng = random.Random(47014)
        for _ in range(100):
            c, history = ref.CoarseCache(1, 4), []
            for i in range(1, 81):
                end = rng.randrange(1, 33)/8
                p = packed(result(i, end))
                c.insert(p)
                history.append((end, i))
                expected = sorted(history)[-8:]
                rows = c.snapshots({}, 10.)
                self.assertEqual([(r['support_end'], r['query_id']) for r in rows], expected)
                self.assertLessEqual(sum(c.occupied), 8)
                self.assertEqual(len(c.storage), 8*6272)
                if rng.random() < .2:
                    self.assertFalse(c.insert(p))
                    self.assertEqual([(r['support_end'], r['query_id']) for r in c.snapshots({}, 10.)], expected)


class QuerySchedulerTests(unittest.TestCase):
    def test_cadence_uses_union_of_actual_samples_and_counts_known_silence(self):
        s = scheduler()
        a = raw(0, intervals=[(0, 3), (2, 5), (0, 3)], values=[None]*10)
        s.observe(0, a, .1)
        s.submit(0, span(), 1, 1, 1001, .1)
        self.assertIsNone(s.take(.1))
        self.assertFalse(s.observe(0, copy.deepcopy(a), .1))
        s.observe(0, raw(1, intervals=[]), .2)
        self.assertIsNone(s.take(.2))
        s.observe(0, raw(2, intervals=[(20, 25)], values=[None]*10), .3)
        job = s.take(.3)
        self.assertEqual(job['query']['query_end'], .1)
        self.assertEqual(s.status(.3)['groups'][0]['acquired_samples'], 10)
        self.assertTrue(finish(s, job, .3))
        s.observe(0, raw(3, values=[None, None, -19.9315685693]+[None]*7), .4)
        s.submit(0, span(3), 2, 2, 1002, .4)
        self.assertIsNotNone(s.take(.4))

    def test_gap_and_wall_time_do_not_bypass_cadence_or_create_catchup_jobs(self):
        s = scheduler()
        s.submit(0, span(), 1, 1, 1001, .1)
        self.assertIsNone(s.take(50.))
        s.observe(0, raw(499), 50.)
        job = s.take(50.)
        self.assertIsNotNone(job)
        finish(s, job, 50.)
        s.submit(0, span(499), 2, 2, 1002, 50.)
        self.assertIsNone(s.take(100.))
        self.assertEqual(s.samples[0], 10)

    def test_latest_pending_replacement_preserves_running_query_and_queue_age(self):
        s = scheduler()
        s.observe(0, raw(0), .1)
        live = span()
        s.submit(0, live, 1, 1, 1001, .1)
        job = s.take(.1)
        for i in (1, 2):
            s.observe(0, raw(i), (i+1)/10)
            live.push(raw(i), (i+1)/10)
            s.submit(0, live, i+1, 1, 1001, (i+1)/10)
        state = s.status(.4)
        self.assertEqual(state['active_query'], 1)
        self.assertEqual(state['groups'][0]['pending_query'], 3)
        self.assertEqual(state['counts']['superseded_pending'], 1)
        self.assertAlmostEqual(state['support_age'], .3)
        self.assertIsNone(s.take(.4))
        self.assertTrue(finish(s, job, .4))
        self.assertEqual(s.caches[0].snapshots({0: 1010}, .4)[0]['query_id'], 1)
        latest = s.take(.4)
        self.assertEqual(latest['query']['query_id'], 3)
        self.assertEqual(len(job['query']['knots']), 1)
        job['query']['knots'][0]['values'][0] = 999
        self.assertEqual(latest['query']['knots'][0]['values'][0], 1.)

    def test_delayed_worker_skips_missed_cycles_and_does_not_burst_same_group(self):
        s = scheduler()
        for i in range(10):
            s.observe(0, raw(i), (i+1)/10)
        s.submit(0, span(9), 1, 1, 1001, 1.)
        job = s.take(1.)
        self.assertEqual(s.next_due[0], 110)
        finish(s, job, 1.)
        s.submit(0, span(9), 2, 1, 1001, 1.)
        self.assertIsNone(s.take(1.))
        s.observe(0, raw(10), 1.1)
        self.assertEqual(s.take(1.1)['query']['query_id'], 2)

    def test_eight_groups_share_one_worker_and_replacement_keeps_fair_order(self):
        s = scheduler(8)
        for slot in range(8):
            s.observe(slot, raw(0, 4+slot), .1)
            s.submit(slot, span(generation=4+slot), 1, 1, 1001, .1)
        s.submit(0, span(), 2, 1, 1001, .1)
        order = []
        for _ in range(8):
            job = s.take(.1)
            order.append(job['group_slot'])
            self.assertIsNone(s.take(.1))
            finish(s, job, .1)
        self.assertEqual(order, list(range(8)))
        self.assertIsNone(s.take(.1))
        self.assertEqual(s.counts['dispatched'], 8)

    def test_retirement_invalidates_work_without_launching_second_worker(self):
        s = scheduler()
        s.observe(0, raw(0), .1)
        s.submit(0, span(), 1, 1, 1001, .1)
        job = s.take(.1)
        s.submit(0, span(), 2, 1, 1001, .1)
        s.bind(0, 5, .1)
        with self.assertRaises(ValueError):
            s.bind(0, 4, .1)
        s.observe(0, raw(1, 5), .2)
        s.submit(0, span(1, 5), 1, 1, 1001, .2)
        self.assertIsNone(s.take(.2))
        self.assertFalse(finish(s, job, .3))
        self.assertEqual(s.caches[0].snapshots({0: 1010}, .3), [])
        new = s.take(.3)
        self.assertEqual(new['query']['generation'], 5)
        self.assertFalse(s.finish(job['ticket'], None, {}, .3))
        self.assertIsNotNone(s.active_slot)
        self.assertTrue(finish(s, new, .3))
        self.assertEqual(s.counts['retired_pending'], 1)
        self.assertEqual(s.counts['invalidated_active'], 1)

    def test_epoch_restart_preserves_stale_ticket_but_resets_audio_clock(self):
        s = scheduler()
        s.observe(0, raw(0), .1)
        s.submit(0, span(), 1, 1, 1001, .1)
        old = s.take(.1)
        s.restart(2, 100, 10, 0.)
        s.bind(0, 4, 0.)
        s.observe(0, raw(0, epoch=2), .1)
        s.submit(0, span(epoch=2), 1, 1, 1001, .1)
        self.assertIsNone(s.take(.1))
        self.assertFalse(s.finish(old['ticket'], None, {}, .1))
        new = s.take(.1)
        self.assertEqual(new['query']['epoch'], 2)
        self.assertGreater(new['ticket'], old['ticket'])
        with self.assertRaises(ValueError):
            s.restart(1, 100, 10, .1)

    def test_failure_and_incomplete_result_release_worker_without_snapshot(self):
        for mode in ('failure', 'incomplete', 'superseded'):
            s = scheduler()
            s.observe(0, raw(0), .1)
            s.submit(0, span(), 1, 1, 1001, .1)
            job = s.take(.1)
            if mode == 'failure':
                self.assertFalse(s.finish(job['ticket'], None, {}, .2))
            else:
                self.assertFalse(finish(s, job, .2, **({'complete': False} if mode == 'incomplete' else {'superseded': True})))
            self.assertIsNone(s.active_slot)
            self.assertEqual(s.counts['incomplete'], 1)
            self.assertEqual(sum(s.caches[0].occupied), 0)

    def test_foreign_future_and_conflicting_completions_leave_active_state_untouched(self):
        s = scheduler()
        s.observe(0, raw(0), .1)
        s.submit(0, span(), 1, 1, 1001, .1)
        job = s.take(.1)
        expected = s.status(.1)
        for key, value in [('generation', 5), ('occurrence_id', 7), ('available_end', .05),
                           ('supporting_audio_end', .05), ('completed_at', .09)]:
            out = result(end=.1, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                s.finish(job['ticket'], out, {}, .1)
            self.assertEqual(s.status(.1), expected)
        self.assertTrue(finish(s, job, .1))

    def test_invalid_and_reordered_observations_do_not_change_clock(self):
        s = scheduler()
        s.observe(0, raw(1), .2)
        old = s.status(.2)
        for r in [raw(0), raw(1, available_end=.25), raw(2, generation=5),
                  raw(2, intervals=[(19, 30)]), raw(2, intervals=[], observed=True)]:
            with self.assertRaises(ValueError):
                s.observe(0, r, .3)
            self.assertEqual(s.status(.2), old)

    def test_older_endings_and_replayed_query_ids_cannot_displace_latest(self):
        s = scheduler()
        s.submit(0, span(2), 2, 2, 1002, .3)
        self.assertFalse(s.submit(0, span(), 3, 1, 1001, .3))
        self.assertFalse(s.submit(0, span(2), 2, 2, 1002, .3))
        self.assertEqual(s.pending[0].metadata()['query_id'], 2)
        self.assertEqual(s.counts['older_query_loss'], 1)
        self.assertEqual(s.counts['stale_query'], 1)

    def test_half_double_cadence_matches_acquired_time_oracle(self):
        for cadence, expected in [(.05, 8), (.1, 4), (.2, 2)]:
            s = ref.QueryScheduler(1, 100, 5, cadence=cadence, groups=1)
            s.bind(0, 4, 0.)
            live = descriptor.SpanDescriptor(1, 4, 100, 0, 5, 0., [1.]*10)
            for i in range(10):
                lo, hi = i*5, (i+1)*5
                acquired = i not in (3, 7)
                r = raw(0, sample_start=lo, sample_end=hi, start=lo/100, end=hi/100, time=hi/100,
                        raw_support_start=lo/100, raw_support_end=hi/100, available_end=hi/100,
                        known_sample_intervals=[(lo, hi)] if acquired else [], observed=acquired,
                        gap=not acquired, values=[1.]*10 if acquired else [None]*10)
                live.push(r, hi/100)
                s.observe(0, r, hi/100)
                s.submit(0, live, i+1, 1, 1001, hi/100)
                job = s.take(hi/100)
                if job:
                    finish(s, job, hi/100)
            self.assertEqual(s.counts['dispatched'], expected)
            self.assertEqual(s.samples[0], 40)

    def test_fixed_payload_and_detached_exports_survive_many_replacements(self):
        s = ref.QueryScheduler(1)
        sizes = s.payload_bytes()
        self.assertEqual(sizes['coarse_snapshots'], 401408)
        self.assertEqual(sizes['query_buffers_including_active_and_scratch'], 10*(40960+176))
        self.assertEqual(sizes['coarse_scratch'], 6272)
        q = ref.QuerySlot(128)
        live = span()
        before = live.snapshot()
        for i in range(100):
            q.capture(live, .1, i, 1, 1001)
        self.assertEqual(live.snapshot(), before)
        exported = q.export()
        live.push(raw(1), .2)
        self.assertEqual(exported['knots'][0]['end'], .1)
        self.assertEqual(s.payload_bytes(), sizes)


class SchedulerIntegrationTests(unittest.TestCase):
    def test_early_worker_finish_delivered_after_deadline_cannot_backdate_cache(self):
        s = scheduler()
        s.observe(0, raw(9), 1.)
        s.submit(0, span(9), 1, 1, 1001, 1.)
        job = s.take(1.)
        out = result(end=1., completed_at=1.25)
        self.assertTrue(s.finish(job['ticket'], out, {(10, 2): (0, 1010)}, 2.))
        snapshots = s.caches[0].snapshots({0: 1010}, 2.)
        self.assertEqual(snapshots[0]['available_end'], 2.)
        ledger = occurrence.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        self.assertIsNone(ledger.advance(2., [], snapshots)[0]['coarse_snapshot'])

    def test_real_matcher_cache_freezes_once_at_original_commit_deadline(self):
        s = scheduler()
        stored = span()
        sealed = {'epoch': 1, 'start': 0., 'support_end': .1, 'committed_at': .6,
                  'delivered_at': .6, 'sequence': 1, 'occurrence_id': 8, 'support_id': 1008,
                  'support': {'available_support': 1.}}
        episode = matcher.committed_episode(stored, sealed, 10, 2)
        cue = span(9)
        s.observe(0, raw(9), 1.)
        s.submit(0, cue, 1, 1, 1001, 1.)
        job = s.take(1.)
        out = matcher.match_query(job['query'], [episode], 1.25)
        self.assertTrue(s.finish(job['ticket'], out, {(10, 2): (0, 1010)}, 1.25))
        ledger = occurrence.OccurrenceLedger(1)
        ledger.stage(record(start=.9, end=1., support=1.), paths((1010, 100, 1.)), 1.)
        writes = ledger.advance(1.5, [hop(1., 1.5)], s.caches[0].snapshots({0: 1010}, 1.5))
        self.assertEqual(writes[0]['coarse_snapshot']['entries'], {1010: 0.})
        self.assertEqual(writes[0]['committed_at'], 1.5)
        frozen = copy.deepcopy(ledger.snapshot()['writes'])
        s.bind(0, 5, 1.5)
        self.assertEqual(s.caches[0].snapshots({0: 1010}, 1.5), [])
        self.assertEqual(ledger.snapshot()['writes'], frozen)
        self.assertIsNotNone(ledger.section_record(1, 0))

    def test_late_result_remains_available_for_diagnostics_but_not_old_commit(self):
        s = scheduler()
        s.observe(0, raw(9), 1.)
        s.submit(0, span(9), 1, 1, 1001, 1.)
        job = s.take(1.)
        finish(s, job, 2.)
        ledger = occurrence.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        write = ledger.advance(2., [], s.caches[0].snapshots({0: 1010}, 2.))[0]
        self.assertIsNone(write['coarse_snapshot'])
        self.assertEqual(write['committed_at'], 1.5)


if __name__ == '__main__':
    unittest.main()
