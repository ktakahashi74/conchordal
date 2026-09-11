"""Independent physical sample unions and chronological retention/rate oracles."""

import copy
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import temporal_memory_reference as ref
import temporal_cognition_reference as cognition
import temporal_occurrence_reference as occurrence
import temporal_matcher_reference as matcher
from test_evaluate_temporal_occurrence_reference import record, paths, hop
from test_evaluate_temporal_query_scheduler_reference import scheduler, span, raw as descriptor_raw


def raw(index, known=True, **extra):
    lo, hi = index*10, (index+1)*10
    return {'epoch': 1, 'sample_rate': 100, 'sample_start': lo, 'sample_end': hi,
            'available_end': hi/100, 'observed': known,
            'known_sample_intervals': [(lo, hi)] if known else [], **extra}


def clock(steps=30, capacity=128, missing=()):
    c = ref.GapClock(1, 100, 10, capacity)
    for i in range(steps):
        c.observe(raw(i, i not in missing), (i+1)/10)
    return c


def write(sequence, end, assignments=None, unknown=0., unassigned=0., costs=None, **extra):
    assignments = {10: 1.} if assignments is None else assignments
    available = math.fsum(assignments.values())+unknown+unassigned
    snapshot = None if costs is None else {
        'epoch': 1, 'generation': 4, 'query_id': sequence, 'occurrence_id': sequence,
        'support_id': 1000+sequence, 'support_end': end, 'supporting_audio_end': end,
        'available_end': end+.5, 'completed': True, 'superseded': False, 'entries': costs}
    return {'epoch': 1, 'sequence': sequence, 'occurrence_id': sequence,
            'support_id': 1000+sequence, 'start': max(0., end-.1), 'support_end': end, 'committed_at': end+.5,
            'delivered_at': end+.5, 'coarse_snapshot': snapshot,
            'support': {'episodes': assignments, 'unknown_episode_support': unknown,
                        'unassigned_available_support': unassigned, 'available_support': available}, **extra}


def memory(c=None, **extra):
    return ref.EpisodeRetention(clock(20) if c is None else c, tau_sec=20., kappa=4., strength_max=3.,
                               r_max=10., capacity=4, rate_capacity=32, **extra)


class GapClockTests(unittest.TestCase):
    def test_partial_sample_union_silence_and_repeat_use_one_physical_clock(self):
        c = ref.GapClock(1, 100, 10)
        r = raw(0, known_sample_intervals=[(0, 3), (2, 5), (0, 3)])
        c.observe(r, .1)
        self.assertAlmostEqual(c.missing, .05)
        self.assertFalse(c.observe(copy.deepcopy(r), .1))
        self.assertEqual(c.prefix(.025), {'lower': 0., 'upper': 0., 'history_lost': False})
        self.assertAlmostEqual(c.prefix(.075)['lower'], .025)
        c.observe(raw(1), .2)
        self.assertAlmostEqual(c.prefix(.2)['lower'], .05)
        self.assertEqual(c.count, 2)

    def test_skipped_hops_and_explicit_gap_are_not_silence_or_double_counted(self):
        c = ref.GapClock(1, 100, 10)
        c.observe(raw(3), .4)
        self.assertEqual(c.prefix(.2), {'lower': .2, 'upper': .2, 'history_lost': False})
        c.gap(100, 1.)
        self.assertAlmostEqual(c.missing, .9)
        self.assertAlmostEqual(c.prefix(.65)['lower'], .55)
        c.observe(raw(10), 1.1)
        self.assertAlmostEqual(c.missing, .9)

    def test_old_mask_eviction_returns_bounds_containing_original_gap_mass(self):
        c = clock(10, capacity=2, missing=(0, 2, 4, 6, 8))
        self.assertEqual(c.evictions, 8)
        b = c.prefix(.3)
        self.assertTrue(b['history_lost'])
        self.assertLessEqual(b['lower'], .2)
        self.assertGreaterEqual(b['upper'], .2)
        self.assertAlmostEqual(c.prefix(1.)['lower'], .5)
        self.assertEqual(len(c.storage), 2*(32+2))

    def test_all_random_prefixes_enclose_independent_uncompressed_sample_union(self):
        rng = random.Random(47015)
        for capacity in (3, 8, 32):
            for _ in range(25):
                c = ref.GapClock(1, 100, 10, capacity)
                acquired = []
                for i in range(20):
                    bits = [rng.random() < .7 for _ in range(10)]
                    acquired.extend(bits)
                    c.observe(raw(i, any(bits), known_sample_intervals=[(i*10+j, i*10+j+1)
                              for j, bit in enumerate(bits) if bit]), (i+1)/10)
                    for sample in range(len(acquired)+1):
                        actual = sum(not x for x in acquired[:sample])/100
                        bounds = c.prefix(sample/100)
                        self.assertLessEqual(bounds['lower'], actual+1e-12)
                        self.assertGreaterEqual(bounds['upper']+1e-12, actual)
                        if not bounds['history_lost']:
                            self.assertAlmostEqual(bounds['lower'], actual, delta=1e-12)
                            self.assertEqual(bounds['lower'], bounds['upper'])

    def test_foreign_future_reordered_and_conflicting_hops_do_not_rewrite_clock(self):
        c = ref.GapClock(1, 100, 10)
        c.observe(raw(1), .2)
        before = bytes(c.storage), c.end, c.missing
        for r in (raw(0), raw(1, False), raw(2, epoch=2), raw(2, available_end=1.),
                  raw(2, known_sample_intervals=[(19, 30)])):
            with self.assertRaises(ValueError):
                c.observe(r, .3)
            self.assertEqual((bytes(c.storage), c.end, c.missing), before)
        with self.assertRaises(ValueError):
            c.prefix(.3)


class RateWindowTests(unittest.TestCase):
    def test_development_envelope_uses_complete_rate_maximum_and_declared_floor(self):
        w = ref.InterferenceWindow(1, 8)
        rows = [w.observe(0., 1, [.25], [1], 10.), w.observe(.1, 2, [1.75], [1], 10.)]
        self.assertEqual(ref.fit_gap_rate_envelope(rows)['r_max'], 4.)
        self.assertEqual(ref.fit_gap_rate_envelope(rows[:1])['r_max'], 1.)
        for values in ([], [{**rows[0], 'envelope_unverified': True}],
                       [{**rows[0], 'rate_history_unknown': True}]):
            with self.assertRaises(ValueError):
                ref.fit_gap_rate_envelope(values)

    def test_exact_one_second_boundary_and_reinforcement_does_not_erase_rate(self):
        w = ref.InterferenceWindow(1, 8)
        w.observe(0., 1, [.75], [1], 2.)
        self.assertEqual(w.observe(.5, 2, [.5], [1], 2.)['largest_retained_one_second_increment'], 1.25)
        self.assertEqual(w.observe(1., 3, [.25], [1], 2.)['largest_retained_one_second_increment'], .75)
        self.assertFalse(w.invalid)

    def test_invalid_envelope_is_sticky_and_overflow_stays_distinct_unknown(self):
        w = ref.InterferenceWindow(1, 2)
        w.observe(0., 1, [1.], [1], 1.)
        self.assertTrue(w.observe(.1, 2, [1.], [1], 1.)['envelope_invalid'])
        self.assertTrue(w.observe(3., 3, [0.], [1], 1.)['envelope_invalid'])
        v = ref.InterferenceWindow(1, 2)
        for i in range(3):
            v.observe(i*.1, i+1, [.5], [1], 10.)
        self.assertEqual(v.losses, 1)
        self.assertFalse(v.invalid)
        later = v.observe(3., 4, [0.], [1], 10.)
        self.assertFalse(later['rate_history_unknown'])
        self.assertTrue(later['envelope_unverified'])

    def test_slot_reuse_does_not_subtract_a_retired_generations_increment(self):
        w = ref.InterferenceWindow(1, 8)
        w.observe(0., 1, [1.], [1], 10.)
        w.totals[0] = 0.
        w.observe(.5, 5, [2.], [5], 10.)
        w.observe(1., 6, [0.], [5], 10.)
        self.assertEqual(w.totals[0], 2.)
        w.observe(1.5, 7, [0.], [5], 10.)
        self.assertEqual(w.totals[0], 0.)

    def test_random_windows_match_independent_all_event_sums(self):
        rng = random.Random(47016)
        for _ in range(50):
            window = ref.InterferenceWindow(4, 64)
            history, time = [], 0.
            for sequence in range(1, 101):
                time += rng.choice([0., .05, .2, .75, 1.])
                inc = [rng.randrange(5)/4 for _ in range(4)]
                history.append((time, inc))
                window.observe(time, sequence, inc, [1]*4, 1e6)
                expected = [math.fsum(v[i] for t, v in history if time-1 < t <= time) for i in range(4)]
                for actual, value in zip(window.totals, expected):
                    self.assertAlmostEqual(actual, value, delta=1e-12)
            self.assertEqual(window.losses, 0)


class RetentionTests(unittest.TestCase):
    def test_original_time_delayed_delivery_and_repeat_do_not_refresh_or_add_strength(self):
        m = memory()
        w = write(1, .1, {10: .5}, delivered_at=2.)
        m.apply(w, 2., [10])
        b = m.availability(0, 2.)
        self.assertAlmostEqual(b['log_upper'], math.log(.5)-1.9/20)
        self.assertEqual(b['log_lower'], b['log_upper'])
        saved = bytes(m.storage)
        self.assertFalse(m.apply(copy.deepcopy(w), 2.))
        self.assertEqual(bytes(m.storage), saved)
        self.assertEqual(m.record(0)['strength'], .5)

    def test_fractional_recurrence_resets_old_uncertainty_then_adds_current_competitor_share(self):
        m = memory(clock(missing=(3, 4, 5)))
        m.apply(write(1, .1), 3., [10])
        m.apply(write(2, .2, {20: 1.}, costs={10: 0.}), 3., [20])
        m.apply(write(3, 1., {10: .25, 20: .5}, unknown=.25, costs={10: math.log(2), 20: 0.}), 3.)
        a, b = m.record(0), m.availability(0, 3.)
        self.assertEqual(a['strength'], 1.25)
        self.assertEqual(a['membership_total'], 1.25)
        self.assertAlmostEqual(a['interference_lower'], .25)
        self.assertAlmostEqual(a['interference_upper'], .375)
        self.assertEqual(b['missing_seconds_upper'], 0.)
        self.assertAlmostEqual(b['log_lower'], math.log(1.25)-2/20-.375/4)

    def test_same_endpoint_recurrence_is_separate_from_first_occurrence_interference(self):
        m = memory(clock(10))
        m.apply(write(1, .1, {10: .5}), 1., [10])
        m.apply(write(2, .1, {10: .25, 20: .5}, costs={10: 0.}), 1., [20])
        self.assertEqual(m.record(0)['strength'], .75)
        self.assertEqual(m.record(0)['interference_lower'], 0.)
        self.assertEqual(m.record(1)['strength'], .5)

    def test_cap_does_not_cap_membership_totals_or_make_retrieval_reinforcement(self):
        m = memory()
        for i in range(1, 6):
            m.apply(write(i, i/10), 2., [10] if i == 1 else [])
        self.assertEqual(m.record(0)['strength'], 3.)
        self.assertEqual(m.record(0)['membership_total'], 5.)
        before = bytes(m.storage)
        for _ in range(10):
            m.recognition({10: 0.}, 0., 2.)
        self.assertEqual(bytes(m.storage), before)

    def test_common_gap_is_added_once_for_each_unrehearsed_episode(self):
        m = memory(clock(20, missing=(2, 3, 4)))
        m.apply(write(1, .1, {10: .5, 20: .5}), 2., [10, 20])
        m.apply(write(2, .2, {}, unknown=1.), 2.)
        a, b = m.availability(0, 2.), m.availability(1, 2.)
        self.assertAlmostEqual(a['missing_seconds_upper'], .3)
        self.assertAlmostEqual(b['gap_interference_upper'], 3.)
        self.assertEqual(a['observed_interference'], 0.)
        self.assertEqual(a['coarse_unknown_interference'], 1.)
        self.assertAlmostEqual(a['log_upper']-a['log_lower'], (1+3)/4)

    def test_invalid_rate_widens_gap_to_zero_without_erasing_known_interference(self):
        c = clock(20, missing=range(3, 10))
        m = ref.EpisodeRetention(c, 20., 4., 3., 1., capacity=2, rate_capacity=16)
        m.apply(write(1, .1), 2., [10])
        m.apply(write(2, .2, {20: 1.}, costs={10: 0.}), 2., [20])
        m.apply(write(3, .3, {20: 1.}, costs={10: 0.}), 2.)
        b = m.availability(0, 2.)
        self.assertTrue(b['envelope_invalid'])
        self.assertEqual(b['log_lower'], -math.inf)
        self.assertAlmostEqual(b['log_upper'], -1.9/20-2/4)
        self.assertEqual(m.recognition({10: 0.}, 0., 2.)['lower'], 0.)
        m.apply(write(4, 1.1, {10: 1.}), 2.)
        renewed = m.availability(0, 2.)
        self.assertEqual(renewed['missing_seconds_upper'], 0.)
        self.assertEqual(renewed['log_lower'], renewed['log_upper'])

    def test_late_write_outside_clock_history_preserves_age_and_bounded_missingness(self):
        c = clock(30, capacity=2, missing=range(1, 30, 2))
        m = memory(c)
        m.apply(write(1, .1, delivered_at=3.), 3., [10])
        b = m.availability(0, 3.)
        self.assertEqual(b['elapsed_sec'], 2.9)
        self.assertTrue(b['clock_history_lost'])
        self.assertLessEqual(b['missing_seconds_lower'], 1.5+1e-12)
        self.assertGreaterEqual(b['missing_seconds_upper'], 1.5-1e-12)
        self.assertLessEqual(b['missing_seconds_upper'], 2.9)

    def test_eviction_uses_upper_bound_and_does_not_revive_absent_links(self):
        m = memory()
        m.apply(write(1, .1, {10: .5, 20: .5}), 2., [10, 20])
        m.apply(write(2, .2, {30: 1.}, costs={10: 0., 20: None}), 2.)
        self.assertEqual(m.eviction_candidate(2.), 0)
        self.assertEqual(m.evict(0), 10)
        m.apply(write(3, .3, {10: 1.}), 2.)
        self.assertEqual(m.record(0)['handle'], 0)
        m.apply(write(4, .4, {40: 1.}), 2., [40])
        self.assertEqual(m.record(0)['handle'], 40)
        self.assertEqual(m.record(0)['strength'], 1.)

    def test_capacity_requires_explicit_graph_eviction_and_rejects_partial_write(self):
        m = ref.EpisodeRetention(clock(20), 20., 4., 3., 10., capacity=1)
        m.apply(write(1, .1), 2., [10])
        before = bytes(m.storage)
        with self.assertRaises(BufferError):
            m.apply(write(2, .2, {20: 1.}), 2., [20])
        self.assertEqual(bytes(m.storage), before)
        self.assertEqual(m.sequence, 1)
        m.evict(m.eviction_candidate(2.))
        m.apply(write(2, .2, {20: 1.}), 2., [20])
        self.assertEqual(m.record(0)['handle'], 20)

    def test_bad_sequence_support_epoch_or_backdating_cannot_change_metadata(self):
        m = memory(clock(10))
        m.apply(write(1, .1), 1., [10])
        before = bytes(m.storage)
        for w in [write(3, .3), write(2, .05), write(2, .2, epoch=2),
                  write(2, .2, {10: 2.}), write(2, .2, costs={10: -1.}), write(1, .1, {10: .5})]:
            with self.assertRaises(ValueError):
                m.apply(w, 1.)
            self.assertEqual(bytes(m.storage), before)
            self.assertEqual(m.sequence, 1)
        with self.assertRaises(ValueError):
            m.availability(0, .9)

    def test_random_sealed_stream_matches_independent_scalar_state(self):
        rng = random.Random(47017)
        for _ in range(50):
            c = clock(100)
            m = ref.EpisodeRetention(c, 20., 4., 3., 100., capacity=2, rate_capacity=64)
            m.apply(write(1, .1, {10: .5, 20: .5}), 10., [10, 20])
            strength, last, low, high = [.5, .5], [.1, .1], [0., 0.], [0., 0.]
            for seq in range(2, 61):
                end = seq/10
                a, b = rng.randrange(3)/4, rng.randrange(3)/4
                assignments = {h: x for h, x in [(10, a), (20, b)] if x}
                missing = 1-a-b
                costs = {h: rng.choice([None, 0., math.log(2)]) for h in (10, 20)}
                m.apply(write(seq, end, assignments, unknown=missing, costs=costs), 10.)
                for j, (own, other, h) in enumerate(((a, b, 10), (b, a, 20))):
                    if own:
                        strength[j] = min(3., strength[j]+own)
                        last[j], low[j], high[j] = end, 0., 0.
                    similarity = None if costs[h] is None else math.exp(-costs[h])
                    low[j] += 0. if similarity is None else other*similarity
                    high[j] += other+missing if similarity is None else (other+missing)*similarity
                    actual = m.availability(j, 10.)
                    self.assertAlmostEqual(actual['log_upper'], math.log(strength[j])-(10-last[j])/20-low[j]/4)
                    self.assertAlmostEqual(actual['log_lower'], math.log(strength[j])-(10-last[j])/20-high[j]/4)

    def test_payload_stays_fixed_and_new_epoch_starts_empty(self):
        c = ref.GapClock(1)
        m = ref.EpisodeRetention(c, 20., 4., 3., 10.)
        sizes = m.payload_bytes()
        self.assertEqual(sizes['episode_metadata'], 65536)
        self.assertEqual(sizes['gap_clock'], 12288)
        self.assertEqual(sizes['rate_window'], 297216)
        self.assertEqual(sum(v for v in sizes.values() if isinstance(v, int)), 446720)
        fresh = ref.EpisodeRetention(ref.GapClock(2, 100, 10), 20., 4., 3., 10.)
        with self.assertRaises(ValueError):
            fresh.apply(write(1, .1), 1., [10])
        self.assertEqual(fresh.record(0)['handle'], 0)
        self.assertEqual(m.payload_bytes(), sizes)

    def test_forged_heard_support_in_known_acquisition_gap_is_rejected(self):
        m = memory(clock(10, missing=(0,)))
        with self.assertRaisesRegex(ValueError, 'acquisition loss'):
            m.apply(write(1, .1), 1., [10])
        self.assertEqual(m.sequence, 0)
        self.assertEqual(m.record(0)['handle'], 0)


class AssayAndIntegrationTests(unittest.TestCase):
    def test_later_clock_input_cannot_backdate_retention_or_pre_target_copy(self):
        c = clock(10)
        m = memory(c)
        m.apply(write(1, .1), 1., [10])
        frozen = m.assay_snapshot(1.)
        for index in range(10, 20):
            c.observe(raw(index), (index+1)/10)
        before = bytes(m.storage)
        with self.assertRaises(ValueError):
            m.apply(write(2, .2), 1.5)
        with self.assertRaises(ValueError):
            m.assay_snapshot(1.5)
        with self.assertRaises(ValueError):
            m.availability(0, 1.5)
        self.assertEqual(bytes(m.storage), before)
        self.assertGreater(ref.assay_recognition(frozen, 1, {10: 0.}, 0., 2.)['upper'], 0.)

    def test_frozen_assay_advances_only_time_and_ignores_later_live_recurrence_and_gap(self):
        c = clock(10)
        m = memory(c)
        m.apply(write(1, .1), 1., [10])
        frozen = m.assay_snapshot(1.)
        for index in range(10, 15):
            c.observe(raw(index), (index+1)/10)
        c.gap(200, 2.)
        m.apply(write(2, 1.5), 2.)
        out = ref.assay_recognition(frozen, 1, {10: 0.}, 0., 2.)
        expected = cognition.recognition_probability([-1.9/20], [0.], 0.)
        self.assertEqual(out['lower'], out['upper'])
        self.assertAlmostEqual(out['upper'], expected)
        self.assertNotEqual(out, m.recognition({10: 0.}, 0., 2.))
        self.assertEqual(len(frozen), 32+4*24)
        for epoch, end in ((2, 2.), (1, .9)):
            with self.assertRaises(ValueError):
                ref.assay_recognition(frozen, epoch, {}, 0., end)

    def test_assay_requires_pre_target_issue_and_supported_retained_matches(self):
        m = memory(clock(6))
        m.apply(write(1, .1), .6, [10])
        with self.assertRaises(ValueError):
            m.assay_snapshot(.5)
        snapshot = m.assay_snapshot(.6)
        self.assertEqual(ref.assay_recognition(snapshot, 1, {20: 10.}, 0., 1.)['upper'], 0.)
        direct = m.recognition({10: 0.}, 0., .6)
        self.assertEqual(ref.assay_recognition(snapshot, 1, {10: 0.}, 0., .6), direct)

    def test_real_ledger_matcher_and_scheduler_feed_original_reinforcement_once(self):
        c = clock(6)
        m = memory(c)
        ledger = occurrence.OccurrenceLedger(1)
        ledger.stage(record(1, 0., .1, 1.), paths((10, 100, 1.)), .1)
        first = ledger.advance(.6, [hop(.1, .6)])[0]
        stored = matcher.committed_episode(span(), first, 10, 2)
        m.apply(first, .6, [10])
        for index in range(6, 15):
            c.observe(raw(index), (index+1)/10)
        s = scheduler()
        s.observe(0, descriptor_raw(9), 1.)
        s.submit(0, span(9), 1, 2, 1002, 1.)
        job = s.take(1.)
        result = matcher.match_query(job['query'], [stored], 1.25)
        s.finish(job['ticket'], result, {(10, 2): (0, 10)}, 1.25)
        ledger.stage(record(2, .9, 1., 1.), paths((10, 100, .5), (20, 200, .5)), 1.)
        second = ledger.advance(1.5, [hop(1., 1.5)], s.caches[0].snapshots({0: 10}, 1.5))[0]
        m.apply(second, 1.5, [20])
        self.assertEqual(m.record(0)['strength'], 1.5)
        self.assertEqual(m.record(0)['last_observed_end'], 1.)
        self.assertEqual(m.record(0)['interference_lower'], .5)
        self.assertAlmostEqual(m.availability(0, 1.5)['log_upper'], math.log(1.5)-.5/20-.5/4)
        self.assertFalse(m.apply(second, 1.5))
        self.assertEqual(m.record(1)['strength'], .5)


if __name__ == '__main__':
    unittest.main()
