"""Independent support sums, deadline cuts and sealed occurrence invariants."""

import copy
import math
import random
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import temporal_occurrence_reference as ref
import temporal_section_reference as section
import temporal_accent_reference as accent
import temporal_cognition_reference as cognition


def hop(start, end, **extra):
    return {'epoch': 1, 'generation': 4, 'start': start, 'end': end, 'observed': True,
            'association_known': True, 'assignment_support': 1., 'articulation': 1,
            'periodic_proposals': [], 'timing_histories': [], 'resolved': True,
            'group_handle': 4, 'bus_energy': 1., 'resolved_energies': {4: 1.}, **extra}


def paths(*cells):
    return [{'path_id': i, 'episode_handle': e, 'context_handle': c, 'weight': w,
             'correspondence': {'status': 'match', 'supported': True, 'search_nonempty': True,
                                'cost': .1, 'frequency_shift_log2': 0., 'tempo_shift_log2': 0.}}
            for i, (e, c, w) in enumerate(cells or [(10, 100, 1.)])]


def record(identity=1, start=0., end=1., support=.5, generation=4):
    activity = section.section_activity([hop(start, end, assignment_support=support, generation=generation)],
                                        [], 1, {generation}, start, end)
    return {'epoch': 1, 'occurrence_id': identity, 'support_id': 1000+identity,
            'generation': generation, 'generation_handles': {generation},
            'start': start, 'support_end': end, 'ordering_known': True,
            'ending_descriptor': [0.]*6, 'activity': activity}


def event(index=0, width=4800, generation=4):
    raw = [{'epoch': 1, 'generation': generation, 'sample_start': (index+i)*width,
            'sample_end': (index+i+1)*width, 'sample_rate': 48000,
            'observed': True, 'association_known': True, 'association_handle': 19,
            'grid_id': 7, 'energy': 2.**(2*value), 'spectrum': [2.**(2*value)]}
           for i, value in enumerate([0., 0., 1.5, 1.5])]
    return accent.accent_window(raw, [0., 0.], [1., 1.])['accent']


def coarse(identity=1, end=1., available=1., **extra):
    return {'epoch': 1, 'generation': 4, 'occurrence_id': identity, 'support_id': 1000+identity,
            'query_id': 1, 'support_end': end, 'available_end': available,
            'supporting_audio_end': end,
            'completed': True, 'entries': {10: 0., 20: math.log(2)}, **extra}


class OccurrenceSupportTests(unittest.TestCase):
    def test_joint_cells_preserve_correlation_and_acoustic_support_enters_once(self):
        rows = paths((10, 100, .6), (20, 200, .3), (10, None, .1))
        output = ref.joint_occurrence_support(rows+rows[:1], .5)
        self.assertEqual(output['joint'], {(10, 100): .3, (20, 200): .15, (10, None): .05})
        self.assertAlmostEqual(output['episodes'][10], .35)
        self.assertNotIn((10, 200), output['joint'])
        self.assertNotIn((20, 100), output['joint'])
        self.assertEqual(output['unobserved_support'], .5)
        self.assertEqual(output['unassigned_available_support'], 0.)

    def test_unknown_and_pruned_path_mass_are_not_redistributed(self):
        support = ref.joint_occurrence_support(paths((10, 100, .25), (None, None, .25)), .8)
        self.assertEqual(support['episodes'], {10: .2})
        self.assertEqual(support['unknown_episode_support'], .2)
        self.assertEqual(support['unassigned_available_support'], .4)
        self.assertEqual(ref.joint_occurrence_support([], .5)['episodes'], {})
        for rows in (paths((10, 100, .7), (20, 200, .7)), paths((10, 100, math.nan))):
            with self.assertRaises(ValueError):
                ref.joint_occurrence_support(rows, .5)
        rows = paths()
        with self.assertRaises(ValueError):
            ref.joint_occurrence_support(rows+[{**rows[0], 'context_handle': 200}], 1.)

    def test_physical_coverage_unions_duplicates_without_borrowing_future_raw_hops(self):
        known = [hop(1., 1.25), hop(1.125, 1.375), hop(1.375, 1.5), hop(1.5, 2.)]
        out = ref.observed_coverage(known+known, 1, {4}, 1., 1.5, 1.5)
        self.assertEqual(out, {'known_seconds': .5, 'missing_seconds': 0., 'fraction': 1.})
        out = ref.observed_coverage([hop(1., 2.)], 1, {4}, 1., 1.5, 1.5)
        self.assertEqual(out['fraction'], 0.)
        self.assertEqual(ref.observed_coverage(known, 2, {4}, 1., 1.5, 1.5)['fraction'], 0.)
        self.assertEqual(ref.observed_coverage(known, 1, {5}, 1., 1.5, 1.5)['fraction'], 0.)


class CommitmentTests(unittest.TestCase):
    def test_provisional_updates_seal_once_and_replay_cannot_change_totals(self):
        ledger = ref.OccurrenceLedger(1)
        original = record()
        ledger.stage(original, paths(), 1.)
        self.assertEqual(ledger.snapshot()['episode_support_totals'], {})
        self.assertEqual(ledger.advance(1.25, [hop(1., 1.25)]), [])
        ledger.revise(1, paths((20, 200, .8), (10, None, .2)), 1.375, 99)
        self.assertEqual(ledger.snapshot()['episode_support_totals'], {})
        writes = ledger.advance(1.5, [hop(1., 1.25), hop(1.25, 1.5)])
        self.assertEqual(writes[0]['support']['episodes'], {20: .4, 10: .1})
        self.assertEqual(writes[0]['committed_at'], 1.5)
        self.assertFalse(writes[0]['unknown_interference'])
        frozen = ledger.snapshot()
        self.assertFalse(ledger.stage(copy.deepcopy(original), paths(), 1.5))
        self.assertEqual(ledger.advance(1.5, []), [])
        self.assertEqual(ledger.snapshot(), frozen)
        writes[0]['ending_descriptor'][0] = 99
        original['ending_descriptor'][0] = 99
        self.assertEqual(ledger.snapshot(), frozen)

    def test_lag_half_double_and_missing_interval_do_not_change_available_support(self):
        for lag in (.25, .5, 1.):
            ledger = ref.OccurrenceLedger(1, lag_sec=lag)
            ledger.stage(record(), paths((10, 100, .75)), 1.)
            self.assertEqual(ledger.advance(1.+lag/2, []), [])
            write = ledger.advance(1.+lag, [hop(1., 1.+lag/2)])[0]
            self.assertEqual(write['support']['episodes'], {10: .375})
            self.assertEqual(write['lag_coverage']['fraction'], .5)
            self.assertTrue(write['unknown_interference'])
            self.assertEqual(write['lag_coverage']['missing_seconds'], lag/2)

    def test_late_processing_keeps_deadline_and_original_retention_time(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        write = ledger.advance(10., [hop(1., 1.5), hop(1.5, 10.)])[0]
        self.assertEqual((write['support_end'], write['committed_at'], write['delivered_at']), (1., 1.5, 10.))
        value = cognition.episode_log_availability(write['support']['episodes'][10],
                                                   10.-write['support_end'], 0., 2., 1.)
        self.assertAlmostEqual(value, math.log(.5)-4.5)
        with self.assertRaises(ValueError):
            ledger.revise(1, paths(), 9., 10)

    def test_late_provisional_evidence_requires_sealing_before_metadata_revision(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.revise(1, paths((20, 200, 1.)), 1.75, 99)
        self.assertEqual(ledger.snapshot(), before)
        write = ledger.advance(1.75, [hop(1., 1.5)])[0]
        self.assertEqual(write['support']['episodes'], {10: .5})
        ledger.revise(1, paths((20, 200, 1.)), 1.75, 99)
        self.assertEqual(ledger.snapshot()['episode_support_totals'], {10: .5})

    def test_original_endpoint_start_id_order_and_batch_validation_are_atomic(self):
        ledger = ref.OccurrenceLedger(1)
        for item in (record(3, .25, 1.25), record(2, .125, 1.25), record(1, .125, 1.25)):
            ledger.stage(item, paths(), 1.25)
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.advance(1.75, [hop(1.25, 1.75)], [coarse(3, 1.25, 1.25, entries={10: math.nan})])
        self.assertEqual(ledger.snapshot(), before)
        writes = ledger.advance(1.75, [hop(1.25, 1.75)])
        self.assertEqual([w['occurrence_id'] for w in writes], [1, 2, 3])
        self.assertEqual([w['sequence'] for w in writes], [1, 2, 3])

    def test_alias_support_conflicts_unheard_epoch_and_overflow_never_create_occurrences(self):
        ledger = ref.OccurrenceLedger(1, pending_limit=1)
        self.assertFalse(ledger.stage(record(support=0.), paths(), 1.))
        self.assertFalse(ledger.stage({**record(), 'epoch': 2}, paths(), 1.))
        ledger.stage(record(), paths(), 1.)
        for other in ({**record(2), 'support_id': 1001}, {**record(), 'ending_descriptor': [1.]*6}):
            with self.assertRaises(ValueError):
                ledger.stage(other, paths(), 1.)
        self.assertFalse(ledger.stage(record(2), paths(), 1.))
        self.assertEqual(ledger.snapshot()['computational_losses'][-1]['reason'], 'pending_capacity')
        ledger.advance(1.5, [])
        self.assertFalse(ledger.stage(record(2), paths(), 1.5))
        self.assertEqual(ledger.snapshot()['computational_losses'][-1]['reason'], 'ending_after_seal_cut')
        self.assertEqual(ledger.snapshot()['committed_count'], 1)
        fresh = ref.OccurrenceLedger(2)
        self.assertEqual(fresh.snapshot()['episode_support_totals'], {})

    def test_inputs_cannot_move_back_before_already_observed_pending_evidence(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        before = ledger.snapshot()
        for operation in (lambda: ledger.advance(.5, []),
                          lambda: ledger.revise(1, paths(), .75, 99),
                          lambda: ledger.deliver_accent(event(), {1: 1.}, .75)):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(ledger.snapshot(), before)


class RevisionAndContextTests(unittest.TestCase):
    def test_revision_compares_original_joint_cells_and_flags_each_occurrence_once(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(support=1.), paths(), 1.)
        ledger.advance(1.5, [])
        original = ledger.snapshot()['writes']
        for weight, expected in ((.75, False), (.749, True), (1., True)):
            result = ledger.revise(1, paths((10, 100, weight), (10, 200, 1-weight)), 2., 99)
            self.assertEqual(result['revision_flag'], expected)
            self.assertEqual(ledger.snapshot()['writes'], original)
        self.assertEqual(ledger.snapshot()['revision_count'], 1)
        self.assertEqual(ledger.snapshot()['revision_rate'], 1.)
        self.assertEqual(ledger.snapshot()['context_support_totals'], {(10, 100): 1.})

    def test_revision_half_double_thresholds_preserve_strict_comparison(self):
        for threshold in (.125, .25, .5):
            ledger = ref.OccurrenceLedger(1, revision_threshold=threshold)
            ledger.stage(record(support=1.), paths(), 1.)
            ledger.advance(1.5, [])
            self.assertFalse(ledger.revise(1, paths((10, 100, 1-threshold)), 2., 8)['revision_flag'])
            self.assertTrue(ledger.revise(1, paths((10, 100, 1-threshold-.01)), 2., 9)['revision_flag'])

    def test_membership_totals_are_uncapped_and_evicted_context_is_never_renormalized(self):
        ledger = ref.OccurrenceLedger(1)
        for i in range(4):
            ledger.stage(record(i+1, i*2., i*2.+1., support=1.),
                         paths((10, 100, .5), (10, 200, .25), (10, None, .25)), i*2.+1.)
            ledger.advance(i*2.+1.5, [])
        frozen = ledger.snapshot()['writes']
        self.assertEqual(ledger.snapshot()['episode_support_totals'], {10: 4.})
        self.assertEqual(ledger.snapshot()['context_edges'], {(10, 100): .5, (10, 200): .25})
        ledger.evict_context(100)
        self.assertEqual(ledger.snapshot()['context_edges'], {(10, 200): .25})
        self.assertEqual(ledger.snapshot()['writes'], frozen)
        ledger.stage(record(5, 8., 9., support=1.), paths(), 9.)
        ledger.advance(9.5, [])
        self.assertEqual(ledger.snapshot()['episode_support_totals'], {10: 5.})
        self.assertEqual(ledger.snapshot()['context_edges'], {(10, 200): .2})

    def test_all_prefixes_match_independent_joint_and_context_sums_over_100_streams(self):
        rng = random.Random(47011)
        for run in range(100):
            ledger = ref.OccurrenceLedger(1)
            expected_episodes, expected_contexts = {}, {}
            for i in range(12):
                available = (.125, .5, 1.)[(run+i) % 3]
                cells = [(rng.choice([10, 20, 30, None]), rng.choice([100, 200, None]), rng.random())
                         for _ in range(4)]
                denominator = sum(c[2] for c in cells) / .875
                cells = [(e, c, weight/denominator) for e, c, weight in cells]
                rows = paths(*cells)
                ledger.stage(record(i, 2.*i, 2.*i+1., support=available), rows+rows[:1], 2.*i+1.)
                ledger.advance(2.*i+1.5, [])
                for episode, context, weight in cells:
                    if episode is not None:
                        expected_episodes.setdefault(episode, []).append(available*weight)
                        if context is not None:
                            expected_contexts.setdefault((episode, context), []).append(available*weight)
                snapshot = ledger.snapshot()
                for episode, terms in expected_episodes.items():
                    self.assertAlmostEqual(snapshot['episode_support_totals'][episode], math.fsum(terms), delta=1e-12)
                for key, terms in expected_contexts.items():
                    self.assertAlmostEqual(snapshot['context_support_totals'][key], math.fsum(terms), delta=1e-12)
                    self.assertAlmostEqual(snapshot['context_edges'][key],
                                           math.fsum(terms)/math.fsum(expected_episodes[key[0]]), delta=1e-12)


class CoarseSnapshotTests(unittest.TestCase):
    def test_later_full_audio_support_cannot_supply_an_earlier_span(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        write = ledger.advance(1.5, [], [coarse(supporting_audio_end=1.125, available=1.25)])[0]
        self.assertIsNone(write['coarse_snapshot'])
        self.assertEqual(ref.occurrence_interference(write, 20, .5)['upper'], .5)

    def test_trailing_gap_or_absent_acoustic_support_cannot_refresh_snapshot(self):
        for audio_end in (.5, None):
            ledger = ref.OccurrenceLedger(1)
            ledger.stage(record(), paths(), 1.)
            write = ledger.advance(1.5, [], [coarse(supporting_audio_end=audio_end)])[0]
            self.assertIsNone(write['coarse_snapshot'])

    def test_same_occurrence_and_query_ids_do_not_bypass_generation_lineage(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        write = ledger.advance(1.5, [], [coarse(generation=5)])[0]
        self.assertIsNone(write['coarse_snapshot'])
        with self.assertRaises(ValueError):
            other = ref.OccurrenceLedger(1)
            other.stage(record(), paths(), 1.)
            other.advance(1.5, [], [coarse(supporting_audio_end=math.inf)])

    def test_exact_occurrence_support_completion_and_original_cut_select_snapshot(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        valid = coarse(end=.9375, available=1.25)
        invalid = [coarse(2), coarse(support_id=999), coarse(end=1.01, available=1.1),
                   coarse(available=1.75), coarse(completed=False), coarse(superseded=True), coarse(end=.8)]
        write = ledger.advance(2., [hop(1., 1.5)], invalid+[valid])[0]
        self.assertEqual(write['coarse_snapshot'], valid)
        valid['entries'][10] = 99
        self.assertEqual(ledger.snapshot()['writes'][0]['coarse_snapshot']['entries'][10], 0.)

    def test_latest_endpoint_then_query_id_breaks_ties_without_waiting_for_future(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        candidates = [coarse(end=.9375, query_id=99), coarse(query_id=2), coarse(query_id=3)]
        self.assertEqual(ledger.advance(1.5, [], candidates)[0]['coarse_snapshot']['query_id'], 3)

    def test_competing_support_and_unknown_similarity_have_separate_bounds(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(support=1.), paths((10, 100, .25), (20, 200, .5), (None, None, .125)), 1.)
        write = ledger.advance(1.5, [], [coarse(entries={10: math.log(2)})])[0]
        result = ref.occurrence_interference(write, 10, .5)
        self.assertEqual(result['recurrence_support'], .25)
        self.assertAlmostEqual(result['lower'], .25)
        self.assertAlmostEqual(result['upper'], .375)
        result = ref.occurrence_interference(write, 20, .5)
        self.assertEqual(result['lower'], 0.)
        self.assertEqual(result['upper'], .5)
        self.assertEqual(ref.occurrence_interference(write, 30, 1.),
                         {'eligible': False, 'lower': 0., 'upper': 0., 'recurrence_support': 0.})


class PendingAccentAndSectionTests(unittest.TestCase):
    def test_projection_resolves_sparse_full_width_handles_after_alias_deduplication(self):
        rows = paths(*[(10+i, 100+i, .125) for i in range(5)])
        handles = [0, 3, 101, 2**53+7, 2**64-1]
        for i, row in enumerate(rows):
            row['path_id'] = handles[i]
            row['correspondence']['cost'] = i/8
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), list(reversed(rows))+rows[1:3], 1.)
        write = ledger.advance(1.5, [hop(1., 1.5)])[0]
        self.assertEqual([r['path_id'] for r in write['support']['paths']], handles)
        for row in rows:
            actual = ledger.section_record(1, row['path_id'])
            self.assertEqual(actual['record']['assignment'], row['correspondence'])
            self.assertEqual(actual['record']['membership'], row['weight'])
            self.assertEqual(actual['activity'], write['activity'])

    def test_projection_keeps_final_pending_revision_and_isolates_all_external_views(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(), paths(), 1.)
        revised = paths((20, 200, .25), (30, 300, .5))
        revised[0]['path_id'], revised[1]['path_id'] = 90, 7
        ledger.revise(1, revised, 1.25, 99)
        revised[0]['path_id'] = 5
        writes = ledger.advance(1.5, [hop(1., 1.5)])
        original = ledger.section_record(1, 90)
        ledger.revise(1, paths(), 1.6, 100)
        external = ledger.snapshot()
        for view in (writes, external['writes']):
            view[0]['support']['paths'].clear()
            view[0]['activity']['numerators'][0] = 1e6
        mutated = ledger.section_record(1, 90)
        mutated['record']['assignment']['cost'] = 1e6
        mutated['record']['ending_descriptor'][0] = 1e6
        mutated['activity']['numerators'][0] = 1e6
        self.assertEqual(ledger.section_record(1, 90), original)
        self.assertEqual(ledger.section_record(1, 7)['record']['membership'], .5)
        with self.assertRaises(ValueError):
            ledger.section_record(1, 0)

    def test_projection_rejects_absent_paths_and_missing_correspondence_without_effects(self):
        for supplied in ([], paths((10, 100, .25), (20, 200, .5))):
            if supplied:
                supplied[0]['path_id'], supplied[1]['path_id'] = 7, 90
                del supplied[1]['correspondence']
            ledger = ref.OccurrenceLedger(1)
            ledger.stage(record(), supplied, 1.)
            ledger.advance(1.5, [])
            frozen = ledger.snapshot()
            for key in (-1, 0, 8, 90, 91, 2**64, None, '7', [], math.nan):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    ledger.section_record(1, key)
                self.assertEqual(ledger.snapshot(), frozen)

    def test_shared_late_accent_can_reach_later_staged_span_but_not_original_ending_descriptor(self):
        ledger = ref.OccurrenceLedger(1)
        raw = [hop(0., .2, span_shares={1: 1.}), hop(.2, .3, span_shares={1: .25, 2: .75}),
               hop(.3, .6, span_shares={2: 1.})]
        spans = {1: {'start': 0., 'support_end': .3}, 2: {'start': .2, 'support_end': .6}}
        owned = section.partition_span_activity(raw, [], spans, 1, {4}, .6)
        first, second = record(1, 0., .3), record(2, .2, .6)
        first['activity'], second['activity'] = owned[1], owned[2]
        ledger.stage(first, paths((10, 100, .5)), .3)
        signal = event()
        self.assertTrue(ledger.deliver_accent(signal, {1: .25, 2: .75}, .4))
        ledger.stage(second, paths((20, 200, .25)), .6)
        self.assertFalse(ledger.deliver_accent(copy.deepcopy(signal), {1: .25, 2: .75}, .6))
        writes = ledger.advance(1.1, [hop(.3, .8), hop(.8, 1.1)])
        self.assertEqual([w['activity']['numerators'][4] for w in writes], [.125, .375])
        self.assertEqual([w['ending_descriptor'] for w in writes], [[0.]*6]*2)
        history = section.SectionHistory(1, 4, 100, 0.)
        delta = section.section_activity([hop(0., 1.1)], [signal], 1, {4}, 0., 1.1)
        history.observe(delta, 1, 4)
        previous = None
        for identity in (1, 2):
            projected = ledger.section_record(identity, 0, previous, raw)
            history.commit(**projected)
            previous = projected['record']
        snapshot = history.snapshot()
        self.assertAlmostEqual(snapshot['recent']['activity']['numerators'][4], .15625)
        self.assertAlmostEqual(snapshot['recent']['activity']['assignment_seconds'], .20625)
        self.assertAlmostEqual(snapshot['recent']['values'][34], .15625/.20625)
        self.assertAlmostEqual(snapshot['cumulative']['values'][34], .5/1.1)
        vector = cognition.section_head_covariates(snapshot['cumulative']['values'], snapshot['recent']['values'],
                                                    0., 0., 1.1, [(0., 1.1)], [], None, [0.]*82, [1.]*82)
        self.assertEqual(len(vector['raw']), 82)

    def test_receipt_after_sealing_is_reported_without_reopening_or_reinforcing(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(end=.3), paths(), .3)
        ledger.advance(.8, [])
        frozen = ledger.snapshot()['writes']
        ledger.deliver_accent(event(), {1: 1.}, .9)
        self.assertEqual(ledger.snapshot()['writes'], frozen)
        self.assertEqual(ledger.snapshot()['computational_losses'][-1]['reason'], 'accent_after_seal')
        self.assertEqual(ledger.snapshot()['writes'][0]['activity']['numerators'][4], 0.)
        self.assertEqual(ledger.snapshot()['episode_support_totals'], {10: .5})

    def test_accent_owner_and_lineage_failures_are_atomic_and_cannot_donate_credit(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(1, 0., .3), paths(), .3)
        ledger.stage(record(2, 0., .3, generation=5), paths(), .3)
        before = ledger.snapshot()
        for owners in ({1: .75, 2: .75}, {1: .25, 2: .75}):
            with self.assertRaises(ValueError):
                ledger.deliver_accent(event(), owners, .4)
            self.assertEqual(ledger.snapshot(), before)
        ledger.deliver_accent(event(), {1: .25}, .4)
        with self.assertRaises(ValueError):
            ledger.deliver_accent(event(), {1: .5}, .4)
        writes = ledger.advance(.8, [])
        self.assertEqual([w['activity']['numerators'][4] for w in writes], [.125, 0.])

    def test_adjacency_uses_original_known_union_and_never_commit_delay_or_assignment_magnitude(self):
        ledger = ref.OccurrenceLedger(1)
        ledger.stage(record(1, 0., 1., support=.1), paths(), 1.)
        ledger.advance(1.5, [])
        ledger.stage(record(2, 2., 3., support=.1), paths(), 3.)
        ledger.advance(3.5, [])
        first = ledger.section_record(1, 0)['record']
        known = [hop(1., 1.5, assignment_support=.01), hop(1.5, 2., assignment_support=.01)]
        full = ledger.section_record(2, 0, first, known+known)
        self.assertEqual(full['adjacency_coverage'], 1.)
        half = ledger.section_record(2, 0, first, known[:1])
        self.assertEqual(half['adjacency_coverage'], .5)
        future = ledger.section_record(2, 0, first, [hop(1., 4.)])
        self.assertEqual(future['adjacency_coverage'], 0.)


if __name__ == '__main__':
    unittest.main()
