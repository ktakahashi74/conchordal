"""Independent short-path enumeration and raw-prefix/cache integration checks."""

import copy
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import temporal_descriptor_reference as descriptor
import temporal_matcher_reference as ref
import temporal_occurrence_reference as occurrence
import temporal_section_reference as section


def knots(values, start=0., step=1., generation=3, times=None):
    result = []
    times = times or [start+(i+1)*step for i in range(len(values))]
    for value, end in zip(values, times):
        vector = value if isinstance(value, list) else [value]+[None]*9
        raw = {'epoch': 1, 'generation': generation, 'start': start, 'end': end,
               'time': end, 'raw_support_start': start, 'raw_support_end': end,
               'available_end': end, 'observed': any(v is not None for v in vector),
               'gap': all(v is None for v in vector), 'values': vector}
        result.append(descriptor.DescriptorKnot.from_raw(raw).snapshot())
        start = end
    return result


def fixed_transform(anchor=0, pitch=0., tempo=0.):
    return {'anchor': anchor, 'applied': [pitch, tempo], 'unrounded': [pitch, tempo],
            'frequency_shift_log2': pitch, 'tempo_shift_log2': tempo,
            'out_of_range': False, 'bound_hit': False, 'unidentified': [False, False]}


def query(values, **extra):
    rows = knots(values, start=100.)
    return {'knots': rows, 'epoch': 1, 'generation': 3, 'query_id': 9,
            'occurrence_id': 4, 'support_id': 5, 'query_end': rows[-1]['end'],
            'supporting_audio_end': rows[-1]['end'], 'available_end': rows[-1]['end'],
            'complete': True, 'superseded': False, 'scales': [1.]*10, **extra}


def episode(values, identity=10, **extra):
    rows = knots(values)
    return {'knots': rows, 'episode_id': identity, 'generation': 2, 'epoch': 1,
            'first_observed_end': rows[-1]['end'], 'available_end': rows[-1]['end'],
            'committed': True, 'scales': [1.]*10, **extra}


def raw(index, **extra):
    return {'epoch': 1, 'generation': 3, 'sample_start': index*512,
            'sample_end': (index+1)*512, 'sample_rate': 48000,
            'start': index*512/48000, 'end': (index+1)*512/48000,
            'time': (index+1)*512/48000, 'raw_support_start': index*512/48000,
            'raw_support_end': (index+1)*512/48000, 'available_end': (index+1)*512/48000,
            'observed': True, 'gap': False, 'values': [float(index)]*10, **extra}


class QueryTests(unittest.TestCase):
    def test_pending_prefix_is_searchable_without_changing_live_pairing(self):
        s = descriptor.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10)
        s.push(raw(0), .1)
        before = s.snapshot()
        q = ref.descriptor_query(s, .1, 1, 2, 3)
        self.assertEqual(len(q['knots']), 1)
        self.assertEqual(s.snapshot(), before)
        s.push(raw(1), .1)
        self.assertEqual(s.bank.count, 1)
        self.assertEqual(s.bank.knot(0).snapshot()['moments'][0][1], .5)
        self.assertEqual(q['knots'][0]['values'][0], 0.)
        self.assertEqual(len(q['packed_knots']), 320)
        self.assertEqual(descriptor.DescriptorKnot(q['packed_knots']).snapshot(), q['knots'][0])

    def test_later_shorter_source_window_does_not_hide_earlier_full_audio_support(self):
        s = descriptor.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10)
        s.push(raw(0, raw_support_end=.5, available_end=.5), .5)
        s.push(raw(1), .5)
        q = ref.descriptor_query(s, .5, 1, 2, 3)
        self.assertEqual(q['supporting_audio_end'], .5)
        self.assertLess(q['query_end'], .5)

    def test_full_bank_query_compresses_only_detached_copy(self):
        s = descriptor.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10, capacity=4)
        for i in range(9):
            s.push(raw(i), 1.)
        before = s.snapshot()
        q = ref.descriptor_query(s, 1., 1, 2, 3)
        self.assertEqual(len(q['knots']), 4)
        self.assertEqual(q['merges'], 1)
        self.assertEqual(s.snapshot(), before)
        self.assertEqual(q['copy_payload_bytes'], 1600)

    def test_gap_does_not_refresh_supporting_audio_age(self):
        s = descriptor.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10)
        s.push(raw(0), .1)
        s.gap(10., 10., 10.)
        q = ref.descriptor_query(s, 10., 1, 2, 3)
        self.assertEqual(q['query_end'], 10.)
        self.assertEqual(q['supporting_audio_end'], 512/48000)
        self.assertTrue(q['knots'][-1]['gap'])
        self.assertTrue(s.has_gap)

    def test_future_cut_and_missing_identity_fail_without_mutation(self):
        s = descriptor.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10)
        s.push(raw(0, available_end=.5), .5)
        before = s.snapshot()
        for args in ((.4, 1, 2, 3), (.5, -1, 2, 3)):
            with self.assertRaises(ValueError):
                ref.descriptor_query(s, *args)
        self.assertEqual(s.snapshot(), before)

    def test_sealed_episode_freezes_span_and_preserves_original_age_on_late_delivery(self):
        s = descriptor.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10)
        s.push(raw(0), .1)
        write = {'epoch': 1, 'start': 0., 'support_end': 512/48000,
                 'committed_at': .6, 'delivered_at': 10., 'sequence': 1,
                 'occurrence_id': 2, 'support_id': 3, 'support': {'available_support': .5}}
        old = ref.committed_episode(s, write, 11, 4)
        self.assertTrue(s.bank.frozen)
        self.assertEqual(old['first_observed_end'], 512/48000)
        self.assertEqual(old['available_end'], 10.)
        self.assertEqual(old['knots'][0]['values'], [0.]*10)
        with self.assertRaises(ValueError):
            s.push(raw(1), 10.)
        q = query([0.], knots=knots([[0.]*10], start=2.), query_end=3.,
                  available_end=3., supporting_audio_end=3.)
        self.assertEqual(ref.match_query(q, [old], 3.)['coarse_entries'], [])

    def test_unheard_or_wrong_span_cannot_be_installed_as_episode(self):
        s = descriptor.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10)
        s.push(raw(0), .1)
        write = {'epoch': 1, 'start': 0., 'support_end': 512/48000,
                 'committed_at': .6, 'delivered_at': 1., 'sequence': 1,
                 'occurrence_id': 2, 'support_id': 3, 'support': {'available_support': 0.}}
        with self.assertRaises(ValueError):
            ref.committed_episode(s, write, 11, 4)
        self.assertFalse(s.bank.frozen)
        write.update(support_end=1., support={'available_support': 1.})
        with self.assertRaises(ValueError):
            ref.committed_episode(s, write, 11, 4)


class CoarseTests(unittest.TestCase):
    def test_transposition_and_faster_cue_use_original_intervals(self):
        cue = knots([9., 10., 11.], start=100., step=.5)
        old = knots([8., 9., 10.])
        t = ref.anchor_transform(cue, old, 0)
        self.assertEqual(t['applied'], [1., 1.])
        self.assertEqual((t['pitch_samples'], t['interval_samples']), (3, 2))
        self.assertEqual(t['pitch_estimate_residual_rms'], 0.)
        self.assertEqual(t['tempo_estimate_residual_rms'], 0.)
        self.assertEqual(ref.coarse_cost(cue, old, 0, t, [1.]*10)['cost'], 0.)

    def test_lower_grid_ties_for_positive_and_negative_shifts(self):
        for shift, expected in ((1/32, 0.), (-1/32, -1/16)):
            t = ref.anchor_transform(knots([8.+shift]), knots([8.]), 0)
            self.assertEqual(t['frequency_shift_log2'], expected)

    def test_missing_steps_keep_positions_and_unidentified_masks(self):
        cue, old = knots([8., None, 9.]), knots([8., None, 9.])
        t = ref.anchor_transform(cue, old, 0)
        self.assertEqual(t['pitch_samples'], 2)
        self.assertEqual(t['interval_samples'], 0)
        self.assertEqual(t['applied'], [0., 0.])
        self.assertEqual(t['unidentified'], [False, True])
        self.assertIsNone(t['tempo_shift_log2'])

    def test_nonpitch_coordinates_do_not_identify_pitch(self):
        a = knots([[None, .5]+[None]*8]*3)
        t = ref.anchor_transform(a, a, 0)
        self.assertIsNone(t['frequency_shift_log2'])
        self.assertEqual(t['tempo_shift_log2'], 0.)
        self.assertEqual(len(ref.refine_anchor(a, a, t, [1.]*10)), 1)

    def test_transform_bounds_are_unknown_not_novelty(self):
        for shift in (-3., -2., 2., 3.):
            a, b = knots([8.+shift, 9.+shift]), knots([8., 9.])
            t = ref.anchor_transform(a, b, 0)
            self.assertTrue(t['bound_hit'])
            self.assertIsNone(ref.coarse_cost(a, b, 0, t, [1.]*10)['cost'])
            self.assertEqual(ref.refine_anchor(a, b, t, [1.]*10), [])

    def test_four_refinements_and_bounded_median_comparisons(self):
        cue = knots([9.-i*.1+.007 for i in range(8)], step=2.**(-.007))
        old = knots([8.]*8)
        t = ref.anchor_transform(cue, old, 0)
        self.assertLessEqual(t['median_comparisons'], 49)
        refined = ref.refine_anchor(cue, old, t, [1.]*10)
        self.assertEqual(len(refined), 4)
        self.assertEqual(len({tuple(x['grid_id']) for x in refined}), 4)
        self.assertEqual(refined, sorted(refined, key=lambda x:(x['coarse_cost'],x['grid_id'])))

    def test_coarse_cost_averages_valid_values_and_masks_without_zero_fill(self):
        a = knots([[8., 2.]+[None]*8, [8.]+[None]*9])
        b = knots([[8., 0.]+[None]*8, [7., 0.]+[None]*8])
        cost = ref.coarse_cost(a, b, 0, fixed_transform(), [1.]*10)
        self.assertEqual(cost['valid_coordinates'], 3)
        self.assertEqual(cost['cost'], 5/3)
        a = knots([None, None])
        self.assertIsNone(ref.coarse_cost(a, b, 0, fixed_transform(), [1.]*10)['cost'])


class DtwTests(unittest.TestCase):
    def test_subsequence_excludes_reference_prefix_and_suffix_from_deletions(self):
        a, b = knots([8., 9., 8.]), knots([20., 18., 8., 9., 8., 30.])
        out = ref.subsequence_dtw(a, b, fixed_transform(2), [1.]*10, 100., band_radius=None)
        self.assertEqual(out['cost'], 0.)
        self.assertEqual(out['reference_start'], 2)
        self.assertEqual(out['deleted_steps'], 0)
        self.assertEqual(out['path'], [('pair', 0, 2), ('pair', 1, 3), ('pair', 2, 4)])

    def test_internal_insertions_and_deletions_keep_explicit_penalties(self):
        cases = [([0., 100., 10.], [0., 10.], .2, 1., .2/3, 'inserted_steps'),
                 ([0., 10.], [0., 100., 10.], 1., .3, .3/2, 'deleted_steps')]
        for av, bv, ip, dp, expected, field in cases:
            out = ref.subsequence_dtw(knots(av), knots(bv), fixed_transform(), [1.]*10,
                                      100., None, ip, dp)
            self.assertAlmostEqual(out['cost'], expected)
            self.assertEqual(out[field], 1)

    def test_short_paths_against_independent_exhaustive_enumeration(self):
        rng = random.Random(47013)
        for _ in range(50):
            a = [rng.uniform(-2., 2.) for _ in range(3)]
            b = [rng.uniform(-2., 2.) for _ in range(4)]
            ip, dp = rng.choice((.5, 1., 2.)), rng.choice((.5, 1., 2.))
            def paths(i, j, cost):
                if i == len(a):
                    yield cost
                    return
                yield from paths(i+1, j, cost+ip)
                if j < len(b):
                    yield from paths(i+1, j+1, cost+(a[i]-b[j])**2)
                    yield from paths(i, j+1, cost+dp)
            expected = min(value for start in range(len(b)) for value in paths(0, start, 0.))/len(a)
            out = ref.subsequence_dtw(knots(a), knots(b), fixed_transform(), [1.]*10,
                                      100., None, ip, dp)
            self.assertAlmostEqual(out['total_cost']/len(a), expected, places=13)

    def test_missing_pairs_are_diagnostics_not_automatic_edit_operations(self):
        a, b = knots([8., None, 9.]), knots([8., None, 9.])
        out = ref.subsequence_dtw(a, b, fixed_transform(), [1.]*10, 100., None)
        self.assertEqual(out['cost'], 0.)
        self.assertEqual(out['missing_pair_steps'], 1)
        self.assertEqual(out['observed_cue_steps'], 2)
        self.assertEqual(out['inserted_steps']+out['deleted_steps'], 0)
        empty = ref.subsequence_dtw(knots([None]), knots([None]), fixed_transform(), [1.]*10, 100., None)
        self.assertFalse(empty['supported'])
        self.assertIsNone(empty['cost'])

    def test_equal_bags_with_different_order_have_different_dtw_costs(self):
        a, b = [0., .2, .8, 1.], [0., .8, .2, 1.]
        self.assertEqual(sorted(a), sorted(b))
        scales = [.1]*10
        same = ref.subsequence_dtw(knots(a), knots(a), fixed_transform(), scales, 100., None)
        shuffled = ref.subsequence_dtw(knots(a), knots(b), fixed_transform(), scales, 100., None)
        self.assertEqual(same['cost'], 0.)
        self.assertGreater(shuffled['cost'], 0.)

    def test_band_uses_affine_original_times_at_unequal_knot_density(self):
        a = knots([1., 2., 3.], start=9., times=[10., 11., 12.])
        b = knots([1., 1., 1., 2., 2., 3.], times=[1., 1.1, 1.2, 2., 2.1, 3.])
        out = ref.subsequence_dtw(a, b, fixed_transform(), [1.]*10, 100., 0)
        self.assertEqual(out['band_ranges'], [(0, 1), (3, 4), (5, 6)])
        self.assertTrue(out['band_disconnected'])
        self.assertFalse(out['supported'])

    def test_nearest_time_tie_uses_earlier_knot(self):
        a = knots([1., 2.], times=[1., 1.5])
        b = knots([1., 2.], times=[1., 2.])
        out = ref.subsequence_dtw(a, b, fixed_transform(), [1.]*10, 100., 0)
        self.assertEqual(out['band_ranges'], [(0, 1), (0, 1)])

    def test_band_edge_flag_and_half_double_widths(self):
        a, b = knots([0., 1., 2.]), knots([9., 0., 1., 2., 9.])
        narrow = ref.subsequence_dtw(a, b, fixed_transform(), [1.]*10, 100., 1)
        self.assertTrue(narrow['band_edge_hit'])
        for radius in (8, 16, 32):
            out = ref.subsequence_dtw(a, b, fixed_transform(), [1.]*10, 100., radius)
            self.assertEqual(out['cost'], 0.)

    def test_causal_validation_rejects_future_or_mixed_lineage(self):
        a, b = knots([1., 2.]), knots([1., 2.])
        for changes in ({'available_end': 101.}, {'epoch': 2}, {'generation': 8}):
            bad = copy.deepcopy(a)
            bad[1].update(changes)
            with self.assertRaises(ValueError):
                ref.subsequence_dtw(bad, b, fixed_transform(), [1.]*10, 100.)

    def test_default_dp_capacity_bound_and_payload_are_explicit(self):
        a = knots([0.]*128)
        out = ref.subsequence_dtw(a, a, fixed_transform(), [1.]*10, 200.)
        self.assertLessEqual(out['dp_cells'], 128*33)
        self.assertEqual(out['dp_payload_bytes'], 129*16+128*33+128*4+128*8)
        self.assertLessEqual(out['band_time_comparisons'], 128*8)
        with self.assertRaises(ValueError):
            ref.subsequence_dtw(knots([0.]*129), a, fixed_transform(), [1.]*10, 200.)

    def test_pitch_motion_and_interval_residuals_keep_separate_evidence(self):
        a, b = knots([8., 9., 11.], step=.5), knots([8., 9., 10.])
        out = ref.subsequence_dtw(a, b, fixed_transform(tempo=1.), [1.]*10,
                                  100., None, 10., 10.)
        self.assertEqual(out['relative_pitch_motion_pairs'], 2)
        self.assertAlmostEqual(out['relative_pitch_motion_rms'], math.sqrt(.5))
        self.assertEqual(out['matched_interval_log2_residual_rms'], 0.)


class SearchTests(unittest.TestCase):
    def test_mid_episode_anchor_retrieves_and_supplies_section_classification(self):
        q = query([8., 9., 8.])
        old = episode([20., 20., 20., 20., 8., 9., 8., 10.])
        out = ref.match_query(q, [old], 120.)
        best = out['matches'][0]
        self.assertEqual(best['anchor'], 4)
        self.assertEqual(best['cost'], 0.)
        self.assertEqual(section.correspondence_category(best, [0.]*6, None), 0)

    def test_non_top_candidates_keep_coarse_interference_and_ties_unknown(self):
        q = query([8., 9.])
        old = [episode([8., 9.], identity=i) for i in range(20)]
        out = ref.match_query(q, old[::-1], 120.)
        self.assertEqual(len(out['coarse_entries']), 20)
        self.assertEqual(len(out['matches']), 16)
        self.assertEqual(out['excluded_episode_ids'], [16, 17, 18, 19])
        self.assertTrue(out['cutoff_tie'])
        self.assertEqual(section.correspondence_category(out['matches'][0], [0.]*6, None), 4)
        snap = ref.coarse_commit_snapshot(out, {(19, 2): 1019}, 120.)
        self.assertEqual(snap['entries'], {1019: 0.})
        write = {'support_end': 103., 'support': occurrence.joint_occurrence_support(
                    [{'path_id': 1, 'episode_handle': 10, 'context_handle': None, 'weight': 1.}], .5),
                 'coarse_snapshot': snap}
        self.assertEqual(occurrence.occurrence_interference(write, 1019, 2.)['lower'], .5)

    def test_generation_binding_does_not_restore_stale_episode(self):
        out = ref.match_query(query([8., 9.]), [episode([8., 9.])], 120.)
        snap = ref.coarse_commit_snapshot(out, {(10, 3): 1003}, 120.)
        self.assertEqual(snap['entries'], {})
        self.assertEqual(snap['available_end'], 120.)
        self.assertEqual(snap['support_end'], 102.)

    def test_empty_unknown_and_bound_hits_never_become_novelty(self):
        for old in ([], [episode([None, None])], [episode([18., 19.])]):
            out = ref.match_query(query([8., 9.]), old, 120.)
            self.assertEqual(out['status'], 'unknown_unretrieved')
            self.assertEqual(out['matches'], [])

    def test_wrong_epoch_uncommitted_future_and_nonprior_episode_excluded(self):
        for changes in ({'epoch': 2}, {'committed': False}, {'available_end': 121.},
                        {'first_observed_end': 102.}):
            out = ref.match_query(query([8., 9.]), [episode([8., 9.], **changes)], 120.)
            self.assertEqual(out['coarse_entries'], [])

    def test_stale_query_aliases_changed_scales_and_capacity_rejected(self):
        for q in (query([8.], superseded=True), query([8.], complete=False),
                  query([8.], available_end=121.)):
            with self.assertRaises(ValueError):
                ref.match_query(q, [], 120.)
        with self.assertRaises(ValueError):
            ref.match_query(query([8.]), [episode([8.])]*2, 120.)
        with self.assertRaises(ValueError):
            ref.match_query(query([8.]), [episode([8.], scales=[2.]*10)], 120.)
        with self.assertRaises(ValueError):
            ref.match_query(query([8.]), [episode([8.], i) for i in range(257)], 120.)

    def test_anchor_cap_loss_is_explicit(self):
        out = ref.match_query(query([8., 9.]), [episode([8., 9.]*8)], 120., anchor_limit=1)
        self.assertTrue(out['coarse_entries'][0]['anchor_cap_loss'])
        self.assertFalse(out['matches'][0]['search_covered'])
        self.assertEqual(out['anchor_evaluations'], 1)

    def test_coarse_completion_reaches_sealed_occurrence_without_refresh(self):
        q = query([8., 9.])
        out = ref.match_query(q, [episode([8., 9.])], 102.)
        snapshot = ref.coarse_commit_snapshot(out, {(10, 2): 1010}, 102.)
        hop = {'epoch': 1, 'generation': 3, 'start': 100., 'end': 102., 'observed': True,
               'association_known': True, 'assignment_support': 1., 'articulation': 1,
               'periodic_proposals': [], 'timing_histories': [], 'resolved': True,
               'group_handle': 3, 'bus_energy': 1., 'resolved_energies': {3: 1.}}
        activity = section.section_activity([hop], [], 1, {3}, 100., 102.)
        record = {'epoch': 1, 'occurrence_id': 4, 'support_id': 5, 'generation': 3,
                  'generation_handles': {3}, 'start': 100., 'support_end': 102.,
                  'ordering_known': True, 'ending_descriptor': [0.]*6, 'activity': activity}
        paths = [{'path_id': 1, 'episode_handle': 1010, 'context_handle': 20,
                  'weight': 1., 'correspondence': out['matches'][0]}]
        ledger = occurrence.OccurrenceLedger(1)
        ledger.stage(record, paths, 102.)
        lag = {**hop, 'start': 102., 'end': 102.5}
        ledger.advance(102.5, [lag], [snapshot])
        projected = ledger.section_record(4, 1)
        self.assertEqual(projected['record']['assignment']['cost'], 0.)
        self.assertEqual(section.correspondence_category(projected['record']['assignment'], [0.]*6, None), 0)
        writes = ledger.snapshot()['writes']
        self.assertEqual(writes[0]['coarse_snapshot']['entries'], {1010: 0.})
        self.assertEqual(writes[0]['support_end'], 102.)
        self.assertEqual(writes[0]['committed_at'], 102.5)
        late = ref.coarse_commit_snapshot(ref.match_query(q, [episode([8., 9.])], 103.), {(10, 2): 1010}, 103.)
        other = occurrence.OccurrenceLedger(1)
        other.stage(record, paths, 102.)
        other.advance(102.5, [lag], [late])
        self.assertIsNone(other.snapshot()['writes'][0]['coarse_snapshot'])
        delivered_late = ref.coarse_commit_snapshot(out, {(10, 2): 1010}, 103.)
        delayed = occurrence.OccurrenceLedger(1)
        delayed.stage(record, paths, 102.)
        self.assertIsNone(delayed.advance(103., [lag], [delivered_late])[0]['coarse_snapshot'])
        for receipt in (101., math.inf):
            with self.assertRaises(ValueError):
                ref.coarse_commit_snapshot(out, {}, receipt)


if __name__ == '__main__':
    unittest.main()
