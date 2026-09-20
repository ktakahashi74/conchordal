"""Hand-derived, perturbation and uncapped checks for accent evidence delivery."""

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import temporal_accent_reference as ref
import temporal_section_reference as section


def raw_hop(index, log_rms=0., width=64, rate=48000, **extra):
    energy = 2. ** (2 * log_rms)
    return {"epoch": 1, "generation": 4, "sample_start": index * width,
            "sample_end": (index+1) * width, "sample_rate": rate,
            "observed": True, "association_known": True, "association_handle": 19,
            "grid_id": 7, "energy": energy, "spectrum": [energy], **extra}


def event(index=0, level=1.5, width=64, **extra):
    hops = [raw_hop(index+i, value, width=width, **extra) for i, value in enumerate([0, 0, level, level])]
    return ref.accent_window(hops, [0, 0], [1, 1])["accent"]


def activity_hop(index, **extra):
    return {"epoch": 1, "generation": 4, "start": index/10, "end": (index+1)/10,
            "observed": True, "association_known": True, "assignment_support": 1.,
            "articulation": 1, "periodic_proposals": [], "timing_histories": [],
            "resolved": True, "group_handle": 7, "bus_energy": 1.,
            "resolved_energies": {7: 1.}, **extra}


class AssignedEnergyTests(unittest.TestCase):
    def test_fractional_ridge_group_assignments_conserve_actual_bus_energy(self):
        output = ref.assigned_group_energy(8, [1, 3], [[.75, .25], [.25, .75]],
                                           [[.5, .25, .25], [0, .5, .5]], 2)
        for actual, expected in zip(output, [[.75, .75], [.625, 2.625], [.625, 2.625]]):
            self.assertEqual(actual['spectrum'], expected)
        self.assertEqual(math.fsum(o['energy'] for o in output), 8)
        self.assertEqual(output, ref.assigned_group_energy(8, [100, 300], [[.75, .25], [.25, .75]],
                                                          [[.5, .25, .25], [0, .5, .5]], 2))

    def test_positive_energy_without_spectral_mass_is_residual_and_flux_is_masked(self):
        output = ref.assigned_group_energy(8, [0], [[1]], [[.5, .5]], 1)
        self.assertEqual(output, [{'energy': 0., 'spectrum': None}, {'energy': 8, 'spectrum': None}])
        silent = ref.assigned_group_energy(0, [0], [[1]], [[.5, .5]], 1)
        self.assertEqual(silent, [{'energy': 0., 'spectrum': [0.]}, {'energy': 0., 'spectrum': [0.]}])
        for bad in ([.4, .5], [.5, -.5], [math.nan, 1.]):
            with self.assertRaises(ValueError):
                ref.assigned_group_energy(8, [1], [[1]], [bad], 1)


class AccentWindowTests(unittest.TestCase):
    def test_four_hops_keep_original_uncertain_event_and_later_evidence_separate(self):
        hops = [raw_hop(i, value) for i, value in enumerate([0, 0, 1.5, 1.5])]
        output = ref.accent_window(hops, [0, 0], [1, 1])
        self.assertEqual(output['saliences'], [0., 1.5, 0.])
        self.assertEqual(output['acquisition_coverage'], 1.)
        self.assertEqual(output['detector_coverage'], 1.)
        accent = output['accent']
        self.assertEqual(accent['weight'], .5)
        self.assertEqual(accent['event_interval'], [128/48000, 192/48000])
        self.assertEqual(accent['raw_support_end'], 256/48000)
        self.assertFalse(ref.accent_at_cut(accent, accent['time']))
        self.assertFalse(ref.accent_at_cut(accent, 255/48000))
        self.assertTrue(ref.accent_at_cut(accent, 256/48000))

    def test_each_raw_hop_and_both_components_are_required_without_partial_weight(self):
        for index in range(4):
            for missing in ('observed', 'association_known', 'spectrum'):
                hops = [raw_hop(i, value) for i, value in enumerate([0, 0, 1.5, 1.5])]
                hops[index][missing] = None if missing == 'spectrum' else False
                out = ref.accent_window(hops, [0, 0], [1, 1])
                self.assertEqual(out['status'], 'unsupported')
                self.assertEqual(out['detector_coverage'], 0.)
                self.assertEqual(out['acquisition_coverage'], .75 if missing == 'observed' else 1.)

    def test_gaps_grid_changes_and_association_changes_cannot_invent_peaks(self):
        for field in ('generation', 'epoch', 'association_handle', 'grid_id'):
            hops = [raw_hop(i, value) for i, value in enumerate([0, 0, 1.5, 1.5])]
            hops[1][field] += 1
            self.assertEqual(ref.accent_window(hops, [0, 0], [1, 1])['status'], 'unsupported')
        hops = [raw_hop(i, value) for i, value in zip([0, 1, 3, 4], [0, 0, 1.5, 1.5])]
        self.assertEqual(ref.accent_window(hops, [0, 0], [1, 1])['acquisition_coverage'], .8)
        hops[2]['sample_start'] = 0
        with self.assertRaises(ValueError):
            ref.accent_window(hops, [0, 0], [1, 1])

    def test_partial_acquisition_uses_sample_union_but_never_admits_a_fractional_peak(self):
        hops = [raw_hop(i, value) for i, value in enumerate([0, 0, 1.5, 1.5])]
        hops[0].update(observed=False, known_sample_intervals=[(0, 16), (8, 32)])
        output = ref.accent_window(hops, [0, 0], [1, 1])
        self.assertEqual(output['acquisition_coverage'], 224/256)
        self.assertEqual(output['detector_coverage'], 0.)
        self.assertIsNone(output['accent'])
        hops[0]['observed'] = True
        self.assertEqual(ref.accent_window(hops, [0, 0], [1, 1])['status'], 'unsupported')
        hops[0]['known_sample_intervals'].append((32, 64))
        self.assertEqual(ref.accent_window(hops, [0, 0], [1, 1])['accent']['weight'], .5)

    def test_plateau_admits_only_first_peak_and_decline_is_not_an_accent(self):
        stream = ref.AccentStream(1, 4, [0, 0], [1, 1])
        results = [stream.push(raw_hop(i, value)) for i, value in enumerate([0, 0, 2, 4, 4, 0, 0])]
        admitted = [r['accent'] for r in results if r is not None and r['accent'] is not None]
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]['time'], 192/48000)
        self.assertEqual(admitted[0]['weight'], 1.)
        silent = [raw_hop(i, energy=0., spectrum=[0.]) for i in range(4)]
        self.assertEqual(ref.accent_window(silent, [0, 0], [1, 1])['saliences'], [0.]*3)

    def test_spectral_only_flux_qualifies_and_normalization_is_separate(self):
        hops = [raw_hop(i, energy=1., spectrum=s) for i, s in enumerate(
            [[2**-24, 1-2**-24], [2**-24, 1-2**-24], [.5, .5], [.5, .5]])]
        out = ref.accent_window(hops, [0, 0], [1, 1])
        self.assertEqual(out['raw_components'][1][0], 0.)
        self.assertGreater(out['accent']['weight'], 0.)
        normalized = ref.accent_window(hops, [1., 1.], [2., 4.])
        self.assertAlmostEqual(normalized['saliences'][1], (out['raw_components'][1][1]-1)/8)
        self.assertEqual(normalized['status'], 'below_threshold')

    def test_strict_threshold_and_half_double_registered_controls(self):
        self.assertIsNone(event(level=1.))
        for threshold, expected in ((.5, 1.), (1., .5), (2., None)):
            for floor in (.5e-6, 1e-6, 2e-6):
                out = ref.accent_window([raw_hop(i, v) for i, v in enumerate([0, 0, 1.5, 1.5])],
                                       [0, 0], [1, 1], threshold, floor)
                self.assertEqual(out['accent']['weight'] if out['accent'] else None, expected)

    def test_stream_is_bounded_repeated_reads_do_not_readmit_and_input_mutation_is_isolated(self):
        stream = ref.AccentStream(1, 4, [0, 0], [1, 1])
        first = raw_hop(0)
        stream.push(first)
        first['energy'] = 123
        with self.assertRaises(ValueError):
            stream.push(first)
        events = []
        for i in range(1, 80):
            hop = raw_hop(i, 1.5 if i % 4 >= 2 else 0.)
            result = stream.push(hop)
            if result and result['accent']:
                events.append(result['accent'])
            self.assertIsNone(stream.push(copy.deepcopy(hop)))
            self.assertLessEqual(len(stream.hops), 3)
        self.assertEqual(len(events), 20)
        self.assertIsNone(stream.push(raw_hop(80, generation=5)))
        with self.assertRaises(ValueError):
            stream.push(raw_hop(0))


class AccentDeliveryTests(unittest.TestCase):
    def test_same_id_with_changed_right_support_cannot_receive_new_credit(self):
        ledger = ref.AccentLedger(1, 4)
        original = event()
        ledger.deliver(original, original['available_end'])
        before = copy.deepcopy(ledger.__dict__)
        changed = copy.deepcopy(original)
        changed['raw_support_intervals'][-1] = (192, 320)
        changed['raw_support_end'] = changed['available_end'] = 320/48000
        with self.assertRaises(ValueError):
            ledger.deliver(changed, changed['available_end'])
        self.assertEqual(ledger.__dict__, before)

    def test_out_of_bank_tied_id_cannot_return_with_larger_evidence_end(self):
        ledger = ref.AccentLedger(1, 4, capacity=1)
        first = event()
        second = copy.deepcopy(first)
        second.update(id=(1, 4, 129, 192), event_interval=[129/48000, 192/48000],
                      raw_support_intervals=[(0, 64), (64, 129), (129, 192), (192, 256)])
        ledger.deliver(first, first['available_end'])
        ledger.deliver(second, second['available_end'])
        before = copy.deepcopy(ledger.__dict__)
        changed = copy.deepcopy(first)
        changed['raw_support_intervals'][-1] = (192, 320)
        changed['raw_support_end'] = changed['available_end'] = 320/48000
        with self.assertRaises(ValueError):
            ledger.deliver(changed, changed['available_end'])
        self.assertEqual(ledger.__dict__, before)

    def test_full_source_window_and_availability_can_exceed_canonical_hops(self):
        ledger = ref.AccentLedger(1, 4)
        accent = event(4)
        accent.update(raw_support_start=0., raw_support_end=600/48000, available_end=640/48000)
        self.assertIsNone(ledger.deliver(accent, 600/48000))
        received = ledger.deliver(accent, 700/48000)
        self.assertEqual(received['accent'], accent)
        self.assertEqual(received['delivered_at'], 700/48000)
        self.assertEqual(ledger.snapshot(0.)['accents'], [accent])

    def test_late_admission_preserves_original_time_and_once_only_credit(self):
        ledger = ref.AccentLedger(1, 4)
        accent = event()
        self.assertIsNone(ledger.deliver(accent, accent['time']))
        delivery = ledger.deliver(accent, 1.)
        self.assertEqual(delivery['accent'], accent)
        self.assertEqual(delivery['delivered_at'], 1.)
        frozen = ledger.snapshot(0.)
        self.assertIsNone(ledger.deliver(copy.deepcopy(accent), 1.))
        self.assertEqual(ledger.snapshot(0.), frozen)
        delivery['accent']['weight'] = .99
        accent['weight'] = .75
        self.assertEqual(ledger.snapshot(0.), frozen)
        with self.assertRaises(ValueError):
            ledger.deliver(accent, 1.)
        self.assertEqual(ledger.snapshot(0.), frozen)

    def test_capacity_eviction_is_chronological_and_does_not_erase_cumulative_credit(self):
        for capacity in (64, 128, 256):
            ledger = ref.AccentLedger(1, 4, capacity=capacity)
            stream = ref.AccentStream(1, 4, [0, 0], [1, 1])
            uncapped = []
            for i in range((capacity+12)*4):
                level = 1.25 + (i//4 % 3)*.25
                result = stream.push(raw_hop(i, level if i % 4 >= 2 else 0.))
                if result and result['accent']:
                    accent = result['accent']
                    uncapped.append(accent)
                    ledger.deliver(accent, accent['available_end'])
            self.assertEqual(ledger.admission_count, len(uncapped))
            self.assertEqual(ledger.admission_weight, math.fsum(a['weight'] for a in uncapped))
            self.assertEqual(ledger.bank, uncapped[-capacity:])
            self.assertEqual(ledger.capacity_evicted_through, uncapped[-capacity-1]['time'])
            self.assertFalse(ledger.snapshot(0.)['capacity_valid'])
            start = ledger.bank[0]['time']
            snapshot = ledger.snapshot(start)
            self.assertTrue(snapshot['capacity_valid'])
            self.assertEqual(snapshot['weight'], math.fsum(a['weight'] for a in uncapped if a['time'] >= start))
            before = copy.deepcopy(ledger.__dict__)
            with self.assertRaises(ValueError):
                ledger.deliver(uncapped[0], ledger.observed_end)
            self.assertEqual(ledger.__dict__, before)

    def test_time_expiry_does_not_replace_cap_loss_and_late_expired_delivery_is_still_counted(self):
        ledger = ref.AccentLedger(1, 4, capacity=2, window_sec=1.)
        for i in range(3):
            accent = event(i*4)
            ledger.deliver(accent, accent['available_end'])
        cap = ledger.capacity_evicted_through
        ledger.advance(2.)
        self.assertEqual(ledger.bank, [])
        self.assertEqual(ledger.capacity_evicted_through, cap)
        ledger.deliver(event(12), 2.)
        self.assertEqual(ledger.bank, [])
        self.assertEqual(ledger.admission_count, 4)
        self.assertEqual(ledger.admission_weight, 2.)
        self.assertTrue(ledger.snapshot(1.)['capacity_valid'])
        self.assertEqual(ledger.snapshot(1.)['weight'], 0.)

    def test_equal_timestamps_evict_lower_stable_id_before_weight(self):
        ledger = ref.AccentLedger(1, 4, capacity=2)
        first = event()
        second = copy.deepcopy(first)
        second.update(id=(1, 4, 129, 192), event_interval=[129/48000, 192/48000], weight=.25,
                      raw_support_intervals=[(0, 64), (64, 129), (129, 192), (192, 256)])
        third = event(4)
        for accent in (first, second, third):
            ledger.deliver(accent, accent['available_end'])
        self.assertEqual([a['id'] for a in ledger.bank], [second['id'], third['id']])
        self.assertEqual(ledger.admission_weight, 1.25)

    def test_all_density_windows_match_uncapped_counts_or_explicit_cap_loss(self):
        ledger = ref.AccentLedger(1, 4)
        accents = [event(i*4, width=512) for i in range(200)]
        for accent in accents:
            ledger.deliver(accent, accent['available_end'])
        for window in (.125, .25, .5, 1., 2., 4., 8., 16.):
            start = max(0., ledger.observed_end - window)
            snapshot = ledger.snapshot(start)
            lost = any(a['time'] >= start for a in accents[:-128])
            self.assertEqual(snapshot['capacity_valid'], not lost)
            if lost:
                self.assertIsNone(snapshot['weight'])
            else:
                self.assertEqual(snapshot['weight'], math.fsum(a['weight'] for a in accents if a['time'] >= start))

    def test_epoch_generation_and_incomplete_evidence_are_never_delivered(self):
        ledger = ref.AccentLedger(1, 4)
        accent = event()
        for key in ('epoch', 'generation'):
            self.assertIsNone(ledger.deliver({**accent, key: 9}, 1.))
        for change in ({'raw_support_end': accent['time']}, {'event_interval': [0., accent['time']]},
                       {'raw_support_intervals': accent['raw_support_intervals'][1:]}, {'weight': 0.}):
            with self.assertRaises(ValueError):
                ledger.deliver({**accent, **change}, 1.)
        self.assertEqual(ledger.admission_count, 0)
        self.assertEqual(ref.AccentLedger(2, 4).snapshot(0.)['cumulative_admission_count'], 0)

    def test_future_right_context_changes_peak_but_cannot_change_original_descriptor_or_counts(self):
        hops = [raw_hop(i, v, width=4800) for i, v in enumerate([0, 0, 1.5, 1.5])]
        accent = ref.accent_window(hops, [0, 0], [1, 1])['accent']
        acoustic = [activity_hop(i, energy=h['energy'], spectrum=h['spectrum'], rise=0., flux=0.,
                                 span_shares={1: 1.}) for i, h in enumerate(hops)]
        kwargs = dict(log2_bins=[1.], epoch=1, generation=4, epoch_start=0., generation_start=0.,
                      span_start=0., support_end=.3, means=[0.]*6, standard_deviations=[1.]*6)
        expected = section.ending_descriptor(acoustic, [], **kwargs)
        for level in (0., 1.5, 3., 8.):
            changed = [*hops[:3], raw_hop(3, level, width=4800)]
            candidate = ref.accent_window(changed, [0, 0], [1, 1])['accent']
            accents = [candidate] if candidate else []
            self.assertEqual(section.ending_descriptor(acoustic, accents, **kwargs), expected)
            self.assertEqual(section.section_activity(acoustic, accents, 1, {4}, 0., .3)['numerators'][4], 0.)
            owned = [{**a, 'span_shares': {1: 1.}} for a in accents]
            self.assertEqual(section.partition_span_activity(acoustic, owned, {1: {'start': 0., 'support_end': .3}},
                                                            1, {4}, .3)[1]['numerators'][4], 0.)
        self.assertEqual(section.section_activity(acoustic, [accent], 1, {4}, 0., .4)['numerators'][4], .5)
        self.assertEqual(expected['raw'][5], 0.)

    def test_delivered_counts_reach_section_after_event_interval_without_retroactive_change(self):
        history, ledger = section.SectionHistory(1, 4, 1, 0.), ref.AccentLedger(1, 4)
        issued = None
        delivery = None
        for i in range(5):
            hop = activity_hop(i)
            delta = section.section_activity([hop], [], 1, {4}, hop['start'], hop['end'])
            if i == 3:
                delivery = ledger.deliver(event(width=4800), .4)
            deliveries = [delivery] if i == 3 else []
            history.observe(delta, 1, 4, deliveries)
            if i == 2:
                issued = history.snapshot()
        self.assertEqual(issued['cumulative']['values'][34], 0.)
        self.assertEqual(history.snapshot()['cumulative']['values'][34], 1.)
        # A reread on a later receive delta cannot add the same sequence again.
        hop = activity_hop(5)
        delta = section.section_activity([hop], [], 1, {4}, .5, .6)
        history.observe(delta, 1, 4, [{**delivery, 'delivered_at': .6}])
        self.assertAlmostEqual(history.snapshot()['cumulative']['values'][34], .5/.6)
        self.assertEqual(issued['cumulative']['values'][34], 0.)

    def test_generation_continuation_keeps_totals_but_uses_new_delivery_sequence(self):
        history = section.SectionHistory(1, 4, 1, 0.)
        first = ref.AccentLedger(1, 4).deliver(event(width=4800), .4)
        delta = section.section_activity([activity_hop(i) for i in range(4)], [], 1, {4}, 0., .4)
        history.observe(delta, 1, 4, [first])
        frozen = history.snapshot()
        child = history.inherit(4, 5)
        second = ref.AccentLedger(1, 5).deliver(event(4, width=4800, generation=5), .8)
        delta = section.section_activity([activity_hop(i, generation=5) for i in range(4, 8)], [], 1, {5}, .4, .8)
        child.observe(delta, 1, 5, [second])
        self.assertAlmostEqual(child.snapshot()['cumulative']['values'][34], 1./.8)
        self.assertEqual(history.snapshot(), frozen)
        self.assertEqual(child.cumulative.integer(15), 1)

    def test_delivery_validation_is_atomic_and_old_section_events_add_no_credit(self):
        history = section.SectionHistory(1, 4, 1, .3)
        ledger = ref.AccentLedger(1, 4)
        delivery = ledger.deliver(event(width=4000), .4)
        delta = section.section_activity([activity_hop(3)], [], 1, {4}, .3, .4)
        before = bytes(history.cumulative.buffer)
        for invalid in ({**delivery, 'delivered_at': .5}, {**delivery, 'weight': .75}):
            with self.assertRaises(ValueError):
                history.observe(delta, 1, 4, [invalid])
            self.assertEqual(bytes(history.cumulative.buffer), before)
        history.observe(delta, 1, 4, [delivery])
        self.assertEqual(history.snapshot()['cumulative']['values'][34], 0.)


if __name__ == '__main__':
    unittest.main()
