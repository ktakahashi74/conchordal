"""Independent raw arithmetic, causal cuts and uncompressed moment comparisons."""

import copy
import math
from pathlib import Path
import random
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import temporal_descriptor_reference as ref
import temporal_accent_reference as accent


def hop(index, energy=1., spectrum=..., **extra):
    return {'epoch': 1, 'generation': 3, 'sample_rate': 48000,
            'sample_start': index*512, 'sample_end': (index+1)*512,
            'raw_support_end': (index+1)*512/48000,
            'available_end': (index+1)*512/48000, 'observed': True,
            'association_known': True, 'association_handle': 4, 'grid_id': 7,
            'bus_energy': max(energy, 1.), 'energy': energy,
            'spectrum': [energy] if spectrum is ... else spectrum, **extra}


def raw(index, values=None, **extra):
    return {'epoch': 1, 'generation': 3, 'sample_rate': 48000,
            'sample_start': index*512, 'sample_end': (index+1)*512,
            'start': index*512/48000, 'end': (index+1)*512/48000,
            'time': (index+1)*512/48000, 'values': [float(index)]*10 if values is None else values,
            'observed': True, 'gap': False, 'raw_support_start': index*512/48000,
            'raw_support_end': (index+1)*512/48000,
            'available_end': (index+1)*512/48000, **extra}


def span(**extra):
    return ref.SpanDescriptor(1, 3, 48000, 0, 512, 0., [1.]*10, **extra)


class RawDescriptorTests(unittest.TestCase):
    def test_assigned_energy_pipeline_and_hand_spectral_values(self):
        groups = accent.assigned_group_energy(8., [1., 2., 1.],
                                             [[1., 1., 1.]], [[.25, .75]], 1)
        for weight, group in zip((.25, .75), groups):
            r = ref.raw_descriptor(hop(0, bus_energy=8., **group), [7., 8., 9.], 1.)
            self.assertEqual(r['values'][:2], [8., math.sqrt(.5)])
            self.assertEqual(r['values'][2], math.log2(math.sqrt(8*weight)))
            self.assertEqual(r['values'][3:6], [None]*3)
            self.assertEqual(r['values'][6:], [weight, .25, .5, .25])

    def test_fraction_endpoints_belong_to_center(self):
        r = ref.raw_descriptor(hop(0, spectrum=[.5, .5]), [7.5, 8.5], 1.)
        self.assertEqual(r['values'][7:], [0., 1., 0.])

    def test_silence_is_known_rms_and_share_but_masked_spectral_shape(self):
        r = ref.raw_descriptor(hop(0, energy=0., bus_energy=0.), [8.], 1.)
        self.assertEqual(r['values'][2], math.log2(1e-6))
        self.assertEqual(r['values'][6], 0.)
        self.assertEqual([r['values'][j] for j in (0, 1, 7, 8, 9)], [None]*5)
        groups = accent.assigned_group_energy(1., [0.], [[1.]], [[.5, .5]], 1)
        r = ref.raw_descriptor(hop(0, **groups[1]), [8.], 1.)
        self.assertEqual(r['values'][2], 0.)
        self.assertIsNone(r['values'][0])

    def test_rise_decline_and_flux_match_independent_hand_and_accent_components(self):
        hops = [hop(i, energy=e) for i, e in enumerate((1., 4., 1., 1.))]
        r = ref.raw_descriptor(hops[1], [8.], 1., hops[0])
        self.assertEqual(r['values'][3:6], [1., 0., 1.])
        self.assertEqual(r['raw_support_start'], 0.)
        r = ref.raw_descriptor(hops[2], [8.], 1., hops[1])
        self.assertEqual(r['values'][3:6], [0., 1., 0.])
        components = accent.accent_window(hops, [0., 0.], [1., 1.])['raw_components']
        for i in range(1, 4):
            values = ref.raw_descriptor(hops[i], [8.], 1., hops[i-1])['values']
            self.assertEqual((values[3], values[5]), components[i-1])

    def test_complete_current_and_predecessor_support_cut(self):
        h = hop(1, raw_support_end=.1, available_end=.2)
        self.assertIsNone(ref.raw_descriptor(h, [8.], .19, hop(0)))
        p = hop(0, available_end=.3)
        r = ref.raw_descriptor(h, [8.], .2, p)
        self.assertEqual(r['values'][3:6], [None]*3)
        r = ref.raw_descriptor(h, [8.], .3, p)
        self.assertEqual(r['available_end'], .3)
        self.assertEqual(r['raw_support_end'], .1)
        self.assertEqual(r['raw_support_start'], 0.)

    def test_unknown_acquisition_association_and_generation_mask(self):
        for changes in ({'observed': False}, {'association_known': False}):
            r = ref.raw_descriptor(hop(0, **changes), [8.], 1.)
            self.assertEqual(r['values'], [None]*10)
        for key in ('epoch', 'generation', 'association_handle', 'grid_id'):
            r = ref.raw_descriptor(hop(1), [8.], 1., hop(0, **{key: 99}))
            self.assertEqual(r['values'][3:6], [None]*3)
        r = ref.raw_descriptor(hop(2), [8.], 1., hop(0))
        self.assertEqual(r['values'][3:6], [None]*3)

    def test_invalid_energy_grid_and_provenance(self):
        for changes in ({'energy': -1.}, {'energy': math.nan}, {'bus_energy': .5},
                        {'raw_support_end': 0.}, {'available_end': 0.},
                        {'spectrum': [-1.]}, {'spectrum': [1., 2.]}):
            with self.assertRaises(ValueError):
                ref.raw_descriptor(hop(0, **changes), [8.], 1.)
        for bins in ([], [math.nan], [8., 7.]):
            with self.assertRaises(ValueError):
                ref.raw_descriptor(hop(0), bins, 1.)

    def test_partial_acquisition_union_and_clipping_preserve_missing_weight(self):
        h = hop(0, known_sample_intervals=[(0, 100), (50, 200), (400, 512)])
        r = ref.raw_descriptor(h, [8.], 1.)
        self.assertEqual(r['values'], [None]*10)
        k = ref.DescriptorKnot.from_raw(r, 100/48000, 450/48000).snapshot()
        self.assertAlmostEqual(k['observed_sec'], 150/48000)
        self.assertAlmostEqual(k['gap_sec'], 200/48000)
        self.assertEqual(k['values'], [None]*10)
        self.assertTrue(k['gap_location_lost'])
        s = span()
        s.push(r, 1.)
        s.finish(1.)
        self.assertAlmostEqual(s.snapshot()['knots'][0]['observed_sec'], 312/48000)

    def test_partial_predecessor_never_supplies_adjacent_difference(self):
        p = hop(0, known_sample_intervals=[(0, 100), (50, 300)])
        r = ref.raw_descriptor(hop(1), [8.], 1., p)
        self.assertEqual(r['values'][3:6], [None]*3)
        self.assertEqual(r['raw_support_start'], 512/48000)
        p = hop(0, known_sample_intervals=[(0, 300), (250, 512)])
        r = ref.raw_descriptor(hop(1), [8.], 1., p)
        self.assertEqual(r['values'][3:6], [0., 0., 0.])


class MomentTests(unittest.TestCase):
    def test_snapshot_literal_moments_masks_signed_zero_and_full_width_ids(self):
        data = bytearray(320)
        weights = [math.nextafter(.9, 0.), .9]+[.95]*8
        means = [12., -0., 1e308, math.ldexp(1., -1074)]+[float(j) for j in range(6)]
        moments = [(weights[j], means[j], j/16) for j in range(10)]
        for j, moment in enumerate(moments):
            struct.pack_into('<3d', data, 24*j, *moment)
        struct.pack_into('<8dQQ', data, 240, .5, .95, 0., 1., 0., 1., 1.25, .05, 2**64-1, 2**53+1)
        out = ref.DescriptorKnot(memoryview(bytes(data))).snapshot()
        self.assertEqual(out, dict(start=0., end=1., time=.5, observed_sec=.95, raw_support_start=0.,
            raw_support_end=1., available_end=1.25, gap_sec=.05, gap=True, gap_location_lost=True,
            coverage=weights, values=[None]+means[1:], moments=moments, epoch=2**64-1, generation=2**53+1))
        self.assertEqual(struct.pack('<9d', *out['values'][1:]), struct.pack('<9d', *means[1:]))
        self.assertEqual(struct.pack('<30d', *(v for row in out['moments'] for v in row)), data[:240])
        self.assertIs(type(out['epoch']), int)
        self.assertIs(type(out['generation']), int)

    def test_binary_layout_and_clipped_assignment_preserve_source_support(self):
        r = raw(1, raw_support_start=0., available_end=.1)
        a, b = r['start']+.001, r['end']-.002
        knot = ref.DescriptorKnot.from_raw(r, a, b)
        self.assertEqual(len(knot.data), 320)
        self.assertEqual(struct.unpack_from('<QQ', knot.data, 304), (1, 3))
        out = knot.snapshot()
        self.assertEqual(out['time'], r['time'])
        self.assertEqual(out['raw_support_start'], 0.)
        self.assertEqual(out['raw_support_end'], r['end'])
        self.assertEqual(out['moments'][0], (b-a, 1., 0.))
        self.assertEqual(out['available_end'], .1)

    def test_weighted_moments_against_two_point_hand_calculation(self):
        a = ref.DescriptorKnot.from_raw(raw(0, [2.]*10))
        b = ref.DescriptorKnot.from_raw(raw(1, [8.]*10))
        a.merge(b)
        weight, mean, error = a.snapshot()['moments'][0]
        self.assertEqual(mean, 5.)
        self.assertAlmostEqual(weight, 1024/48000)
        self.assertAlmostEqual(error, 18*512/48000)
        self.assertAlmostEqual(a.snapshot()['time'], 768/48000)

    def test_missing_coordinate_retains_other_statistics_without_zero_imputation(self):
        a = ref.DescriptorKnot.from_raw(raw(0, [None, 5.]+[None]*8))
        b = ref.DescriptorKnot.from_raw(raw(1, [9., None]+[None]*8))
        a.merge(b)
        out = a.snapshot()
        self.assertEqual(out['moments'][0], (512/48000, 9., 0.))
        self.assertEqual(out['moments'][1], (512/48000, 5., 0.))
        self.assertEqual(out['values'], [None]*10)

    def test_gap_weight_time_coverage_and_location_loss(self):
        r = raw(0, [None]*10, observed=False, gap=True)
        a = ref.DescriptorKnot.from_raw(r)
        self.assertEqual(a.snapshot()['time'], 256/48000)
        self.assertEqual(a.snapshot()['observed_sec'], 0.)
        self.assertFalse(a.snapshot()['gap_location_lost'])
        a.merge(ref.DescriptorKnot.from_raw(raw(1)))
        out = a.snapshot()
        self.assertAlmostEqual(out['time'], 1024/48000)
        self.assertTrue(out['gap'])
        self.assertTrue(out['gap_location_lost'])
        self.assertEqual(out['values'], [None]*10)

    def test_noncontiguous_and_different_lineage_rejected(self):
        a = ref.DescriptorKnot.from_raw(raw(0))
        for r in (raw(2), raw(1, generation=4), raw(1, epoch=2)):
            with self.assertRaises(ValueError):
                a.merge(ref.DescriptorKnot.from_raw(r))

    def test_coverage_threshold_against_direct_durations(self):
        for known in (8, 9, 10):
            a = None
            for i in range(10):
                r = {'epoch': 1, 'generation': 3, 'start': float(i), 'end': float(i+1),
                     'time': float(i+1), 'values': [1. if i < known else None]*10,
                     'observed': True, 'gap': False, 'raw_support_start': float(i),
                     'raw_support_end': float(i+1), 'available_end': float(i+1)}
                b = ref.DescriptorKnot.from_raw(r)
                if a is None:
                    a = b
                else:
                    a.merge(b)
            self.assertEqual(a.snapshot()['coverage'], [known/10]*10)
            self.assertEqual(a.snapshot()['values'], [1. if known >= 9 else None]*10)


class CompressionTests(unittest.TestCase):
    def test_equal_priority_protects_first_and_newest_and_earliest_tie(self):
        bank = ref.BoundedDescriptor([1.]*10, capacity=4)
        for i in range(5):
            bank.append(ref.DescriptorKnot.from_raw(raw(i, [2.]*10)))
        out = bank.snapshot()
        self.assertEqual([(k['start'], k['end']) for k in out['knots']],
                         [(0., 512/48000), (512/48000, 1536/48000),
                          (1536/48000, 2048/48000), (2048/48000, 2560/48000)])
        self.assertEqual(out['reconstruction_error'], 0.)
        self.assertEqual(out['priority_coordinate_evaluations'], 20)

    def test_global_scales_change_priority_without_dropping_coordinates(self):
        chosen = []
        for scales in ([1.]*10, [100., 1.]+[1.]*8):
            bank = ref.BoundedDescriptor(scales, 4)
            for i, values in enumerate(([0.]*10, [0.]*10, [10., 0.]+[0.]*8,
                                        [10., 2.]+[0.]*8, [0.]*10)):
                bank.append(ref.DescriptorKnot.from_raw(raw(i, values)))
            chosen.append([(k['start'], k['end']) for k in bank.snapshot()['knots']])
        self.assertNotEqual(chosen[0], chosen[1])

    def test_error_matches_direct_uncompressed_reconstruction_for_100_streams(self):
        rng = random.Random(47012)
        for _ in range(100):
            scales = [rng.uniform(.1, 3.) for _ in range(10)]
            bank = ref.BoundedDescriptor(scales, 8)
            records = []
            for i in range(40):
                values = [None if rng.random() < .15 else rng.gauss(0., 2.) for _ in range(10)]
                r = raw(i, values)
                records.append(r)
                bank.append(ref.DescriptorKnot.from_raw(r))
            direct_terms = []
            for k in bank.snapshot()['knots']:
                contained = [r for r in records if r['start'] >= k['start'] and r['end'] <= k['end']]
                for j, sd in enumerate(scales):
                    samples = [(r['end']-r['start'], r['values'][j]) for r in contained
                               if r['values'][j] is not None]
                    weight = math.fsum(w for w, _ in samples)
                    self.assertAlmostEqual(weight, k['moments'][j][0], places=13)
                    if weight:
                        mean = math.fsum(w*x for w, x in samples)/weight
                        self.assertAlmostEqual(mean, k['moments'][j][1], places=12)
                        direct_terms.extend(w*((x-k['moments'][j][1])/sd)**2 for w, x in samples)
            direct = math.fsum(direct_terms)
            self.assertLessEqual(abs(direct-bank.reconstruction_error), 1e-11*max(1., direct))

    def test_large_centroid_offsets_preserve_small_nonzero_error(self):
        bank = ref.BoundedDescriptor([1.]*10, 4)
        for i in range(20):
            bank.append(ref.DescriptorKnot.from_raw(raw(i, [1e8+i*.01]*10)))
        self.assertGreater(bank.reconstruction_error, 0.)
        self.assertTrue(math.isfinite(bank.reconstruction_error))

    def test_bank_and_scratch_capacities_half_double_with_long_input(self):
        for capacity in (64, 128, 256):
            bank = ref.BoundedDescriptor([1.]*10, capacity)
            identity = id(bank.storage)
            for i in range(capacity+20):
                bank.append(ref.DescriptorKnot.from_raw(raw(i)))
            self.assertEqual(bank.count, capacity)
            self.assertEqual(id(bank.storage), identity)
            self.assertEqual(len(bank.storage), capacity*320)
            self.assertEqual(len(bank.scratch.data), 320)
            self.assertEqual(bank.priority_coordinate_evaluations, 20*(capacity-2)*10)

    def test_freeze_cut_and_repeat_read_do_not_change_payload(self):
        bank = ref.BoundedDescriptor([1.]*10)
        bank.append(ref.DescriptorKnot.from_raw(raw(0, available_end=.2)))
        before = bytes(bank.storage)
        with self.assertRaises(ValueError):
            bank.freeze(.1)
        self.assertEqual(bytes(bank.storage), before)
        frozen = bank.freeze(.2)
        bank.snapshot()['knots'][0]['values'][0] = 999.
        self.assertEqual(bank.freeze(.3), frozen)
        with self.assertRaises(ValueError):
            bank.append(ref.DescriptorKnot.from_raw(raw(1)))


class CadenceTests(unittest.TestCase):
    def test_pairing_clipped_start_and_coincident_span_endpoint_once(self):
        stream = ref.SpanDescriptor(1, 3, 48000, 0, 512, 700/48000,
                                    [1.]*10, span_end=2048/48000)
        for i in range(1, 4):
            stream.push(raw(i), 1.)
        stream.finish(1.)
        knots = stream.snapshot()['knots']
        self.assertEqual([(k['start'], k['end']) for k in knots],
                         [(700/48000, 1024/48000), (1024/48000, 2048/48000)])
        self.assertEqual(knots[0]['raw_support_start'], 512/48000)

    def test_half_double_cadence_uses_same_raw_samples(self):
        for cadence, lengths in ((1, [1]*7), (2, [2, 2, 2, 1]), (4, [4, 3])):
            stream = span(cadence=cadence)
            for i in range(7):
                stream.push(raw(i), 1.)
            stream.finish(1.)
            knots = stream.snapshot()['knots']
            self.assertEqual(len(knots), len(lengths))
            for k, n in zip(knots, lengths):
                self.assertAlmostEqual(k['end']-k['start'], n*512/48000)

    def test_explicit_gap_reports_coalesce_without_hop_fill(self):
        stream = span()
        stream.push(raw(0), 1.)
        stream.gap(100., 100., 100.)
        stream.gap(1000., 1000., 1000.)
        self.assertEqual(stream.bank.count, 1)
        self.assertTrue(stream.has_gap)
        stream.finish(1000.)
        out = stream.snapshot()
        self.assertEqual(len(out['knots']), 2)
        self.assertEqual(out['knots'][1]['values'], [None]*10)
        self.assertEqual(out['knots'][1]['observed_sec'], 0.)

    def test_implicit_gap_resume_has_three_insertions_maximum(self):
        stream = span()
        stream.push(raw(0), 1.)
        stream.push(raw(3), 1.)
        self.assertEqual(stream.last_insertions, 3)
        self.assertEqual(stream.max_insertions, 3)
        self.assertEqual([k['gap'] for k in stream.snapshot()['knots']], [False, True, False])
        self.assertEqual(stream.snapshot()['knots'][1]['start'], 512/48000)
        self.assertEqual(stream.snapshot()['knots'][1]['end'], 1536/48000)

    def test_acquisition_gap_raw_records_coalesce(self):
        stream = span()
        for i in range(4):
            stream.push(raw(i, [None]*10, observed=False, gap=True), 1.)
        self.assertEqual(stream.bank.count, 0)
        stream.finish(1.)
        self.assertEqual(stream.bank.count, 1)
        self.assertEqual(stream.snapshot()['knots'][0]['gap_sec'], 2048/48000)

    def test_replay_conflict_and_new_generation_never_reuse_weight(self):
        stream = span()
        r = raw(0)
        stream.push(r, 1.)
        state = stream.snapshot()
        self.assertFalse(stream.push(copy.deepcopy(r), 1.))
        self.assertEqual(stream.snapshot(), state)
        for invalid in (raw(0, [9.]*10), raw(1, generation=4), raw(1, epoch=2)):
            with self.assertRaises(ValueError):
                stream.push(invalid, 1.)
        self.assertEqual(stream.snapshot(), state)
        fresh = ref.SpanDescriptor(1, 4, 48000, 512, 512, 512/48000, [1.]*10)
        fresh.push(raw(1, generation=4), 1.)
        fresh.finish(1.)
        self.assertEqual(fresh.snapshot()['knots'][0]['start'], 512/48000)

    def test_cut_clock_and_declared_incomplete_span_rejected_without_flush(self):
        stream = span(span_end=1024/48000)
        stream.push(raw(0), .1)
        state = stream.snapshot()
        for operation in (lambda: stream.push(raw(1), .05),
                          lambda: stream.push(raw(1, available_end=.3), .2),
                          lambda: stream.finish(.2)):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(stream.snapshot(), state)

    def test_transposition_and_tempo_copy_observable_equivariance(self):
        outputs = []
        for shift, speed in ((0., 1), (2., 2)):
            stream = ref.SpanDescriptor(1, 3, 48000//speed, 0, 512, 0., [1.]*10)
            previous = None
            for i in range(8):
                h = hop(i, energy=1.+i)
                h['sample_rate'] //= speed
                h['raw_support_end'] *= speed
                h['available_end'] *= speed
                r = ref.raw_descriptor(h, [8.+shift], 1., previous)
                stream.push(r, 1.)
                previous = h
            stream.finish(1.)
            outputs.append(stream.snapshot())
        for a, b in zip(outputs[0]['knots'], outputs[1]['knots']):
            self.assertAlmostEqual(b['time'], 2*a['time'])
            self.assertAlmostEqual(b['values'][0], a['values'][0]+2)
            self.assertEqual(b['values'][1:], a['values'][1:])
            self.assertAlmostEqual(b['observed_sec'], 2*a['observed_sec'])


if __name__ == '__main__':
    unittest.main()
