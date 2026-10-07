"""Retained independent raw arithmetic and causal-support controls."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import temporal_feature_reference as ref
import temporal_accent_reference as accent


def hop(index, energy=1., spectrum=..., **extra):
    return {'epoch': 1, 'generation': 3, 'sample_rate': 48000,
            'sample_start': index*512, 'sample_end': (index+1)*512,
            'raw_support_end': (index+1)*512/48000,
            'available_end': (index+1)*512/48000, 'observed': True,
            'association_known': True, 'association_handle': 4, 'grid_id': 7,
            'bus_energy': max(energy, 1.), 'energy': energy,
            'spectrum': [energy] if spectrum is ... else spectrum, **extra}


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
        self.assertEqual(r['known_sample_intervals'], [(0, 200), (400, 512)])
        self.assertTrue(r['gap'])

    def test_partial_predecessor_never_supplies_adjacent_difference(self):
        p = hop(0, known_sample_intervals=[(0, 100), (50, 300)])
        r = ref.raw_descriptor(hop(1), [8.], 1., p)
        self.assertEqual(r['values'][3:6], [None]*3)
        self.assertEqual(r['raw_support_start'], 512/48000)
        p = hop(0, known_sample_intervals=[(0, 300), (250, 512)])
        r = ref.raw_descriptor(hop(1), [8.], 1., p)
        self.assertEqual(r['values'][3:6], [0., 0., 0.])
