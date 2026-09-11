"""Native bounded DP versus the independent Python recurrence and full driver."""

import copy
import ctypes as ct
import gc
import importlib.util
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_temporal_matcher_kernel as native
import temporal_matcher_reference as reference
from temporal_consumer_packet_reference import pack_knots

spec = importlib.util.spec_from_file_location('matcher_test_fixtures',
                                             ROOT / 'tests/test_evaluate_temporal_matcher_reference.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def exact(value):
    if isinstance(value, float):
        return ('f64', struct.pack('<d', value))
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(exact(item) for item in value)
    return (type(value).__name__, value)


def packed_record(record):
    return {**record,'knots':native.PackedKnots(pack_knots(record['knots']) if record['knots'] else b'')}


class NativeMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='temporal-dtw-test-')
        cls.library = Path(cls.directory.name) / 'libtemporal_dtw.so'
        subprocess.run(['rustc', '--edition', '2021', '--crate-type', 'cdylib', '-O',
                        '--cfg', 'feature="python-binding"',
                        str(ROOT / 'scripts/temporal_matcher_kernel.rs'), '-o', str(cls.library)],
                       check=True, capture_output=True, text=True)
        cls.numeric_library = Path(cls.directory.name) / 'libtemporal_numeric.so'
        subprocess.run(['rustc', '--edition', '2021', '--crate-type', 'cdylib', '-D', 'warnings', '-O',
                        str(ROOT / 'scripts/temporal_matcher_kernel.rs'), '-o', str(cls.numeric_library)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.worker = native.NativeMatcher(self.library)

    def compare(self, cue, refs, transform=None, **kwargs):
        transform = transform or fixtures.fixed_transform()
        before = copy.deepcopy((cue, refs, transform))
        scales = kwargs.pop('scales', [1.]*10)
        a = reference.subsequence_dtw(cue, refs, transform, scales, 10000., **kwargs)
        b = self.worker.subsequence_dtw(cue, refs, transform, scales, 10000., **kwargs)
        self.assertEqual(exact(a), exact(b))
        self.assertEqual((cue, refs, transform), before)
        return b

    def test_random_masks_original_times_penalties_and_bands(self):
        rng = random.Random(47034)
        for case in range(700):
            rows = []
            for start in (100., 0.):
                values, times, clock = [], [], start
                for _ in range(rng.randint(1, 14)):
                    values.append([None if rng.random() < .4 else rng.uniform(-2., 2.) for _ in range(10)])
                    clock += rng.choice((.125, .5, 1., 2.))
                    times.append(clock)
                rows.append(fixtures.knots(values, start=start, times=times))
            cue, refs = rows
            transform = fixtures.fixed_transform(rng.randrange(len(refs)),
                                                  rng.choice((-.5, 0., .25)),
                                                  rng.choice((-1., 0., .5, 1.)))
            with self.subTest(case=case):
                self.compare(cue, refs, transform, scales=[rng.uniform(.1, 2.) for _ in range(10)],
                             band_radius=rng.choice((None, 0, 1, 2, 16, 128, 1000)),
                             insertion_penalty=rng.choice((.5, 1., 2.)),
                             deletion_penalty=rng.choice((.5, 1., 2.)))

    def test_all_masks_and_exact_ties(self):
        for mask in range(1024):
            vector = [0. if mask & (1 << j) else None for j in range(10)]
            result = self.compare(fixtures.knots([vector]*2, start=100.), fixtures.knots([vector]*3),
                                  band_radius=None)
            self.assertEqual(result['path'], [('pair', 0, 0), ('pair', 1, 1)])

    def test_full_capacity_and_reused_scratch(self):
        for n, m in ((128, 128), (1, 1), (127, 2), (2, 128), (128, 127)):
            for band in (None, 0, 16):
                a = fixtures.knots([[math.sin(i*(k+1)) for k in range(10)] for i in range(n)], start=1000.)
                b = fixtures.knots([[math.cos(i*(k+1)) for k in range(10)] for i in range(m)])
                self.compare(a, b, band_radius=band)

    def test_empty_unknown_bounds_and_missing_support(self):
        rows = fixtures.knots([None, None, None])
        self.compare([], rows)
        self.compare(rows, [])
        self.compare(rows, rows)
        for flag in ('bound_hit', 'out_of_range'):
            transform = fixtures.fixed_transform()
            transform[flag] = True
            self.compare(rows, rows, transform)

    def test_nearest_time_earlier_tie_and_disconnected_band(self):
        cue = fixtures.knots([0., 0., 0.], start=100., times=[101., 102., 200.])
        refs = fixtures.knots([0., 0., 0.], times=[1., 3., 4.])
        self.compare(cue, refs, band_radius=0)
        result = self.compare(cue[:2], refs, band_radius=0)
        self.assertEqual(result['band_ranges'], [(0, 1), (0, 1)])

    def test_gap_diagnostics_and_extreme_supported_floats(self):
        self.compare(fixtures.knots([1., None, 2., 3.], start=100.),
                     fixtures.knots([1., None, 2., 3.]))
        self.compare(fixtures.knots([1e300], start=100.), fixtures.knots([-1e300]))
        self.compare(fixtures.knots([1., 2.], start=100.), fixtures.knots([1., 2.]),
                     fixtures.fixed_transform(tempo=-2000.))

    def test_invalid_inputs_match_reference_rejection(self):
        rows = fixtures.knots([1., 2.])
        for field, value in (('time', math.nan), ('available_end', 20000.),
                             ('observed_sec', -1.), ('raw_support_start', 10.)):
            broken = copy.deepcopy(rows)
            broken[0][field] = value
            for function in (reference.subsequence_dtw, self.worker.subsequence_dtw):
                with self.assertRaises(ValueError):
                    function(broken, rows, fixtures.fixed_transform(), [1.]*10, 10000.)
        for arguments in ({'scales': [0.]*10}, {'band_radius': -1}, {'insertion_penalty': 0.}):
            for function in (reference.subsequence_dtw, self.worker.subsequence_dtw):
                kwargs = dict(arguments)
                scales = kwargs.pop('scales', [1.]*10)
                with self.assertRaises(ValueError):
                    function(rows, rows, fixtures.fixed_transform(), scales, 10000., **kwargs)

    def test_native_boundary_rejects_without_touching_output(self):
        self.compare(fixtures.knots([0.]), fixtures.knots([0.]))
        c = self.worker.config
        before = bytes(self.worker.output)
        for field, value in (('n', 129), ('m', 0), ('anchor', 1), ('band', -2),
                             ('ratio', -1.), ('shift', math.nan)):
            old = getattr(c, field)
            setattr(c, field, value)
            self.assertEqual(self.worker.call(self.worker.cue, self.worker.reference,
                                              ct.byref(c), ct.byref(self.worker.output)), 1)
            self.assertEqual(bytes(self.worker.output), before)
            setattr(c, field, old)
        self.assertEqual(self.worker.call(None, None, None, None), 1)

    def test_full_query_rank_cache_cutoff_and_original_function_isolation(self):
        original = reference.subsequence_dtw
        for missing in (False, True):
            values = [None if missing and i % 3 == 0 else math.sin(i) for i in range(12)]
            q = fixtures.query(values)
            episodes = [fixtures.episode(values, identity=i) for i in range(20)]
            a = reference.match_query(q, episodes, 1000.)
            b = self.worker.driver.match_query(q, episodes, 1000.)
            self.assertEqual(exact(a), exact(b))
            self.assertTrue(b['cutoff_tie'])
            self.assertEqual(len(b['excluded_episode_ids']), 4)
        self.assertIs(reference.subsequence_dtw, original)

    def test_native_anchor_inventory_against_every_python_anchor(self):
        rng = random.Random(47036)
        for case in range(500):
            sequences = []
            for start, maximum in ((100., 32), (0., 128)):
                values, times, clock = [], [], start
                for i in range(rng.randint(1, maximum)):
                    values.append([None if rng.random() < .45 else rng.uniform(-3., 3.) for _ in range(10)])
                    clock += rng.choice((.125, .5, 1., 2.))
                    times.append(clock)
                sequences.append(fixtures.knots(values, start=start, times=times))
            cue, refs = sequences
            scales = [rng.uniform(.1, 3.) for _ in range(10)]
            spacing, limit = rng.choice((1, 2, 4, 8, 128, 1000)), rng.choice((1, 8, 32, 128, 1000))
            bounds, grid = (rng.choice((.125, 1., 2., 8.)), 2.), rng.choice((1/16, 1/64, .1, .003))
            self.worker._pack(cue, self.worker.cue)
            self.worker._pack(refs, self.worker.reference)
            count = self.worker._run_coarse(len(cue), len(refs), scales, spacing, limit, bounds, grid)
            expected = list(range(0, min(len(refs), spacing*limit), spacing))
            self.assertEqual(count, len(expected))
            candidates, comparison_total, bound_total = [], 0, 0
            for index, anchor in enumerate(expected):
                transform = reference.anchor_transform(cue, refs, anchor, bounds, grid)
                score = reference.coarse_cost(cue, refs, anchor, transform, scales)
                comparison_total += transform['median_comparisons']
                bound_total += transform['bound_hit']
                if score['cost'] is not None:
                    candidates.append((score['cost'],anchor,index,transform))
                row = self.worker.anchors[index]
                with self.subTest(case=case, anchor=anchor):
                    self.assertEqual(exact(score), exact({'cost': None if math.isnan(row.cost) else row.cost,
                        'valid_coordinates': row.valid, 'paired_steps': row.pairs}))
                    lengths = (row.pitch_samples, row.interval_samples)
                    self.assertEqual(exact(transform['unrounded']), exact([
                        row.unrounded[j] if lengths[j] else None for j in range(2)]))
                    self.assertEqual(exact(transform['applied']), exact(list(row.applied)))
                    self.assertEqual(transform['median_comparisons'], row.comparisons)
                    self.assertEqual(transform['out_of_range'], bool(row.out_of_range))
                    self.assertEqual(transform['bound_hit'], bool(row.bound_hit))
                    self.assertEqual(transform['pitch_samples'], row.pitch_samples)
                    self.assertEqual(transform['interval_samples'], row.interval_samples)
            diag = self.worker.anchor_diagnostic
            self.assertEqual(diag.evaluated,count)
            self.assertEqual(diag.comparisons,comparison_total)
            self.assertEqual(diag.bound_anchors,bound_total)
            if not candidates:
                self.assertEqual(diag.index,-1)
            else:
                _,_,index,transform = min(candidates,key=lambda row:row[:2])
                self.assertEqual(diag.index,index)
                for errors,length,key in ((diag.pitch_error,transform['pitch_samples'],'pitch_estimate_residual_rms'),
                                          (diag.interval_error,transform['interval_samples'],'tempo_estimate_residual_rms')):
                    value = math.sqrt(math.fsum(errors[:length])/length) if length else None
                    self.assertEqual(exact(value),exact(transform[key]))

    def test_anchor_signed_zero_half_grid_boundaries_and_all_missing(self):
        for pitch in (-0., 0., -.03125, .03125, -1.96875, 1.96875, -2., 2., None):
            for missing in (False, True):
                cue = fixtures.knots([pitch, None if missing else pitch], start=100.)
                refs = fixtures.knots([0., 0.])
                self.worker._pack(cue, self.worker.cue)
                self.worker._pack(refs, self.worker.reference)
                self.worker._run_coarse(2, 2, [1.]*10)
                row = self.worker.anchors[0]
                transform = reference.anchor_transform(cue, refs, 0)
                score = reference.coarse_cost(cue, refs, 0, transform, [1.]*10)
                self.assertEqual(exact(transform['applied']), exact(list(row.applied)))
                self.assertEqual(exact(score['cost']), exact(None if math.isnan(row.cost) else row.cost))

    def test_random_whole_queries_preserve_causality_ranking_and_caps(self):
        rng = random.Random(47037)
        for case in range(90):
            values = [[None if rng.random() < .3 else rng.uniform(-.2, .2) for _ in range(10)]
                      for _ in range(rng.randint(1, 15))]
            q = fixtures.query(values)
            episodes = []
            for identity in range(rng.randint(1, 25)):
                copied = copy.deepcopy(values)
                if identity % 2:
                    rng.shuffle(copied)
                e = fixtures.episode(copied, identity=identity)
                if identity % 7 == 1:
                    e['available_end'] = 2000.
                if identity % 11 == 3:
                    e['epoch'] = 2
                if identity % 13 == 4:
                    e['first_observed_end'] = q['query_end']
                episodes.append(e)
            kwargs = {'candidate_limit': rng.choice((1, 4, 16)), 'anchor_spacing': rng.choice((1, 2, 4, 8)),
                      'anchor_limit': rng.choice((1, 8, 32)), 'band_radius': rng.choice((None, 0, 2, 16)),
                      'transform_limit': rng.choice((1, 2, 4))}
            before = copy.deepcopy((q, episodes))
            a = reference.match_query(q, episodes, 1000., **kwargs)
            b = self.worker.match_query(q, episodes, 1000., **kwargs)
            with self.subTest(case=case):
                self.assertEqual(exact(a), exact(b))
                self.assertEqual((q, episodes), before)

    def test_coarse_invalid_native_fields_leave_results_unchanged(self):
        rows = fixtures.knots([0., 0.])
        self.worker._pack(rows, self.worker.cue)
        self.worker._pack(rows, self.worker.reference)
        self.worker._run_coarse(2, 2, [1.]*10)
        c = self.worker.coarse_config
        before = bytes(self.worker.anchors)
        diagnostic_before = bytes(self.worker.anchor_diagnostic)
        for field, value in (('n', 129), ('m', 129), ('spacing', 0), ('limit', 0), ('grid', 0.)):
            old = getattr(c, field)
            setattr(c, field, value)
            self.assertEqual(self.worker.coarse_call(self.worker.cue, self.worker.reference,
                                                    ct.byref(c), self.worker.anchors, ct.byref(self.worker.anchor_diagnostic)), 1)
            self.assertEqual(bytes(self.worker.anchors), before)
            self.assertEqual(bytes(self.worker.anchor_diagnostic), diagnostic_before)
            setattr(c, field, old)
        self.assertEqual(self.worker.coarse_call(None, None, None, None, None), 1)

    def test_coarse_rounding_overflow_and_log_domain_match_reference(self):
        for case in ('pitch_overflow', 'grid_overflow', 'log_domain'):
            cue, refs = fixtures.knots([1., 2.]), fixtures.knots([1., 2.])
            grid = 1/16
            if case == 'pitch_overflow':
                cue[0]['values'][0], refs[0]['values'][0] = 1e308, -1e308
            if case == 'grid_overflow':
                cue[0]['values'][0], grid = 1e100, 1e-300
            if case == 'log_domain':
                cue[0]['time'], cue[1]['time'] = 0., 1e308
                refs[0]['time'], refs[1]['time'] = 0., 1e-308
            self.worker._pack(cue, self.worker.cue)
            self.worker._pack(refs, self.worker.reference)
            error = ValueError if case == 'log_domain' else OverflowError
            with self.assertRaises(error):
                reference.anchor_transform(cue, refs, 0, grid=grid)
            if case == 'log_domain':
                with self.assertRaises(error):
                    self.worker._run_coarse(2, 2, [1.]*10, grid=grid)
            else:
                try:
                    result = self.worker._run_coarse(2, 2, [1.]*10, grid=grid)
                except error:
                    pass
                else:
                    self.assertIsNone(result)

    def test_extreme_unused_anchor_diagnostics_keep_reference_failures(self):
        for value in (2e153, 1e200):
            q = fixtures.query([0.]*4)
            episodes = [fixtures.episode([0., 0., 0., 0., 0., 0., value, 0.])]
            before = copy.deepcopy((q, episodes))
            try:
                expected = reference.match_query(q, episodes, 1000.)
            except OverflowError:
                with self.assertRaises(OverflowError):
                    self.worker.match_query(q, episodes, 1000.)
            else:
                actual = self.worker.match_query(q, episodes, 1000.)
                self.assertEqual(exact(expected), exact(actual))
            self.assertEqual((q, episodes), before)

    def test_overflowed_time_intervals_preserve_nan_diagnostics(self):
        q, episode = fixtures.query([0.]*4), fixtures.episode([0.]*4)
        for obj, times in ((q, [-1e308, 1e308, 1.05e308, 1.1e308]),
                           (episode, [-9e307, 9e307, 9.5e307, 1e308])):
            start = times[0]-1e306
            for row, end in zip(obj['knots'], times):
                row.update(start=start, end=end, time=end, raw_support_start=start,
                           raw_support_end=end, available_end=end, observed_sec=1.)
                start = end
            obj['available_end'] = times[-1]
        q['query_end'] = q['supporting_audio_end'] = 1.1e308
        episode['first_observed_end'] = 1e308
        expected = reference.match_query(q, [episode], 1.2e308)
        self.assertEqual(expected['status'], 'completed')
        self.assertTrue(math.isnan(expected['matches'][0]['tempo_estimate_residual_rms']))
        self.assertEqual(exact(expected), exact(self.worker.match_query(q, [episode], 1.2e308)))

    def test_native_validation_random_geometry_masks_and_rejections(self):
        rng = random.Random(47039)
        fields = ('start','end','time','observed_sec','raw_support_start','raw_support_end','available_end')
        for case in range(1200):
            rows = fixtures.knots([[None if rng.random() < .3 else rng.uniform(-2.,2.) for _ in range(10)]
                                   for _ in range(rng.randint(1,16))])
            selected = rows[rng.randrange(len(rows))]
            if case % 4:
                field = rng.choice(fields)
                selected[field] = rng.choice((math.nan, math.inf, -math.inf, -1., 0., .5, 1001.))
            elif case % 8 == 0:
                selected['values'][rng.randrange(10)] = rng.choice((math.nan, math.inf))
            before = copy.deepcopy(rows)
            expected_error = None
            try:
                reference._validate(rows,1000.,128)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                expected_error = type(error)
            with self.subTest(case=case):
                if expected_error is None:
                    self.worker.validate_and_pack(rows,self.worker.cue,1000.)
                else:
                    with self.assertRaises(expected_error):
                        self.worker.validate_and_pack(rows,self.worker.cue,1000.)
                self.assertEqual(exact(rows),exact(before))

    def test_validation_preserves_full_width_and_opaque_lineage(self):
        for epoch,generation in ((2**80+1,2**100+7),('epoch-a','generation-a')):
            rows = fixtures.knots([1.,2.])
            for row in rows:
                row['epoch'],row['generation'] = epoch,generation
            self.worker.validate_and_pack(rows,self.worker.cue,10.)
            for field,changed in (('epoch',str(epoch)+'x'),('generation',str(generation)+'x')):
                original = rows[1][field]
                rows[1][field] = changed
                with self.assertRaises(ValueError):
                    reference._validate(rows,10.,128)
                with self.assertRaises(ValueError):
                    self.worker.validate_and_pack(rows,self.worker.cue,10.)
                rows[1][field] = original

    def test_validation_original_time_bounds_and_observation_tolerance(self):
        rows = fixtures.knots([1.,2.])
        rows[0]['time'] = -1.
        for offset,valid in ((0.,True),(1e-12,True),(2e-12,False)):
            rows[0]['observed_sec'] = 1.+offset
            if valid:
                reference._validate(rows,10.,128)
                self.worker.validate_and_pack(rows,self.worker.cue,10.)
            else:
                with self.assertRaises(ValueError):
                    self.worker.validate_and_pack(rows,self.worker.cue,10.)
        rows[0]['observed_sec'] = -0.
        self.worker.validate_and_pack(rows,self.worker.cue,10.)
        self.assertEqual(struct.pack('<d',self.worker.cue[0].observed_sec),struct.pack('<d',-0.))

    def test_canonical_validation_uses_native_and_noncanonical_uses_reference(self):
        rows = fixtures.knots([1.,2.])
        with mock.patch.object(reference,'_validate',wraps=reference._validate) as python_validation:
            self.worker.validate_and_pack(rows,self.worker.cue,10.)
            self.assertEqual(python_validation.call_count,0)
            rows[0]['start'] = 0
            self.worker.validate_and_pack(rows,self.worker.cue,10.)
            self.assertEqual(python_validation.call_count,1)
            rows[0]['start'] = 0.
            rows[0]['values'][0] = 1
            self.worker.validate_and_pack(rows,self.worker.cue,10.)
            self.assertEqual(python_validation.call_count,2)

    def test_malformed_shapes_preserve_reference_error_precedence(self):
        examples = []
        for size in (9,11):
            rows = fixtures.knots([1.,2.]); rows[0]['values'] = [0.]*size; examples.append(rows)
        for field in ('values','start','available_end','generation'):
            rows = fixtures.knots([1.,2.]); del rows[1][field]; examples.append(rows)
        rows = fixtures.knots([1.,2.]); rows[0]['values'][0] = 'invalid'; examples.append(rows)
        rows = fixtures.knots([1.,2.]); rows[0]['values'][0] = math.nan; del rows[1]['epoch']; examples.append(rows)
        for rows in examples:
            try:
                reference._validate(rows,10.,128)
            except (KeyError,TypeError,ValueError,OverflowError) as error:
                with self.assertRaises(type(error)):
                    self.worker.validate_and_pack(rows,self.worker.cue,10.)
            else:
                self.fail('malformed fixture unexpectedly accepted')

    def test_validation_rechecks_same_objects_and_original_availability_cut(self):
        rows = fixtures.knots([1.,2.])
        self.worker.validate_and_pack(rows,self.worker.cue,10.)
        rows[1]['available_end'] = 11.
        with self.assertRaises(ValueError):
            self.worker.validate_and_pack(rows,self.worker.cue,10.)
        self.worker.validate_and_pack(rows,self.worker.cue,12.)
        rows[1]['generation'] += 1
        with self.assertRaises(ValueError):
            self.worker.validate_and_pack(rows,self.worker.cue,12.)
        rows[1]['generation'] -= 1
        self.worker.validate_and_pack(rows,self.worker.cue,12.)

    def test_native_validator_is_read_only_and_checks_abi_bounds(self):
        rows = fixtures.knots([1.]*128)
        self.worker._pack(rows,self.worker.cue)
        before = bytes(self.worker.cue),bytes(self.worker.output)
        self.assertEqual(self.worker.validate_call(self.worker.cue,128,128.),0)
        for count,cut in ((129,1000.),(128,127.),(128,math.inf),(128,math.nan)):
            self.assertEqual(self.worker.validate_call(self.worker.cue,count,cut),1)
        self.assertEqual(self.worker.validate_call(None,0,1.),1)
        self.assertEqual((bytes(self.worker.cue),bytes(self.worker.output)),before)
        self.assertEqual(self.worker.validate_call(self.worker.cue,0,1.),0)

    def test_cpython_transfer_matches_python_bytes_and_preserves_buffer_tails(self):
        rng = random.Random(47040)
        for case in range(320):
            rows = fixtures.knots([[None if rng.random() < .3 else rng.uniform(-10.,10.) for _ in range(10)]
                                   for _ in range(rng.randint(1,32))])
            if case % 2:
                rows[0]['values'][0] = -0.
                rows[0]['observed_sec'] = -0.
            for row in rows:
                row['epoch'],row['generation'] = 2**80+1,2**100+7
            ct.memset(ct.addressof(self.worker.cue),0xa5,ct.sizeof(self.worker.cue))
            ct.memset(ct.addressof(self.worker.reference),0x5a,ct.sizeof(self.worker.reference))
            self.assertTrue(self.worker._pack(rows,self.worker.cue))
            with mock.patch.object(self.worker,'pack_call',return_value=1):
                self.assertTrue(self.worker._pack(rows,self.worker.reference))
            end = len(rows)*ct.sizeof(native.Knot)
            self.assertEqual(bytes(self.worker.cue)[:end],bytes(self.worker.reference)[:end])
            self.assertEqual(bytes(self.worker.cue)[end:],b'\xa5'*(ct.sizeof(self.worker.cue)-end))
            self.assertEqual(bytes(self.worker.reference)[end:],b'\x5a'*(ct.sizeof(self.worker.reference)-end))

    def test_cpython_fallback_shapes_types_and_reference_lifetimes(self):
        class Row(dict):
            pass
        class Number(float):
            pass
        source = fixtures.knots([1.,2.])
        variants = [tuple(source),[Row(source[0]),source[1]]]
        numbers = copy.deepcopy(source)
        numbers[0]['values'][0] = Number(1.)
        variants.append(numbers)
        identity = copy.deepcopy(source)
        value = float('nan')
        for row in identity:
            row['epoch'] = value
        variants.append(identity)
        for rows in variants:
            self.worker._pack(rows,self.worker.cue)
            with mock.patch.object(self.worker,'pack_call',return_value=1):
                self.worker._pack(rows,self.worker.reference)
            end = len(rows)*ct.sizeof(native.Knot)
            self.assertEqual(bytes(self.worker.cue)[:end],bytes(self.worker.reference)[:end])
        with self.assertRaises(ValueError):
            self.worker.validate_and_pack(identity,self.worker.cue,10.)
        gc.collect()
        before = sys.getrefcount(Row),sys.getrefcount(Number)
        for _ in range(300):
            self.worker._pack(variants[1],self.worker.cue)
            self.worker._pack(numbers,self.worker.cue)
        gc.collect()
        self.assertEqual((sys.getrefcount(Row),sys.getrefcount(Number)),before)
        row = Row(source[0]); handle = weakref.ref(row)
        self.worker._pack([row],self.worker.cue)
        del row
        gc.collect()
        self.assertIsNone(handle())

    def test_numeric_library_build_has_no_python_binding_dependency(self):
        symbols = subprocess.check_output(['nm','-u',str(self.numeric_library)],text=True)
        self.assertNotRegex(symbols,r'\b_?Py\w+')
        library = ct.CDLL(str(self.numeric_library))
        with self.assertRaises(AttributeError):
            getattr(library,'temporal_pack_python')
        validate = library.temporal_validate
        validate.argtypes,validate.restype = self.worker.validate_call.argtypes,ct.c_int
        self.worker._pack(fixtures.knots([1.,2.]),self.worker.cue)
        self.assertEqual(validate(self.worker.cue,2,10.),0)

    def test_native_abi_mismatch_is_rejected_before_buffer_use(self):
        with self.assertRaises(AttributeError):
            getattr(self.worker.library,'temporal_layout')
        class ShortConfig(ct.Structure):
            _fields_ = [('n',ct.c_uint32)]
        with mock.patch.object(native,'Config',ShortConfig):
            with self.assertRaisesRegex(RuntimeError,'native ABI layout mismatch: ShortConfig'):
                native.NativeMatcher(self.library)
        self.assertEqual(self.worker.library.temporal_layout_v4(99),0)

    def test_winning_anchor_retains_original_power_sample_order(self):
        value = float.fromhex('0x1.e2cb19a44a7ecp+0')
        cue,refs = fixtures.knots([value,0.,0.,0.,0.]),fixtures.knots([0.]*8)
        self.worker._pack(cue,self.worker.cue)
        self.worker._pack(refs,self.worker.reference)
        self.worker._run_coarse(len(cue),len(refs),[1.]*10,limit=1)
        diag = self.worker.anchor_diagnostic
        self.assertEqual(diag.index,0)
        self.assertEqual(exact(list(diag.pitch_error[:5])),exact([value**2,0.,0.,0.,0.]))
        self.assertNotEqual(diag.pitch_error[0],value*value)
        # A successful no-winner call must invalidate the preceding diagnostics.
        missing = fixtures.knots([None]*3)
        self.worker._pack(missing,self.worker.cue)
        self.worker._run_coarse(3,len(refs),[1.]*10)
        self.assertEqual(diag.index,-1)
        self.assertEqual(diag.evaluated,2)

    def test_canonical_query_uses_native_anchor_diagnostics(self):
        q = fixtures.query([.1,.2,.3,.1,.2,.3,.1,.2])
        episodes = [fixtures.episode([0.,.1,.2,0.,.1,.2,0.,.1],identity=i) for i in range(20)]
        expected = reference.match_query(q,episodes,1000.)
        with mock.patch.object(reference,'anchor_transform',side_effect=AssertionError('Python anchor')):
            actual = self.worker.match_query(q,episodes,1000.)
        self.assertEqual(exact(actual),exact(expected))
        q['knots'][0]['values'][0] = 0
        expected = reference.match_query(q,episodes,1000.)
        with mock.patch.object(reference,'anchor_transform',wraps=reference.anchor_transform) as fallback:
            actual = self.worker.match_query(q,episodes,1000.)
        self.assertGreater(fallback.call_count,0)
        self.assertEqual(exact(actual),exact(expected))

    def test_query_batch_uses_three_crossings_and_one_copy_per_descriptor(self):
        q = fixtures.query([.1,.2,.3,.1,.2,.3,.1,.2])
        episodes = [fixtures.episode([0.,.1,.2,0.,.1,.2,0.,.1],identity=i) for i in range(20)]
        expected = reference.match_query(q,episodes,1000.)
        with mock.patch.object(self.worker,'pack_query_call',wraps=self.worker.pack_query_call) as pack, \
             mock.patch.object(self.worker,'coarse_query_call',wraps=self.worker.coarse_query_call) as coarse, \
             mock.patch.object(self.worker,'refine_query_call',wraps=self.worker.refine_query_call) as refine, \
             mock.patch.object(self.worker,'subsequence_dtw',side_effect=AssertionError('repacked DTW')), \
             mock.patch.object(reference,'refine_anchor',side_effect=AssertionError('Python refinement')):
            actual = self.worker.match_query(q,episodes,1000.)
        self.assertTrue(self.worker.last_query_batched)
        self.assertEqual((pack.call_count,coarse.call_count,refine.call_count),(1,1,1))
        self.assertEqual(self.worker.batch.copied_knots,21*8)
        self.assertEqual(exact(actual),exact(expected))

    def test_query_batch_owns_rows_scales_identity_and_original_cuts(self):
        q = fixtures.query([.1,.2,.3,.1],epoch=2**100,query_id='query-original',generation=2**99)
        episodes = [fixtures.episode([0.,.1,.2,0.],identity='episode-original',epoch=2**100,generation=2**101)]
        bounds = [2.,2.]
        expected = reference.match_query(q,episodes,1000.,bounds=bounds)
        call = self.worker.coarse_query_call
        def mutate_after_transfer(batch):
            q.update(epoch=0,generation=0,query_id='changed',available_end=9999.)
            q['scales'][0] = 999.
            q['knots'][0]['values'][0] = 999.
            q['knots'].clear()
            episodes[0].update(episode_id='changed',generation=0,available_end=9999.)
            episodes[0]['knots'].clear()
            episodes.clear()
            bounds[:] = [.01,.01]
            return call(batch)
        with mock.patch.object(self.worker,'coarse_query_call',side_effect=mutate_after_transfer):
            actual = self.worker.match_query(q,episodes,1000.,bounds=bounds)
        self.assertTrue(self.worker.last_query_batched)
        self.assertEqual(exact(actual),exact(expected))
        saved = exact(actual)
        self.worker.match_query(fixtures.query([0.]),[fixtures.episode([0.])],1000.)
        self.assertEqual(exact(actual),saved)

    def test_query_batch_probe_preserves_earlier_exception_before_later_malformed_input(self):
        q = fixtures.query([1e200,0.,0.,0.,0.])
        first = fixtures.episode([0.]*8,identity=1)
        later = fixtures.episode([0.],identity=2)
        del later['knots'][0]['time']
        for episodes in ([first,later],[later,first]):
            errors=[]
            for call in (reference.match_query,self.worker.match_query):
                with self.assertRaises((OverflowError,KeyError)) as caught:
                    call(q,episodes,1000.)
                errors.append((type(caught.exception),str(caught.exception)))
            self.assertEqual(errors[0],errors[1])
        self.assertFalse(self.worker.last_query_batched)

    def test_query_batch_skips_ineligible_rows_and_rechecks_cut_on_reuse(self):
        q = fixtures.query([.1,.2,.3])
        episodes = [fixtures.episode([0.,.1,.2],identity=i) for i in range(5)]
        for i,change in enumerate(({'epoch':2},{'committed':False},{'available_end':2000.},
                                    {'first_observed_end':q['query_end']})):
            episodes[i].update(change)
            del episodes[i]['knots']
            del episodes[i]['scales']
        expected = reference.match_query(q,episodes,1000.)
        actual = self.worker.match_query(q,episodes,1000.)
        self.assertTrue(self.worker.last_query_batched)
        self.assertEqual(self.worker.batch.n_episode,1)
        self.assertEqual(self.worker.batch.copied_knots,6)
        self.assertEqual(exact(actual),exact(expected))
        q['knots'][0]['available_end'] = 1001.
        for call in (reference.match_query,self.worker.match_query):
            with self.assertRaises(ValueError):
                call(q,episodes,1000.)

    def test_query_batch_extended_controls_use_full_reference_without_truncation(self):
        q = fixtures.query([0.,.1])
        episodes = [fixtures.episode([0.,.1],identity=i) for i in range(257)]
        for bank,options in ((episodes,{'max_episodes':257}),
                             (episodes[:20],{'candidate_limit':20})):
            expected = reference.match_query(q,bank,1000.,**options)
            actual = self.worker.match_query(q,bank,1000.,**options)
            self.assertEqual(exact(actual),exact(expected))
            self.assertFalse(self.worker.last_query_batched)

    def test_query_batch_native_boundaries_and_abi(self):
        q = fixtures.query([0.,.1])
        self.worker.match_query(q,[fixtures.episode([0.,.1])],1000.)
        self.assertTrue(self.worker.last_query_batched)
        b = self.worker.batch
        for function,field,value in ((self.worker.coarse_query_call,'n_episode',257),
                                     (self.worker.refine_query_call,'n_group',17),
                                     (self.worker.refine_query_call,'limit',0)):
            old = getattr(b,field)
            setattr(b,field,value)
            self.assertEqual(function(ct.byref(b)),1)
            setattr(b,field,old)
        for function in (self.worker.coarse_query_call,self.worker.refine_query_call):
            self.assertEqual(function(None),1)
        with self.assertRaises(AttributeError):
            getattr(self.worker.library,'temporal_layout_v2')
        for sources in ([],[[]]*258,[[{}]*129]):
            self.assertEqual(self.worker.pack_query_call(sources,native.PACK_KEYS,native.PACK_TYPES,ct.byref(b),1000.),1)
        expected = reference.match_query(q,[fixtures.episode([0.,.1])],1000.)
        actual = self.worker.match_query(q,[fixtures.episode([0.,.1])],1000.)
        self.assertTrue(self.worker.last_query_batched)
        self.assertEqual(exact(actual),exact(expected))

    def test_packed_projection_preserves_coverage_division_masks_and_f64_metadata(self):
        rng = random.Random(47128)
        for case in range(400):
            rows = fixtures.knots([[rng.uniform(-1.,1.) for _ in range(10)] for _ in range(3)],
                                  step=rng.uniform(.01,1.),generation=2**63+17)
            data = bytearray(pack_knots(rows))
            for i,row in enumerate(rows):
                duration = row['end']-row['start']
                for d in range(10):
                    weight = rng.choice((0.,duration, .9*duration, math.nextafter(.9*duration,0.),
                                         math.nextafter(.9*duration,math.inf)))
                    struct.pack_into('<d',data,i*320+d*24,weight)
                struct.pack_into('<d',data,i*320+8,-0.)
                struct.pack_into('<Q',data,i*320+304,2**64-1)
                struct.pack_into('<d',data,i*320+296,rng.choice((0.,duration/2,duration)))
            blobs = [bytes(data)]
            expected = list(native.PackedKnots(blobs[0]))
            self.worker._pack(expected,self.worker.cue)
            self.assertEqual(self.worker.packed_sources_call(blobs,native.PACKED_TYPES,
                              ct.byref(self.worker.packed_sources)),0)
            self.assertEqual(self.worker.unpack_query_call(ct.byref(self.worker.packed_sources),
                              ct.byref(self.worker.batch),1000.),0)
            self.assertEqual(bytes(self.worker.batch.knots[0])[:3*ct.sizeof(native.Knot)],
                             bytes(self.worker.cue)[:3*ct.sizeof(native.Knot)])
            self.worker.packed_sources.count = 0

    def test_packed_whole_queries_preserve_all_outputs_without_python_materialization(self):
        rng = random.Random(47129)
        for case in range(80):
            q = fixtures.query([[rng.uniform(-1.,1.) if rng.random()>.2 else None for _ in range(10)]
                                 for _ in range(rng.randint(1,8))])
            episodes = [fixtures.episode([[rng.uniform(-1.,1.) if rng.random()>.2 else None for _ in range(10)]
                         for _ in range(rng.randint(1,8))],identity=i) for i in range(rng.randint(1,5))]
            options = dict(band_radius=rng.choice((None,0,2,16)),transform_limit=rng.choice((1,2,4,8)),
                           anchor_spacing=rng.choice((1,2,4)),anchor_limit=rng.choice((1,4,32)),
                           fine_grid=rng.choice((1/128,1/64,1/32)),bounds=(2.,2.))
            expected = reference.match_query(q,episodes,1000.,**options)
            pq,pe = packed_record(q),[packed_record(e) for e in episodes]
            with mock.patch.object(native.PackedKnots,'__getitem__',side_effect=AssertionError('Python packed expansion')):
                actual = self.worker.match_query(pq,pe,1000.,**options)
            self.assertTrue(self.worker.last_query_batched)
            self.assertTrue(self.worker.last_query_packed)
            self.assertEqual(self.worker.packed_sources.count,0)
            self.assertEqual(exact(actual),exact(expected))

    def test_packed_projection_keeps_reference_lazy_rejection_order(self):
        q = packed_record(fixtures.query([1e200,0.,0.,0.,0.]))
        first = packed_record(fixtures.episode([0.]*8,identity=1))
        later = packed_record(fixtures.episode([0.],identity=2))
        blob = bytearray(later['knots']._data)
        start = struct.unpack_from('<d',blob,256)[0]
        struct.pack_into('<d',blob,264,start)
        later['knots'] = native.PackedKnots(bytes(blob))
        for episodes in ([first,later],[later,first]):
            errors=[]
            for function in (reference.match_query,self.worker.match_query):
                with self.assertRaises((OverflowError,ZeroDivisionError)) as caught:
                    function(q,episodes,1000.)
                errors.append((type(caught.exception),str(caught.exception)))
            self.assertEqual(errors[0],errors[1])
        self.assertEqual(self.worker.packed_sources.count,0)

    def test_packed_sources_require_immutable_exact_payload_and_valid_native_bounds(self):
        data = pack_knots(fixtures.knots([0.,.1]))
        for bad in (bytearray(data),memoryview(data),data[:-1]):
            with self.assertRaises(ValueError):native.PackedKnots(bad)
        source = self.worker.packed_sources
        for blobs in ([],[data]*258,[bytearray(data)],[data[:-1]],[bytes(129*320)]):
            self.assertEqual(self.worker.packed_sources_call(blobs,native.PACKED_TYPES,ct.byref(source)),1)
        self.assertEqual(self.worker.unpack_query_call(None,None,1000.),1)
        blobs=[data]
        self.assertEqual(self.worker.packed_sources_call(blobs,native.PACKED_TYPES,ct.byref(source)),0)
        before=bytes(self.worker.batch)
        for field,value in (('count',258),('count',0)):
            old=getattr(source,field);setattr(source,field,value)
            self.assertEqual(self.worker.unpack_query_call(ct.byref(source),ct.byref(self.worker.batch),1000.),1)
            self.assertEqual(bytes(self.worker.batch),before);setattr(source,field,old)
        source.length[0]=1
        self.assertEqual(self.worker.unpack_query_call(ct.byref(source),ct.byref(self.worker.batch),1000.),1)
        self.assertEqual(bytes(self.worker.batch),before)
        source.count=0
        with self.assertRaises(AttributeError):getattr(self.worker.library,'temporal_layout_v3')

    def test_packed_query_slot_and_real_bank_exports_are_detached_and_match_originals(self):
        sys.path.insert(0,str(ROOT/'tests'))
        import test_evaluate_temporal_bank_reference as bank_fixtures
        import test_evaluate_temporal_query_scheduler_reference as scheduler_fixtures
        import temporal_query_scheduler_reference as scheduler_reference
        bank = bank_fixtures.bank(2)
        handle=bank.reserve()
        write=bank_fixtures.event(1,.1,[(handle,handle,1.)])
        bank.apply(write,3.,bank_fixtures.admissions(write,[handle]))
        slot=scheduler_reference.QuerySlot(128)
        slot.capture(scheduler_fixtures.span(40),5.,91,92,93)
        q,episodes=slot.export(),bank.episodes(5.)
        pq,pe=slot.export(packed=True),bank.episodes(5.,packed=True)
        self.assertEqual(exact({**pq,'knots':list(pq['knots'])}),exact(q))
        self.assertEqual(exact([{**e,'knots':list(e['knots'])} for e in pe]),exact(episodes))
        expected=reference.match_query(q,episodes,5.)
        slot.storage[:]=bytes(len(slot.storage));bank.knots[:]=bytes(len(bank.knots))
        actual=self.worker.match_query(pq,pe,5.)
        self.assertTrue(self.worker.last_query_packed)
        self.assertEqual(exact(actual),exact(expected))
        with self.assertRaises(ValueError):bank.episodes(0.,packed=True)
        for data in (b'',pack_knots(fixtures.knots([None,None]))):
            pq['knots']=native.PackedKnots(data);pq['supporting_audio_end']=None
            expected=reference.match_query(pq,pe,5.)
            actual=self.worker.match_query(pq,pe,5.)
            self.assertTrue(self.worker.last_query_packed)
            self.assertEqual(exact(actual),exact(expected))

    def test_traceback_pow_rounding_and_wide_magnitudes(self):
        rng = random.Random(47101)
        counterexample = float.fromhex('-0x1.e2cb19a44a7ecp+0')
        self.assertNotEqual(counterexample*counterexample,counterexample**2)
        values = [counterexample,0.,-0.] + [math.ldexp(rng.uniform(-2.,2.),rng.randint(-520,480))
                                          for _ in range(320)]
        for value in values:
            result = self.compare(fixtures.knots([0.,value]),fixtures.knots([0.,0.]),
                                  scales=[max(abs(value),1e-150)]+[1.]*9,
                                  band_radius=None,insertion_penalty=4.,deletion_penalty=4.)
            self.assertEqual(result['relative_pitch_motion_pairs'],1)
            self.assertEqual(exact(self.worker.output.motion_error[0]),exact(value**2))

    def test_traceback_extreme_diagnostics_preserve_reference_exceptions(self):
        cue,refs = fixtures.knots([0.,1e200]),fixtures.knots([0.,0.])
        transform,scales = fixtures.fixed_transform(),[1e200]+[1.]*9
        for call in (reference.subsequence_dtw,self.worker.subsequence_dtw):
            with self.assertRaises(OverflowError):
                call(cue,refs,transform,scales,10.,band_radius=None,
                     insertion_penalty=4.,deletion_penalty=4.)
        self.assertEqual(self.worker.output.motion_count,0)
        self.compare(fixtures.knots([-1e308,1e308]),fixtures.knots([-1e308,1e308]),
                     scales=[1e308]+[1.]*9,band_radius=None)
        # A normal call following fallback must not expose stale diagnostics.
        result = self.compare(fixtures.knots([0.]),fixtures.knots([0.]))
        self.assertEqual(result['relative_pitch_motion_pairs'],0)
        self.assertEqual(result['matched_interval_pairs'],0)

    def test_traceback_gap_truth_is_lazy_and_noncanonical_types_fall_back(self):
        class BadGap:
            def __bool__(self):
                raise RuntimeError('gap evaluated')
        cue,refs = fixtures.knots([0.,0.]),fixtures.knots([0.,0.])
        cue[0]['gap'] = True
        cue[1]['gap'] = BadGap()
        expected = reference.subsequence_dtw(cue,refs,fixtures.fixed_transform(),[1.]*10,10.,band_radius=None)
        actual = self.worker.subsequence_dtw(cue,refs,fixtures.fixed_transform(),[1.]*10,10.,band_radius=None)
        self.assertEqual(exact(actual),exact(expected))
        cue[0]['gap'] = False
        for call in (reference.subsequence_dtw,self.worker.subsequence_dtw):
            with self.assertRaisesRegex(RuntimeError,'gap evaluated'):
                call(cue,refs,fixtures.fixed_transform(),[1.]*10,10.,band_radius=None)
        del cue[1]['gap']
        for call in (reference.subsequence_dtw,self.worker.subsequence_dtw):
            with self.assertRaises(KeyError):
                call(cue,refs,fixtures.fixed_transform(),[1.]*10,10.,band_radius=None)
        del cue[0]['gap']
        self.compare(cue[:1],refs[:1])
        cue = fixtures.knots([0.,1.])
        cue[0]['values'][0] = 0
        expected = reference.subsequence_dtw(cue,refs,fixtures.fixed_transform(),[1.]*10,10.)
        with mock.patch.object(reference,'subsequence_dtw',wraps=reference.subsequence_dtw) as fallback:
            actual = self.worker.subsequence_dtw(cue,refs,fixtures.fixed_transform(),[1.]*10,10.)
        fallback.assert_called_once()
        self.assertEqual(exact(actual),exact(expected))


if __name__ == '__main__':
    unittest.main()
