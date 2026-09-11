"""M0 native coarse/DTW experiment with the unchanged Python matcher as oracle.

Canonical Python objects use a GIL-held native transfer adapter; other shapes
retain Python handling. Full-query timing includes transfer and diagnostics.
This is not a production worker.
"""

import ctypes as ct
import importlib.util
import math
from pathlib import Path
import struct

import temporal_matcher_reference as reference
from temporal_descriptor_reference import PackedKnots

PACK_KEYS = ('values','time','start','end','observed_sec','raw_support_start',
             'raw_support_end','available_end','gap','epoch','generation')
PACK_TYPES = (float,list,dict,bool,None,int,str)
PACKED_TYPES = (bytes,list)


class Knot(ct.Structure):
    _fields_ = [('values', ct.c_double*10), ('time', ct.c_double), ('mask', ct.c_uint64),
                ('timing', ct.c_uint64), ('start', ct.c_double), ('end', ct.c_double),
                ('observed_sec', ct.c_double), ('raw_start', ct.c_double), ('raw_end', ct.c_double),
                ('available_end', ct.c_double), ('lineage_changed', ct.c_uint64), ('gap', ct.c_uint64)]


class CoarseConfig(ct.Structure):
    _fields_ = [('n', ct.c_uint32), ('m', ct.c_uint32), ('spacing', ct.c_uint32),
                ('limit', ct.c_uint32), ('bounds', ct.c_double*2), ('grid', ct.c_double),
                ('scales', ct.c_double*10)]


class Anchor(ct.Structure):
    _fields_ = [('cost', ct.c_double), ('unrounded', ct.c_double*2), ('applied', ct.c_double*2),
                ('valid', ct.c_uint32), ('pairs', ct.c_uint32), ('comparisons', ct.c_uint32),
                ('pitch_samples', ct.c_uint32), ('interval_samples', ct.c_uint32),
                ('out_of_range', ct.c_uint32), ('bound_hit', ct.c_uint32)]


class AnchorDiagnostic(ct.Structure):
    _fields_ = [('index',ct.c_int32), ('evaluated',ct.c_uint32),
                ('comparisons',ct.c_uint32), ('bound_anchors',ct.c_uint32),
                ('pitch_error',ct.c_double*8), ('interval_error',ct.c_double*8)]


class Config(ct.Structure):
    _fields_ = [('n', ct.c_uint32), ('m', ct.c_uint32), ('anchor', ct.c_uint32),
                ('band', ct.c_int32), ('shift', ct.c_double), ('ratio', ct.c_double),
                ('tempo_shift', ct.c_double), ('scales', ct.c_double*10),
                ('insertion', ct.c_double), ('deletion', ct.c_double)]


class Output(ct.Structure):
    _fields_ = [('total', ct.c_double), ('endpoint', ct.c_uint32),
                ('cells', ct.c_uint32), ('time_comparisons', ct.c_uint32), ('width', ct.c_uint32),
                ('starts', ct.c_uint16*128), ('ends', ct.c_uint16*128), ('parents', ct.c_uint8*16384),
                ('coordinate_error', ct.c_double*10), ('coordinate_count', ct.c_uint32*10),
                ('path', ct.c_int16*768), ('path_len', ct.c_uint32), ('reference_start', ct.c_uint32),
                ('observed', ct.c_uint32), ('matched', ct.c_uint32), ('missing', ct.c_uint32),
                ('inserted', ct.c_uint32), ('deleted', ct.c_uint32), ('valid_coordinates', ct.c_uint32),
                ('band_edge', ct.c_uint32), ('motion_count', ct.c_uint32), ('interval_count', ct.c_uint32),
                ('motion_error', ct.c_double*128), ('interval_error', ct.c_double*128)]


class QueryBatch(ct.Structure):
    _fields_ = [('knots', (Knot*128)*257), ('lengths', ct.c_uint32*257),
                ('best', Anchor*256), ('diagnostic', AnchorDiagnostic*256),
                ('scratch', Anchor*128), ('coarse', CoarseConfig),
                ('trials', Config*64), ('episodes', ct.c_uint32*64),
                ('groups', ct.c_uint32*17), ('selected', ct.c_uint32*64),
                ('outputs', Output*64), ('n_episode', ct.c_uint32),
                ('n_group', ct.c_uint32), ('n_selected', ct.c_uint32),
                ('limit', ct.c_uint32), ('copied_knots', ct.c_uint32)]


class PackedSources(ct.Structure):
    _fields_ = [('data', ct.c_void_p*257), ('length', ct.c_size_t*257), ('count', ct.c_uint32)]


class _ReferenceQuery(Exception):
    """The canonical bounded fast path did not publish a result."""


class NativeMatcher:
    """One single-caller worker scratch space; buffers never escape a call."""

    def __init__(self, library):
        self.library = ct.CDLL(str(Path(library).resolve()))
        layout = self.library.temporal_layout_v4
        layout.argtypes, layout.restype = [ct.c_uint32], ct.c_size_t
        for index, shape in enumerate((Knot, Config, Output, CoarseConfig, Anchor, AnchorDiagnostic, QueryBatch, PackedSources)):
            if layout(index) != ct.sizeof(shape):
                raise RuntimeError(f'native ABI layout mismatch: {shape.__name__}')
        self.call = self.library.temporal_dtw
        self.call.argtypes = [ct.POINTER(Knot), ct.POINTER(Knot), ct.POINTER(Config), ct.POINTER(Output)]
        self.call.restype = ct.c_int
        self.cue = (Knot*128)()
        self.reference = (Knot*128)()
        self.config = Config()
        self.output = Output()
        self.coarse_config = CoarseConfig()
        self.anchors = (Anchor*128)()
        self.anchor_diagnostic = AnchorDiagnostic()
        self.coarse_call = self.library.temporal_coarse
        self.coarse_call.argtypes = [ct.POINTER(Knot), ct.POINTER(Knot), ct.POINTER(CoarseConfig), ct.POINTER(Anchor), ct.POINTER(AnchorDiagnostic)]
        self.coarse_call.restype = ct.c_int
        self.validate_call = self.library.temporal_validate
        self.validate_call.argtypes = [ct.POINTER(Knot), ct.c_uint32, ct.c_double]
        self.validate_call.restype = ct.c_int
        self.python_library = ct.PyDLL(str(Path(library).resolve()))
        self.pack_call = self.python_library.temporal_pack_python
        self.pack_call.argtypes = [ct.py_object, ct.py_object, ct.py_object, ct.POINTER(Knot), ct.c_uint32]
        self.pack_call.restype = ct.c_int
        self.batch = QueryBatch()
        self.pack_query_call = self.python_library.temporal_pack_query_python
        self.pack_query_call.argtypes = [ct.py_object, ct.py_object, ct.py_object, ct.POINTER(QueryBatch), ct.c_double]
        self.pack_query_call.restype = ct.c_int
        self.coarse_query_call = self.library.temporal_coarse_query
        self.refine_query_call = self.library.temporal_refine_query
        for function in (self.coarse_query_call, self.refine_query_call):
            function.argtypes, function.restype = [ct.POINTER(QueryBatch)], ct.c_int
        self.packed_sources = PackedSources()
        self.packed_sources_call = self.python_library.temporal_packed_sources_python
        self.packed_sources_call.argtypes = [ct.py_object, ct.py_object, ct.POINTER(PackedSources)]
        self.packed_sources_call.restype = ct.c_int
        self.unpack_query_call = self.library.temporal_unpack_query
        self.unpack_query_call.argtypes = [ct.POINTER(PackedSources), ct.POINTER(QueryBatch), ct.c_double]
        self.unpack_query_call.restype = ct.c_int
        self.last_query_batched = False
        self.last_query_packed = False
        self.knot_format = struct.Struct('<11d2Q6d2Q')
        assert self.knot_format.size == ct.sizeof(Knot)
        self.fixed_payload_bytes = (ct.sizeof(self.cue)+ct.sizeof(self.reference)
                                    + ct.sizeof(self.config)+ct.sizeof(self.output)
                                    + ct.sizeof(self.coarse_config)+ct.sizeof(self.anchors)+ct.sizeof(self.anchor_diagnostic)
                                    + ct.sizeof(self.batch)+ct.sizeof(self.packed_sources))
        # Expose the numerical experiment to the existing matcher tests.
        spec = importlib.util.spec_from_file_location('temporal_native_query_driver', reference.__file__)
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)
        self.driver.subsequence_dtw = self.subsequence_dtw
        self.driver.match_query = self.match_query

    def _pack(self, source, target, timing=True):
        if len(source) > 128:
            raise ValueError('native experiment capacity is 128 knots')
        if self.pack_call(source, PACK_KEYS, PACK_TYPES, target, timing) == 0:
            return True
        previous, canonical = None, True
        for i, knot in enumerate(source):
            if len(knot['values']) != 10:
                raise ValueError('ten-coordinate knots required')
            mask, values = 0, [0.]*10
            for j, value in enumerate(knot['values']):
                if value is not None:
                    mask |= 1 << j
                    values[j] = value
                    canonical = canonical and type(value) is float
            metadata = (knot['start'], knot['end'], knot['observed_sec'], knot['raw_support_start'],
                        knot['raw_support_end'], knot['available_end'])
            canonical = canonical and type(knot['time']) is float and all(type(x) is float for x in metadata)
            changed = bool(previous is not None and (knot['epoch'] != previous['epoch']
                                                      or knot['generation'] != previous['generation']))
            supported_timing = bool(timing and mask and knot['observed_sec'] > 0 and not knot['gap'])
            gap = knot.get('gap') if type(knot) is dict else None
            gap_flag = int(gap) if type(gap) is bool else 2
            self.knot_format.pack_into(target, i*self.knot_format.size, *values, knot['time'], mask,
                                       supported_timing, *metadata, changed, gap_flag)
            previous = knot
        return canonical

    def validate_and_pack(self, source, target, observed_end, max_knots=128, timing=True):
        if not math.isfinite(observed_end) or len(source) > max_knots:
            raise ValueError('finite observation cut and bounded descriptor lengths required')
        try:
            canonical = self._pack(source, target, timing)
        except (KeyError, TypeError, ValueError, OverflowError, struct.error):
            # Reproduce the reference's error precedence for malformed objects.
            reference._validate(source, observed_end, max_knots)
            raise
        if not canonical or type(observed_end) is not float:
            reference._validate(source, observed_end, max_knots)
        elif self.validate_call(target, len(source), observed_end) != 0:
            reference._validate(source, observed_end, max_knots)
            raise AssertionError('native validator rejected a valid f64 descriptor')
        return canonical

    def _run_coarse(self, n, m, scales, spacing=4, limit=32, bounds=(2., 2.), grid=1/16):
        if (len(bounds) != 2 or not all(math.isfinite(x) and x > 0 for x in (*bounds, grid))
                or not all(isinstance(x, int) and x > 0 for x in (spacing, limit))):
            raise ValueError('positive bounded anchor spacing, count, bounds and grid required')
        c = self.coarse_config
        c.n, c.m, c.spacing, c.limit = n, m, min(128, spacing), min(128, limit)
        c.scales[:], c.bounds[:], c.grid = scales, bounds, grid
        status = self.coarse_call(self.cue, self.reference, ct.byref(c), self.anchors, ct.byref(self.anchor_diagnostic))
        if status == 4:
            return None
        if status == 2:
            raise OverflowError('non-finite native transform rounding')
        if status != 0:
            raise ValueError('invalid native coarse input or logarithm')
        return min(limit, (m+spacing-1)//spacing)

    def match_query(self, query, episodes, observed_end, candidate_limit=16, anchor_spacing=4,
                    anchor_limit=32, transform_limit=4, band_radius=16, max_episodes=256,
                    bounds=(2., 2.), coarse_grid=1/16, fine_grid=1/64,
                    insertion_penalty=1., deletion_penalty=1., max_knots=128):
        options = dict(candidate_limit=candidate_limit, anchor_spacing=anchor_spacing,
                       anchor_limit=anchor_limit, transform_limit=transform_limit,
                       band_radius=band_radius, max_episodes=max_episodes, bounds=bounds,
                       coarse_grid=coarse_grid, fine_grid=fine_grid,
                       insertion_penalty=insertion_penalty, deletion_penalty=deletion_penalty,
                       max_knots=max_knots)
        self.last_query_batched = False
        self.last_query_packed = False
        try:
            result = self._match_query_batch(query, episodes, observed_end, **options)
        except (_ReferenceQuery, KeyError, TypeError, ValueError, OverflowError):
            # Probing later inputs must not replace an earlier reference error.
            # Worker inputs are detached and must stay immutable during this call.
            self.last_query_packed = False
            return reference.match_query(query, episodes, observed_end, **options)
        self.last_query_batched = True
        return result

    def _match_query_batch(self, query, episodes, observed_end, candidate_limit,
                           anchor_spacing, anchor_limit, transform_limit, band_radius,
                           max_episodes, bounds, coarse_grid, fine_grid,
                           insertion_penalty, deletion_penalty, max_knots):
        if (type(query) is not dict or type(episodes) is not list
                or type(observed_end) is not float or not math.isfinite(observed_end)
                or not all(type(x) is int and x > 0 for x in
                    (candidate_limit, anchor_spacing, anchor_limit, transform_limit, max_episodes, max_knots))
                or len(episodes) > min(256, max_episodes)
                or type(bounds) not in (list, tuple) or len(bounds) != 2
                or not all(type(x) is float and math.isfinite(x) and x > 0
                           for x in (*bounds, coarse_grid, fine_grid, insertion_penalty, deletion_penalty))
                or band_radius is not None and (type(band_radius) is not int or band_radius < 0)):
            raise _ReferenceQuery
        bounds = tuple(bounds)
        # Copy only immutable identity/cut fields. The scheduler already owns a
        # detached input; this is not a concurrent live-bank snapshot operation.
        header = {key:query[key] for key in ('query_id','occurrence_id','support_id','epoch','generation',
                                           'query_end','supporting_audio_end','available_end','complete','superseded')}
        if (any(type(header[k]) not in (int,str) for k in
                ('query_id','occurrence_id','support_id','epoch','generation'))
                or any(type(header[k]) is not float for k in ('query_end','available_end'))
                or header['supporting_audio_end'] is not None and type(header['supporting_audio_end']) is not float
                or header['complete'] is not True or header['superseded'] is not False
                or header['available_end'] > observed_end or header['query_end'] > observed_end):
            raise _ReferenceQuery
        if type(query['scales']) not in (list,tuple):
            raise _ReferenceQuery
        scales = tuple(query['scales'])
        if len(scales) != 10 or not all(type(x) is float and math.isfinite(x) and x > 0 for x in scales):
            raise _ReferenceQuery
        sources = [query['knots']]
        inventory, seen = [], set()
        for episode in episodes:
            if type(episode) is not dict:
                raise _ReferenceQuery
            identity, generation = episode['episode_id'], episode['generation']
            if type(identity) not in (int,str) or type(generation) not in (int,str) or identity in seen:
                raise _ReferenceQuery
            seen.add(identity)
            if type(episode['epoch']) not in (int,str) or type(episode['committed']) is not bool:
                raise _ReferenceQuery
            if episode['epoch'] != header['epoch'] or not episode['committed']:
                continue
            if type(episode['available_end']) is not float or type(episode['first_observed_end']) is not float:
                raise _ReferenceQuery
            if episode['available_end'] > observed_end or episode['first_observed_end'] >= header['query_end']:
                continue
            if (type(episode['scales']) is not type(query['scales'])
                    or len(episode['scales']) != 10
                    or any(type(x) is not float for x in episode['scales'])
                    or tuple(episode['scales']) != scales):
                raise _ReferenceQuery
            sources.append(episode['knots'])
            inventory.append({'episode_id':identity, 'generation':generation})
        packed = all(type(rows) is PackedKnots for rows in sources)
        if any((not packed and type(rows) is not list) or len(rows) > min(128,max_knots) for rows in sources):
            raise _ReferenceQuery
        b = self.batch
        if packed:
            # These local strong references outlive the GIL-released projection.
            blobs = [rows._data for rows in sources]
            try:
                if (self.packed_sources_call(blobs, PACKED_TYPES, ct.byref(self.packed_sources)) != 0
                        or self.unpack_query_call(ct.byref(self.packed_sources),ct.byref(b),observed_end) != 0):
                    raise _ReferenceQuery
            finally:
                # Invalidate borrowed addresses before their owners can go away.
                self.packed_sources.count = 0
            del blobs
            self.last_query_packed = True
        elif self.pack_query_call(sources, PACK_KEYS, PACK_TYPES, ct.byref(b), observed_end) != 0:
            raise _ReferenceQuery
        # Neither the native stages nor result construction read source rows again.
        del sources
        query = header
        n = b.lengths[0]
        c = b.coarse
        c.spacing, c.limit = min(128,anchor_spacing), min(128,anchor_limit)
        c.bounds[:], c.grid, c.scales[:] = bounds, coarse_grid, scales
        if self.coarse_query_call(ct.byref(b)) != 0:
            raise _ReferenceQuery
        cache, candidates, anchor_evaluations, comparisons = [], [], 0, 0
        for index, episode in enumerate(inventory):
            identity, generation = episode['episode_id'], episode['generation']
            diag = b.diagnostic[index]
            anchor_evaluations += diag.evaluated
            comparisons += diag.comparisons
            bound_anchors = diag.bound_anchors
            anchor_total = (b.lengths[index+1]+anchor_spacing-1)//anchor_spacing
            best = None
            if diag.index >= 0:
                row = b.best[index]
                anchor = diag.index*anchor_spacing
                lengths = (row.pitch_samples,row.interval_samples)
                estimates = [row.unrounded[i] if lengths[i] else None for i in range(2)]
                applied = list(row.applied)
                transform = {'frequency_shift_log2':applied[0] if lengths[0] else None,
                             'tempo_shift_log2':applied[1] if lengths[1] else None,
                             'unrounded':estimates,'applied':applied,
                             'pitch_samples':lengths[0],'interval_samples':lengths[1],
                             'pitch_estimate_residual_rms':math.sqrt(math.fsum(diag.pitch_error[:lengths[0]])/lengths[0]) if lengths[0] else None,
                             'tempo_estimate_residual_rms':math.sqrt(math.fsum(diag.interval_error[:lengths[1]])/lengths[1]) if lengths[1] else None,
                             'median_comparisons':row.comparisons,'out_of_range':bool(row.out_of_range),
                             'bound_hit':bool(row.bound_hit),'unidentified':[x is None for x in estimates],
                             'anchor':anchor,'grid_id':[reference._round_lower(x/coarse_grid,1.) for x in applied]}
                best = {'cost':row.cost,'anchor':anchor,'valid_coordinates':row.valid,'transform':transform}
            cache.append({'episode_id': identity, 'episode_generation': generation,
                          'cost': None if best is None else best['cost'],
                          'similarity': None if best is None else math.exp(-best['cost']),
                          'anchor': None if best is None else best['anchor'],
                          'unidentified': [True, True] if best is None else best['transform']['unidentified'],
                          'valid_coordinates': 0 if best is None else best['valid_coordinates'],
                          'bound_anchors': bound_anchors, 'anchor_cap_loss': anchor_total > anchor_limit})
            if best is not None:
                candidates.append({**best, 'episode': episode, 'index':index})
        candidates.sort(key=lambda c: (c['cost'], c['episode']['episode_id']))
        selected, excluded = candidates[:candidate_limit], candidates[candidate_limit:]
        cutoff_tie = bool(excluded and selected and excluded[0]['cost'] == selected[-1]['cost'])
        if len(selected) > 16:
            raise _ReferenceQuery
        matches, refinements, dp_cells, trials = [], 0, 0, []
        b.n_group, b.limit = len(selected), min(4,transform_limit)
        for group, candidate in enumerate(selected):
            b.groups[group] = len(trials)
            coarse = candidate['transform']
            choices = [[0.] if x is None else sorted({math.floor(x/fine_grid)*fine_grid,
                                                       math.ceil(x/fine_grid)*fine_grid})
                       for x in coarse['unrounded']]
            refinements += max(1,math.prod(1 if x is None or x/fine_grid == math.floor(x/fine_grid) else 2
                                          for x in coarse['unrounded']))
            variants = []
            for pitch in choices[0]:
                for tempo in choices[1]:
                    if any(abs(x) >= bound for x,bound in zip((pitch,tempo),bounds)):
                        continue
                    transform = {**coarse, 'applied':[pitch,tempo],
                                 'frequency_shift_log2':None if coarse['unidentified'][0] else pitch,
                                 'tempo_shift_log2':None if coarse['unidentified'][1] else tempo,
                                 'grid_id':[round(pitch/fine_grid),round(tempo/fine_grid)], 'bound_hit':False}
                    variants.append(transform)
            # Stable native cost sorting then uses this exact secondary key.
            variants.sort(key=lambda t:t['grid_id'])
            for transform in variants:
                slot, index = len(trials), candidate['index']
                c = b.trials[slot]
                c.n, c.m, c.anchor = n, b.lengths[index+1], candidate['anchor']
                c.band = -1 if band_radius is None else min(128,band_radius)
                c.shift, c.ratio = transform['applied'][0], 2.**transform['applied'][1]
                c.tempo_shift = transform['applied'][1]
                c.scales[:], c.insertion, c.deletion = scales, insertion_penalty, deletion_penalty
                b.episodes[slot] = index
                trials.append((candidate,transform))
        b.groups[len(selected)] = len(trials)
        if self.refine_query_call(ct.byref(b)) != 0:
            raise _ReferenceQuery
        for slot in range(b.n_selected):
            candidate, transform = trials[b.selected[slot]]
            episode, index = candidate['episode'], candidate['index']
            result = self._format_output(b.outputs[slot], n, b.lengths[index+1], transform)
            dp_cells += result['dp_cells']
            matches.append({**result, 'episode_id': episode['episode_id'],
                            'episode_generation': episode['generation'], 'anchor': candidate['anchor'],
                            'transform_grid_id': transform['grid_id'], 'cutoff_tie': cutoff_tie,
                            'ambiguous_cutoff': cutoff_tie,
                            'search_covered': not cutoff_tie and not result['band_edge_hit']
                            and (b.lengths[index+1]+anchor_spacing-1)//anchor_spacing <= anchor_limit,
                            'search_nonempty': bool(cache)})
        matches.sort(key=lambda c: (c['cost'] if c['cost'] is not None else math.inf,
                                   c['episode_id'], c['transform_grid_id']))
        return {'query_id': query['query_id'], 'occurrence_id': query['occurrence_id'],
                'support_id': query['support_id'], 'epoch': query['epoch'], 'generation': query['generation'],
                'query_end': query['query_end'], 'supporting_audio_end': query['supporting_audio_end'],
                'available_end': query['available_end'], 'completed_at': observed_end,
                'complete': True, 'superseded': False, 'coarse_entries': cache, 'matches': matches,
                'excluded_episode_ids': [c['episode']['episode_id'] for c in excluded],
                'cutoff_tie': cutoff_tie, 'search_nonempty': bool(cache),
                'status': 'completed' if matches else 'unknown_unretrieved',
                'anchor_evaluations': anchor_evaluations, 'median_comparisons': comparisons,
                'refinement_evaluations': refinements, 'dp_cells': dp_cells,
                'limitations': ['single best anchor', 'density/tempo approximation',
                                'unfitted scales', 'no bank lifecycle or queue scheduling']}

    def subsequence_dtw(self, cue, refs, transform, scales, observed_end,
                        band_radius=16, insertion_penalty=1., deletion_penalty=1., max_knots=128):
        canonical_cue = self.validate_and_pack(cue, self.cue, observed_end, max_knots, timing=False)
        canonical_reference = self.validate_and_pack(refs, self.reference, observed_end, max_knots, timing=False)
        if (len(scales) != 10 or not all(math.isfinite(x) and x > 0 for x in scales)
                or not all(math.isfinite(x) and x > 0 for x in (insertion_penalty, deletion_penalty))
                or band_radius is not None and (not isinstance(band_radius, int) or band_radius < 0)):
            raise ValueError('ten positive scales, penalties and nonnegative band radius required')
        n, m = len(cue), len(refs)
        empty = {'status': 'unknown', 'supported': False, 'cost': None, 'path': [],
                 'dp_cells': 0, 'band_edge_hit': False, 'missing_pair_steps': 0}
        if not n or not m or transform['out_of_range'] or transform['bound_hit']:
            return empty
        anchor = transform['anchor']
        if not 0 <= anchor < m:
            raise ValueError('transform anchor must belong to this reference')
        if n > 128 or m > 128:
            raise ValueError('native experiment capacity is 128 knots')
        c = self.config
        c.n, c.m, c.anchor = n, m, anchor
        c.band = -1 if band_radius is None else min(128, band_radius)
        c.shift, c.ratio = transform['applied'][0], 2.**transform['applied'][1]
        c.tempo_shift = transform['applied'][1]
        c.scales[:] = scales
        c.insertion, c.deletion = insertion_penalty, deletion_penalty
        status = self.call(self.cue, self.reference, ct.byref(c), ct.byref(self.output))
        if status == 4 or not canonical_cue or not canonical_reference:
            return reference.subsequence_dtw(cue, refs, transform, scales, observed_end,
                        band_radius, insertion_penalty, deletion_penalty, max_knots)
        if status != 0:
            raise ValueError('invalid native bounded input')
        return self._format_output(self.output, n, m, transform)

    def _format_output(self, out, n, m, transform):
        empty = {'status': 'unknown', 'supported': False, 'cost': None, 'path': [],
                 'dp_cells': 0, 'band_edge_hit': False, 'missing_pair_steps': 0}
        ranges = [(out.starts[i], out.ends[i]) for i in range(n)]
        if not math.isfinite(out.total):
            return {**empty, 'dp_cells': out.cells, 'band_disconnected': True,
                    'band_ranges': ranges, 'band_time_comparisons': out.time_comparisons}
        path = [(('pair','insert','delete')[operation-1], i if i >= 0 else None, j if j >= 0 else None)
                for operation, i, j in struct.iter_unpack('<hhh', memoryview(out.path).cast('B')[:out.path_len*6])]
        coordinate_error, coordinate_count = list(out.coordinate_error), list(out.coordinate_count)
        motion_error = out.motion_error[:out.motion_count]
        interval_error = out.interval_error[:out.interval_count]
        observed = out.observed
        supported = observed > 0 and out.valid_coordinates > 0
        return {'status': 'match' if supported else 'unknown', 'supported': supported,
                'cost': out.total/observed if supported else None, 'total_cost': out.total,
                'observed_cue_steps': observed, 'matched_cue_steps': out.matched,
                'matched_step_fraction': out.matched/observed if observed else 0.,
                'missing_pair_steps': out.missing, 'inserted_steps': out.inserted, 'deleted_steps': out.deleted,
                'coordinate_residuals': [value/count if count else None
                                         for value, count in zip(coordinate_error, coordinate_count)],
                'coordinate_counts': coordinate_count, 'path': path,
                'relative_pitch_motion_rms': math.sqrt(math.fsum(motion_error)/len(motion_error)) if motion_error else None,
                'matched_interval_log2_residual_rms': math.sqrt(math.fsum(interval_error)/len(interval_error)) if interval_error else None,
                'relative_pitch_motion_pairs': len(motion_error), 'matched_interval_pairs': len(interval_error),
                'frequency_shift_log2': transform['frequency_shift_log2'],
                'tempo_shift_log2': transform['tempo_shift_log2'],
                'pitch_estimate_residual_rms': transform.get('pitch_estimate_residual_rms'),
                'tempo_estimate_residual_rms': transform.get('tempo_estimate_residual_rms'),
                'bound_hit': transform['bound_hit'], 'band_edge_hit': bool(out.band_edge),
                'dp_cells': out.cells, 'dp_payload_bytes': (m+1)*16+n*out.width+n*4+m*8,
                'band_ranges': ranges, 'band_time_comparisons': out.time_comparisons,
                'field_visits': out.cells*20, 'traceback_coordinate_visits': len(path)*10,
                'reference_start': out.reference_start, 'search_completed': True}
