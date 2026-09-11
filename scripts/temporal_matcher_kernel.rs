//! Standalone M0 numerical experiment; not linked into the instrument.
//! Build with rustc --edition 2021 --crate-type cdylib -O.
//! Enable python-binding only for the CPython transfer adapter.

#[cfg(feature = "python-binding")]
use std::ffi::c_void;
use std::slice;

const CAPACITY: usize = 128;

#[repr(C)]
pub struct Knot {
    pub values: [f64; 10],
    pub time: f64,
    pub mask: u64,
    pub timing: u64,
    pub start: f64,
    pub end: f64,
    pub observed_sec: f64,
    pub raw_start: f64,
    pub raw_end: f64,
    pub available_end: f64,
    pub lineage_changed: u64,
    pub gap: u64,
}

#[cfg(feature = "python-binding")]
extern "C" {
    fn PyObject_Type(value: *mut c_void) -> *mut c_void;
    fn Py_DecRef(value: *mut c_void);
    fn PyList_Size(value: *mut c_void) -> isize;
    fn PyList_GetItem(value: *mut c_void, index: isize) -> *mut c_void;
    fn PyBytes_Size(value: *mut c_void) -> isize;
    fn PyBytes_AsString(value: *mut c_void) -> *const u8;
    fn PyTuple_Size(value: *mut c_void) -> isize;
    fn PyTuple_GetItem(value: *mut c_void, index: isize) -> *mut c_void;
    fn PyDict_GetItem(value: *mut c_void, key: *mut c_void) -> *mut c_void;
    fn PyFloat_AsDouble(value: *mut c_void) -> f64;
    fn PyObject_IsTrue(value: *mut c_void) -> i32;
    fn PyObject_RichCompareBool(left: *mut c_void, right: *mut c_void, operation: i32) -> i32;
}

#[cfg(feature = "python-binding")]
unsafe fn exact_python_type(value: *mut c_void, expected: *mut c_void) -> bool {
    if value.is_null() {
        return false;
    }
    let actual = unsafe { PyObject_Type(value) };
    let equal = actual == expected;
    unsafe { Py_DecRef(actual) };
    equal
}

/// Copy exact Python floats and builtin containers to fixed native storage.
/// Return one to use the original Python packing path for other shapes/types.
/// No Python object is retained. Existing key objects avoid string allocation.
///
/// # Safety
/// The GIL must be held. Object pointers must be live; keys/types are tuples of
/// eleven schema keys/seven builtin types or sentinels. Output covers 128 Knots
/// and must not alias Python storage. Call through ctypes.PyDLL, never CDLL.
#[cfg(feature = "python-binding")]
#[no_mangle]
pub unsafe extern "C" fn temporal_pack_python(
    source: *mut c_void,
    keys: *mut c_void,
    types: *mut c_void,
    output: *mut Knot,
    timing: u32,
) -> i32 {
    if source.is_null()
        || keys.is_null()
        || types.is_null()
        || output.is_null()
        || unsafe { PyTuple_Size(keys) } != 11
        || unsafe { PyTuple_Size(types) } != 7
    {
        return 1;
    }
    let mut key = [std::ptr::null_mut(); 11];
    let mut expected = [std::ptr::null_mut(); 7];
    for (i, value) in key.iter_mut().enumerate() {
        *value = unsafe { PyTuple_GetItem(keys, i as isize) };
    }
    for (i, value) in expected.iter_mut().enumerate() {
        *value = unsafe { PyTuple_GetItem(types, i as isize) };
    }
    if !unsafe { exact_python_type(source, expected[1]) } {
        return 1;
    }
    let n = unsafe { PyList_Size(source) };
    if n < 0 || n as usize > CAPACITY {
        return 1;
    }
    let out = unsafe { slice::from_raw_parts_mut(output, n as usize) };
    let mut previous_epoch = std::ptr::null_mut();
    let mut previous_generation = std::ptr::null_mut();
    let mut previous_supported = false;
    for (i, knot) in out.iter_mut().enumerate() {
        let row = unsafe { PyList_GetItem(source, i as isize) };
        if !unsafe { exact_python_type(row, expected[2]) } {
            return 1;
        }
        let values = unsafe { PyDict_GetItem(row, key[0]) };
        if !unsafe { exact_python_type(values, expected[1]) }
            || unsafe { PyList_Size(values) } != 10
        {
            return 1;
        }
        knot.mask = 0;
        knot.values.fill(0.0);
        for j in 0..10 {
            let value = unsafe { PyList_GetItem(values, j as isize) };
            if value == expected[4] {
                continue;
            }
            if !unsafe { exact_python_type(value, expected[0]) } {
                return 1;
            }
            knot.mask |= 1 << j;
            knot.values[j] = unsafe { PyFloat_AsDouble(value) };
        }
        let mut metadata = [0.0; 7];
        for (j, value) in metadata.iter_mut().enumerate() {
            let object = unsafe { PyDict_GetItem(row, key[j + 1]) };
            if !unsafe { exact_python_type(object, expected[0]) } {
                return 1;
            }
            *value = unsafe { PyFloat_AsDouble(object) };
        }
        knot.time = metadata[0];
        knot.start = metadata[1];
        knot.end = metadata[2];
        knot.observed_sec = metadata[3];
        knot.raw_start = metadata[4];
        knot.raw_end = metadata[5];
        knot.available_end = metadata[6];
        let gap = unsafe { PyDict_GetItem(row, key[8]) };
        knot.gap = if unsafe { exact_python_type(gap, expected[3]) } {
            u64::from(unsafe { PyObject_IsTrue(gap) } != 0)
        } else {
            2
        };
        knot.timing = 0;
        if timing != 0 && knot.mask != 0 && knot.observed_sec > 0.0 {
            if knot.gap == 2 {
                return 1;
            }
            knot.timing = u64::from(knot.gap == 0);
        }
        let epoch = unsafe { PyDict_GetItem(row, key[9]) };
        let generation = unsafe { PyDict_GetItem(row, key[10]) };
        let supported = [epoch, generation].iter().all(|value| unsafe {
            exact_python_type(*value, expected[5]) || exact_python_type(*value, expected[6])
        });
        knot.lineage_changed = 0;
        if i > 0 {
            if !supported || !previous_supported {
                return 1;
            }
            // Restrict this shortcut to builtin int/str equality. Other identity
            // types retain Python's != semantics through the reference path.
            let epoch_changed = unsafe { PyObject_RichCompareBool(epoch, previous_epoch, 3) };
            let generation_changed = if epoch_changed == 0 {
                unsafe { PyObject_RichCompareBool(generation, previous_generation, 3) }
            } else {
                0
            };
            if epoch_changed < 0 || generation_changed < 0 {
                return 1;
            }
            knot.lineage_changed = u64::from(epoch_changed != 0 || generation_changed != 0);
        }
        previous_epoch = epoch;
        previous_generation = generation;
        previous_supported = supported;
    }
    0
}

/// Validate a packed f64 descriptor at its original observation cut.
/// Return zero on success and one for invalid input. No buffer is modified.
///
/// # Safety
/// The pointer must cover n aligned live Knots. The caller computes the opaque
/// epoch/generation equality flag without narrowing those identities to f64.
#[no_mangle]
pub unsafe extern "C" fn temporal_validate(knots: *const Knot, n: u32, observed_end: f64) -> i32 {
    if knots.is_null() || n as usize > CAPACITY || !observed_end.is_finite() {
        return 1;
    }
    let rows = unsafe { slice::from_raw_parts(knots, n as usize) };
    for (i, k) in rows.iter().enumerate() {
        if k.mask > 1023
            || k.values
                .iter()
                .enumerate()
                .any(|(j, value)| k.mask & (1 << j) != 0 && !value.is_finite())
            || ![
                k.start,
                k.end,
                k.time,
                k.observed_sec,
                k.raw_start,
                k.raw_end,
                k.available_end,
            ]
            .iter()
            .all(|value| value.is_finite())
            || !(k.raw_start <= k.start
                && k.start < k.end
                && k.end <= k.raw_end
                && k.raw_end <= k.available_end
                && k.available_end <= observed_end)
            || !(0.0 <= k.observed_sec && k.observed_sec <= k.end - k.start + 1e-12)
            || k.time > k.raw_end
            || (i > 0
                && (k.start != rows[i - 1].end
                    || k.time <= rows[i - 1].time
                    || k.lineage_changed != 0))
        {
            return 1;
        }
    }
    0
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Config {
    pub n: u32,
    pub m: u32,
    pub anchor: u32,
    pub band: i32,
    pub shift: f64,
    pub ratio: f64,
    pub tempo_shift: f64,
    pub scales: [f64; 10],
    pub insertion: f64,
    pub deletion: f64,
}

#[repr(C)]
pub struct Output {
    pub total: f64,
    pub endpoint: u32,
    pub cells: u32,
    pub time_comparisons: u32,
    pub width: u32,
    pub starts: [u16; CAPACITY],
    pub ends: [u16; CAPACITY],
    pub parents: [u8; CAPACITY * CAPACITY],
    pub coordinate_error: [f64; 10],
    pub coordinate_count: [u32; 10],
    pub path: [[i16; 3]; CAPACITY * 2],
    pub path_len: u32,
    pub reference_start: u32,
    pub observed: u32,
    pub matched: u32,
    pub missing: u32,
    pub inserted: u32,
    pub deleted: u32,
    pub valid_coordinates: u32,
    pub band_edge: u32,
    pub motion_count: u32,
    pub interval_count: u32,
    pub motion_error: [f64; CAPACITY],
    pub interval_error: [f64; CAPACITY],
}

#[repr(C)]
pub struct CoarseConfig {
    pub n: u32,
    pub m: u32,
    pub spacing: u32,
    pub limit: u32,
    pub bounds: [f64; 2],
    pub grid: f64,
    pub scales: [f64; 10],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Anchor {
    pub cost: f64,
    pub unrounded: [f64; 2],
    pub applied: [f64; 2],
    pub valid: u32,
    pub pairs: u32,
    pub comparisons: u32,
    pub pitch_samples: u32,
    pub interval_samples: u32,
    pub out_of_range: u32,
    pub bound_hit: u32,
}

#[repr(C)]
pub struct AnchorDiagnostic {
    pub index: i32,
    pub evaluated: u32,
    pub comparisons: u32,
    pub bound_anchors: u32,
    pub pitch_error: [f64; 8],
    pub interval_error: [f64; 8],
}

/// Check C ABI buffer sizes before dereferencing caller-owned storage.
#[no_mangle]
pub extern "C" fn temporal_layout_v4(kind: u32) -> usize {
    match kind {
        0 => std::mem::size_of::<Knot>(),
        1 => std::mem::size_of::<Config>(),
        2 => std::mem::size_of::<Output>(),
        3 => std::mem::size_of::<CoarseConfig>(),
        4 => std::mem::size_of::<Anchor>(),
        5 => std::mem::size_of::<AnchorDiagnostic>(),
        6 => std::mem::size_of::<QueryBatch>(),
        7 => std::mem::size_of::<PackedSources>(),
        _ => 0,
    }
}

const EPISODES: usize = 256;
const TRIALS: usize = 64;

#[repr(C)]
pub struct PackedSources {
    pub data: [*const u8; EPISODES + 1],
    pub length: [usize; EPISODES + 1],
    pub count: u32,
}

/// Borrow immutable Python bytes while the caller keeps the source list alive.
///
/// # Safety
/// The GIL is held. types is (bytes, list); sources and output are live. Pointers
/// are borrowed only until the caller finishes temporal_unpack_query. No Python
/// object reference is retained by this function or by the numeric decoder.
#[cfg(feature = "python-binding")]
#[no_mangle]
pub unsafe extern "C" fn temporal_packed_sources_python(
    sources: *mut c_void,
    types: *mut c_void,
    output: *mut PackedSources,
) -> i32 {
    if sources.is_null()
        || types.is_null()
        || output.is_null()
        || unsafe { PyTuple_Size(types) } != 2
    {
        return 1;
    }
    let bytes_type = unsafe { PyTuple_GetItem(types, 0) };
    let list_type = unsafe { PyTuple_GetItem(types, 1) };
    if !unsafe { exact_python_type(sources, list_type) } {
        return 1;
    }
    let n = unsafe { PyList_Size(sources) };
    if !(1..=EPISODES as isize + 1).contains(&n) {
        return 1;
    }
    let out = unsafe { &mut *output };
    out.count = 0;
    for i in 0..n as usize {
        let data = unsafe { PyList_GetItem(sources, i as isize) };
        if !unsafe { exact_python_type(data, bytes_type) } {
            return 1;
        }
        let length = unsafe { PyBytes_Size(data) };
        if length < 0 || length as usize > CAPACITY * 320 || length % 320 != 0 {
            return 1;
        }
        out.data[i] = unsafe { PyBytes_AsString(data) };
        out.length[i] = length as usize;
    }
    out.count = n as u32;
    0
}

/// Project the registered little-endian 320-byte moment payload into match knots.
/// Coverage uses the original division before its 0.9 threshold. Failed probes
/// defer to the unchanged Python sequence, including its lazy exception order.
///
/// # Safety
/// sources contains live immutable byte ranges until return. It and batch must
/// be aligned, disjoint and exclusively owned; byte payloads may be unaligned.
/// The caller may release the GIL because it owns references to immutable bytes.
#[no_mangle]
pub unsafe extern "C" fn temporal_unpack_query(
    sources: *const PackedSources,
    batch: *mut QueryBatch,
    observed_end: f64,
) -> i32 {
    if sources.is_null() || batch.is_null() {
        return 1;
    }
    let s = unsafe { &*sources };
    if s.count == 0 || s.count as usize > EPISODES + 1 || !observed_end.is_finite() {
        return 1;
    }
    for i in 0..s.count as usize {
        if s.data[i].is_null() || s.length[i] > CAPACITY * 320 || s.length[i] % 320 != 0 {
            return 1;
        }
    }
    let b = unsafe { &mut *batch };
    b.n_episode = 0;
    b.n_selected = 0;
    b.copied_knots = 0;
    for i in 0..s.count as usize {
        let data = unsafe { slice::from_raw_parts(s.data[i], s.length[i]) };
        let n = data.len() / 320;
        let mut previous_lineage = [0u64; 2];
        for (j, bytes) in data.chunks_exact(320).enumerate() {
            let number = |index: usize| {
                f64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap())
            };
            let k = &mut b.knots[i][j];
            k.start = number(32);
            k.end = number(33);
            let duration = k.end - k.start;
            if duration == 0.0 {
                return 4;
            }
            k.mask = 0;
            k.values.fill(0.0);
            for d in 0..10 {
                if number(3 * d) / duration >= 0.9 {
                    k.mask |= 1 << d;
                    k.values[d] = number(3 * d + 1);
                }
            }
            k.time = number(30);
            k.observed_sec = number(31);
            k.raw_start = number(34);
            k.raw_end = number(35);
            k.available_end = number(36);
            k.gap = u64::from(number(37) > 0.0);
            k.timing = u64::from(k.mask != 0 && k.observed_sec > 0.0 && k.gap == 0);
            let lineage = [
                u64::from_le_bytes(bytes[304..312].try_into().unwrap()),
                u64::from_le_bytes(bytes[312..320].try_into().unwrap()),
            ];
            k.lineage_changed = u64::from(j > 0 && lineage != previous_lineage);
            previous_lineage = lineage;
        }
        if unsafe { temporal_validate(b.knots[i].as_ptr(), n as u32, observed_end) } != 0 {
            return 4;
        }
        b.lengths[i] = n as u32;
        b.copied_knots += n as u32;
    }
    b.n_episode = s.count - 1;
    0
}

#[repr(C)]
pub struct QueryBatch {
    pub knots: [[Knot; CAPACITY]; EPISODES + 1],
    pub lengths: [u32; EPISODES + 1],
    pub best: [Anchor; EPISODES],
    pub diagnostic: [AnchorDiagnostic; EPISODES],
    pub scratch: [Anchor; CAPACITY],
    pub coarse: CoarseConfig,
    pub trials: [Config; TRIALS],
    pub episodes: [u32; TRIALS],
    pub groups: [u32; 17],
    pub selected: [u32; TRIALS],
    pub outputs: [Output; TRIALS],
    pub n_episode: u32,
    pub n_group: u32,
    pub n_selected: u32,
    pub limit: u32,
    pub copied_knots: u32,
}

/// Transfer a detached worker query exactly once, including original cut checks.
/// A failed probe publishes nothing; the caller reruns the Python oracle to
/// preserve lazy rejection order. No Python objects are retained.
///
/// # Safety
/// The GIL is held. sources is a list of cue plus at most 256 descriptor lists;
/// keys/types follow temporal_pack_python. batch owns disjoint live storage.
#[cfg(feature = "python-binding")]
#[no_mangle]
pub unsafe extern "C" fn temporal_pack_query_python(
    sources: *mut c_void,
    keys: *mut c_void,
    types: *mut c_void,
    batch: *mut QueryBatch,
    observed_end: f64,
) -> i32 {
    if sources.is_null()
        || keys.is_null()
        || types.is_null()
        || batch.is_null()
        || unsafe { PyTuple_Size(types) } != 7
    {
        return 1;
    }
    let list_type = unsafe { PyTuple_GetItem(types, 1) };
    if !unsafe { exact_python_type(sources, list_type) } {
        return 1;
    }
    let count = unsafe { PyList_Size(sources) };
    if !(1..=EPISODES as isize + 1).contains(&count) {
        return 1;
    }
    let b = unsafe { &mut *batch };
    b.n_episode = 0;
    b.n_selected = 0;
    b.copied_knots = 0;
    for i in 0..count as usize {
        let source = unsafe { PyList_GetItem(sources, i as isize) };
        if unsafe { temporal_pack_python(source, keys, types, b.knots[i].as_mut_ptr(), 1) } != 0 {
            return 1;
        }
        let n = unsafe { PyList_Size(source) } as u32;
        if unsafe { temporal_validate(b.knots[i].as_ptr(), n, observed_end) } != 0 {
            return 1;
        }
        b.lengths[i] = n;
        b.copied_knots += n;
    }
    b.n_episode = count as u32 - 1;
    0
}

/// Run all coarse searches over owned rows without Python crossings.
///
/// # Safety
/// batch covers one aligned, exclusively owned QueryBatch. Failed calls expose
/// no complete result. Only diagnostics with nonnegative index have a best row.
#[no_mangle]
pub unsafe extern "C" fn temporal_coarse_query(batch: *mut QueryBatch) -> i32 {
    if batch.is_null() {
        return 1;
    }
    let b = unsafe { &mut *batch };
    if b.n_episode as usize > EPISODES || b.lengths[0] as usize > CAPACITY {
        return 1;
    }
    b.coarse.n = b.lengths[0];
    for i in 0..b.n_episode as usize {
        b.coarse.m = b.lengths[i + 1];
        let status = unsafe {
            temporal_coarse(
                b.knots[0].as_ptr(),
                b.knots[i + 1].as_ptr(),
                &b.coarse,
                b.scratch.as_mut_ptr(),
                &mut b.diagnostic[i],
            )
        };
        if status != 0 {
            return status;
        }
        let index = b.diagnostic[i].index;
        if index >= 0 {
            b.best[i] = b.scratch[index as usize];
        }
    }
    0
}

/// Rank the at-most-four bracketing trials per candidate, then run bounded DTW.
/// Trial order is Python's grid-ID order; stable cost sorting preserves its ties.
/// Nonfinite coarse costs defer to Python's exact sorting/exception semantics.
///
/// # Safety
/// batch covers exclusively owned storage. Groups partition at most 64 trials;
/// each trial refers to a packed episode. Outputs are valid only on success.
#[no_mangle]
pub unsafe extern "C" fn temporal_refine_query(batch: *mut QueryBatch) -> i32 {
    if batch.is_null() {
        return 1;
    }
    let b = unsafe { &mut *batch };
    if b.n_episode as usize > EPISODES
        || b.n_group > 16
        || b.limit == 0
        || b.limit > 4
        || b.groups[0] != 0
        || b.groups[b.n_group as usize] as usize > TRIALS
    {
        return 1;
    }
    for group in b.groups[..=b.n_group as usize].windows(2) {
        if group[0] > group[1] || group[1] - group[0] > 4 {
            return 1;
        }
    }
    for i in 0..b.groups[b.n_group as usize] as usize {
        let e = b.episodes[i] as usize;
        let c = &b.trials[i];
        if e >= b.n_episode as usize
            || c.n != b.lengths[0]
            || c.m != b.lengths[e + 1]
            || c.n == 0
            || c.n as usize > CAPACITY
            || c.m as usize > CAPACITY
            || c.anchor >= c.m
            || !c.shift.is_finite()
            || c.scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return 1;
        }
    }
    b.n_selected = 0;
    for g in 0..b.n_group as usize {
        let mut order = [0usize; 4];
        let mut costs = [0.0; 4];
        let mut count = 0;
        for i in b.groups[g] as usize..b.groups[g + 1] as usize {
            let c = &b.trials[i];
            let a = &b.knots[0];
            let r = &b.knots[b.episodes[i] as usize + 1];
            let mut total = 0.0;
            let mut valid = 0;
            for j in 0..8.min(c.n as usize).min((c.m - c.anchor) as usize) {
                let k = c.anchor as usize + j;
                let mask = a[j].mask & r[k].mask;
                let mut residual = 0.0;
                for d in 0..10 {
                    if mask & (1 << d) != 0 {
                        let shift = if d == 0 { c.shift } else { 0.0 };
                        let delta = (a[j].values[d] - r[k].values[d] - shift) / c.scales[d];
                        residual += delta * delta;
                        valid += 1;
                    }
                }
                total += residual;
            }
            if valid == 0 {
                continue;
            }
            let cost = total / valid as f64;
            if !cost.is_finite() {
                return 4;
            }
            let mut index = count;
            while index > 0 && cost < costs[index - 1] {
                costs[index] = costs[index - 1];
                order[index] = order[index - 1];
                index -= 1;
            }
            costs[index] = cost;
            order[index] = i;
            count += 1;
        }
        for &i in order.iter().take(count.min(b.limit as usize)) {
            let slot = b.n_selected as usize;
            let status = unsafe {
                temporal_dtw(
                    b.knots[0].as_ptr(),
                    b.knots[b.episodes[i] as usize + 1].as_ptr(),
                    &b.trials[i],
                    &mut b.outputs[slot],
                )
            };
            if status != 0 {
                return status;
            }
            b.selected[slot] = i as u32;
            b.n_selected += 1;
        }
    }
    0
}

/// Return zero on success, one for invalid input, two for rounding overflow,
/// three for an invalid logarithm, or four to defer extreme RMS arithmetic to
/// the reference. Only the declared anchor prefix is valid on success.
///
/// # Safety
/// Pointers must cover config.n/config.m aligned Knots, one CoarseConfig, and
/// 128 aligned Anchors and one AnchorDiagnostic. Outputs must not alias inputs
/// or each other. The caller owns the buffers.
#[no_mangle]
pub unsafe extern "C" fn temporal_coarse(
    cue: *const Knot,
    reference: *const Knot,
    config: *const CoarseConfig,
    output: *mut Anchor,
    diagnostic: *mut AnchorDiagnostic,
) -> i32 {
    if cue.is_null()
        || reference.is_null()
        || config.is_null()
        || output.is_null()
        || diagnostic.is_null()
    {
        return 1;
    }
    let c = unsafe { &*config };
    let (n, m, spacing, limit) = (
        c.n as usize,
        c.m as usize,
        c.spacing as usize,
        c.limit as usize,
    );
    if n > CAPACITY
        || m > CAPACITY
        || spacing == 0
        || spacing > CAPACITY
        || limit == 0
        || limit > CAPACITY
        || !c.grid.is_finite()
        || c.grid <= 0.0
        || c.bounds
            .iter()
            .chain(c.scales.iter())
            .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return 1;
    }
    let a = unsafe { slice::from_raw_parts(cue, n) };
    let b = unsafe { slice::from_raw_parts(reference, m) };
    for sequence in [a, b] {
        for (i, knot) in sequence.iter().enumerate() {
            if knot.mask > 1023
                || knot.timing > 1
                || !knot.time.is_finite()
                || (i > 0 && knot.time <= sequence[i - 1].time)
                || knot
                    .values
                    .iter()
                    .enumerate()
                    .any(|(j, value)| knot.mask & (1 << j) != 0 && !value.is_finite())
            {
                return 1;
            }
        }
    }
    let out = unsafe { slice::from_raw_parts_mut(output, CAPACITY) };
    let diag = unsafe { &mut *diagnostic };
    diag.index = -1;
    diag.evaluated = 0;
    diag.comparisons = 0;
    diag.bound_anchors = 0;
    let mut best_cost = f64::INFINITY;
    let mut best_samples = [[0.0; 8]; 2];
    for (index, anchor) in (0..m).step_by(spacing).take(limit).enumerate() {
        let steps = 8.min(n).min(m - anchor);
        let mut samples = [[0.0; 8]; 2];
        let mut lengths = [0; 2];
        let mut previous_timing = false;
        for i in 0..steps {
            let (left, right) = (&a[i], &b[anchor + i]);
            if left.mask & right.mask & 1 != 0 {
                samples[0][lengths[0]] = left.values[0] - right.values[0];
                lengths[0] += 1;
            }
            let timing = left.timing != 0 && right.timing != 0;
            if i > 0 && timing && previous_timing {
                let ci = left.time - a[i - 1].time;
                let ri = right.time - b[anchor + i - 1].time;
                if ci > 0.0 && ri > 0.0 {
                    let ratio = ri / ci;
                    if ratio.is_nan() {
                        return 4;
                    }
                    if ratio <= 0.0 {
                        return 3;
                    }
                    samples[1][lengths[1]] = ratio.log2();
                    lengths[1] += 1;
                }
            }
            previous_timing = timing;
        }
        let mut row = Anchor {
            cost: f64::NAN,
            unrounded: [0.0; 2],
            applied: [0.0; 2],
            valid: 0,
            pairs: 0,
            comparisons: 0,
            pitch_samples: lengths[0] as u32,
            interval_samples: lengths[1] as u32,
            out_of_range: 0,
            bound_hit: 0,
        };
        for dimension in 0..2 {
            let count = lengths[dimension];
            if count == 0 {
                continue;
            }
            let mut values = samples[dimension];
            // Preserve the reference's bounded insertion-sort comparison count.
            for i in 0..count {
                let value = values[i];
                let mut position = i;
                while position > 0 {
                    row.comparisons += 1;
                    if values[position - 1] <= value {
                        break;
                    }
                    values[position] = values[position - 1];
                    position -= 1;
                }
                values[position] = value;
            }
            let value = if count % 2 == 1 {
                values[count / 2]
            } else {
                (values[count / 2 - 1] + values[count / 2]) / 2.0
            };
            // The reference computes every anchor's RMS even if it loses.
            // Preserve its overflow/inf behavior through an exact slow path.
            if values[..count]
                .iter()
                .any(|sample| (*sample - value).abs() > f64::MAX.sqrt() / 8.0)
            {
                return 4;
            }
            let quotient = value / c.grid;
            if !quotient.is_finite() {
                return if quotient.is_nan() { 3 } else { 2 };
            }
            // Python floor returns integer zero even for negative floating zero.
            let floor = if quotient == 0.0 {
                0.0
            } else {
                quotient.floor()
            };
            let rounded = (if quotient - floor <= 0.5 {
                floor
            } else {
                floor + 1.0
            }) * c.grid;
            row.unrounded[dimension] = value;
            row.applied[dimension] = rounded;
            row.out_of_range |= u32::from(value.abs() > c.bounds[dimension]);
            row.bound_hit |= u32::from(
                value.abs() >= c.bounds[dimension] || rounded.abs() >= c.bounds[dimension],
            );
        }
        if row.bound_hit == 0 && row.out_of_range == 0 {
            let mut total = 0.0;
            for i in 0..steps {
                let mask = a[i].mask & b[anchor + i].mask;
                let mut residual = 0.0;
                for j in 0..10 {
                    if mask & (1 << j) != 0 {
                        let shift = if j == 0 { row.applied[0] } else { 0.0 };
                        let delta =
                            (a[i].values[j] - b[anchor + i].values[j] - shift) / c.scales[j];
                        residual += delta * delta;
                        row.valid += 1;
                    }
                }
                // Coarse cost sums each pair before adding it to the anchor.
                total += residual;
            }
            row.pairs = steps as u32;
            if row.valid != 0 {
                row.cost = total / row.valid as f64;
            }
        }
        diag.evaluated += 1;
        diag.comparisons += row.comparisons;
        diag.bound_anchors += row.bound_hit;
        // Anchors arrive in order, so strict cost comparison keeps the first tie.
        if !row.cost.is_nan() && (diag.index < 0 || row.cost < best_cost) {
            diag.index = index as i32;
            best_cost = row.cost;
            best_samples = samples;
        }
        out[index] = row;
    }
    if diag.index >= 0 {
        let row = &out[diag.index as usize];
        // Keep the original sample order for the caller's unchanged math.fsum.
        for (i, sample) in best_samples[0][..row.pitch_samples as usize]
            .iter()
            .enumerate()
        {
            diag.pitch_error[i] = (*sample - row.unrounded[0]).powf(std::hint::black_box(2.0));
        }
        for (i, sample) in best_samples[1][..row.interval_samples as usize]
            .iter()
            .enumerate()
        {
            diag.interval_error[i] = (*sample - row.unrounded[1]).powf(std::hint::black_box(2.0));
        }
    }
    0
}

/// Return zero on success, one for invalid input, or four for Python diagnostics.
///
/// # Safety
/// Pointers must refer to aligned, disjoint live buffers. Cue/reference contain
/// config.n/config.m knots; output covers one Output. The caller owns all buffers.
#[no_mangle]
pub unsafe extern "C" fn temporal_dtw(
    cue: *const Knot,
    reference: *const Knot,
    config: *const Config,
    output: *mut Output,
) -> i32 {
    if cue.is_null() || reference.is_null() || config.is_null() || output.is_null() {
        return 1;
    }
    let c = unsafe { &*config };
    let n = c.n as usize;
    let m = c.m as usize;
    if n == 0
        || m == 0
        || n > CAPACITY
        || m > CAPACITY
        || c.anchor as usize >= m
        || c.band < -1
        || c.band > CAPACITY as i32
        || !c.shift.is_finite()
        || !c.ratio.is_finite()
        || c.ratio < 0.0
        || !c.insertion.is_finite()
        || c.insertion <= 0.0
        || !c.deletion.is_finite()
        || c.deletion <= 0.0
        || c.scales.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return 1;
    }
    let a = unsafe { slice::from_raw_parts(cue, n) };
    let b = unsafe { slice::from_raw_parts(reference, m) };
    for sequence in [a, b] {
        for (i, knot) in sequence.iter().enumerate() {
            if knot.mask > 1023
                || !knot.time.is_finite()
                || (i > 0 && knot.time <= sequence[i - 1].time)
                || knot
                    .values
                    .iter()
                    .enumerate()
                    .any(|(j, value)| knot.mask & (1 << j) != 0 && !value.is_finite())
            {
                return 1;
            }
        }
    }
    let out = unsafe { &mut *output };
    let width = if c.band == -1 {
        m
    } else {
        m.min(2 * c.band as usize + 1)
    };
    // Fixed physical storage also covers the registered unbanded control.
    let mut previous = [0.0; CAPACITY + 1];
    let mut current = [f64::INFINITY; CAPACITY + 1];
    out.starts.fill(0);
    out.ends.fill(0);
    out.parents.fill(0);
    out.cells = 0;
    out.time_comparisons = 0;
    out.width = width as u32;
    for (i, knot) in a.iter().enumerate() {
        let (lo, hi) = if c.band == -1 {
            (0, m)
        } else {
            let predicted = b[c.anchor as usize].time + c.ratio * (knot.time - a[0].time);
            let (mut left, mut right) = (0, m);
            while left < right {
                let middle = (left + right) / 2;
                out.time_comparisons += 1;
                if b[middle].time < predicted {
                    left = middle + 1;
                } else {
                    right = middle;
                }
            }
            let mut center = (m - 1).min(left);
            if left > 0 && (left == m || predicted - b[left - 1].time <= b[left].time - predicted) {
                center = left - 1;
            }
            (
                center.saturating_sub(c.band as usize),
                m.min(center + c.band as usize + 1),
            )
        };
        out.starts[i] = lo as u16;
        out.ends[i] = hi as u16;
        current[..=m].fill(f64::INFINITY);
        if lo == 0 {
            current[0] = (i + 1) as f64 * c.insertion;
        }
        for j in lo..hi {
            let mask = knot.mask & b[j].mask;
            let mut residual = 0.0;
            let mut count = 0;
            for k in 0..10 {
                if mask & (1 << k) != 0 {
                    let shift = if k == 0 { c.shift } else { 0.0 };
                    let delta = (knot.values[k] - b[j].values[k] - shift) / c.scales[k];
                    residual += delta * delta;
                    count += 1;
                }
            }
            let step = if count == 0 {
                0.0
            } else {
                residual / count as f64
            };
            let mut cost = previous[j] + step;
            let mut operation = 1;
            // Strict comparisons preserve diagonal/insertion/deletion tie order.
            let insertion = previous[j + 1] + c.insertion;
            if insertion < cost {
                cost = insertion;
                operation = 2;
            }
            let deletion = current[j] + c.deletion;
            if deletion < cost {
                cost = deletion;
                operation = 3;
            }
            current[j + 1] = cost;
            out.parents[i * width + j - lo] = operation;
            out.cells += 1;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let mut endpoint = 1;
    for j in 2..=m {
        if previous[j] < previous[endpoint] {
            endpoint = j;
        }
    }
    out.endpoint = endpoint as u32;
    out.total = previous[endpoint];
    out.path_len = 0;
    out.coordinate_error.fill(0.0);
    out.coordinate_count.fill(0);
    out.reference_start = 0;
    out.observed = a.iter().filter(|row| row.observed_sec > 0.0).count() as u32;
    out.matched = 0;
    out.missing = 0;
    out.inserted = 0;
    out.deleted = 0;
    out.valid_coordinates = 0;
    out.band_edge = 0;
    out.motion_count = 0;
    out.interval_count = 0;
    if !out.total.is_finite() {
        return 0;
    }
    let (mut i, mut j) = (n, endpoint);
    while i > 0 {
        if out.path_len as usize == out.path.len() {
            return 1;
        }
        let step = &mut out.path[out.path_len as usize];
        out.path_len += 1;
        if j == 0 {
            *step = [2, (i - 1) as i16, -1];
            out.inserted += 1;
            i -= 1;
            continue;
        }
        let (lo, hi) = (out.starts[i - 1] as usize, out.ends[i - 1] as usize);
        if j - 1 < lo || j > hi {
            return 1;
        }
        if c.band != -1 && ((j - 1 == lo && lo > 0) || (j == hi && hi < m)) {
            out.band_edge = 1;
        }
        match out.parents[(i - 1) * width + j - 1 - lo] {
            1 => {
                *step = [1, (i - 1) as i16, (j - 1) as i16];
                let mask = a[i - 1].mask & b[j - 1].mask;
                for k in 0..10 {
                    if mask & (1 << k) != 0 {
                        let shift = if k == 0 { c.shift } else { 0.0 };
                        let delta = (a[i - 1].values[k] - b[j - 1].values[k] - shift) / c.scales[k];
                        // Match the reference's reverse-trace accumulation order.
                        out.coordinate_error[k] += delta * delta;
                        out.coordinate_count[k] += 1;
                    }
                }
                out.valid_coordinates += mask.count_ones();
                out.matched += u32::from(mask != 0);
                out.missing += u32::from(mask == 0);
                i -= 1;
                j -= 1;
            }
            2 => {
                *step = [2, (i - 1) as i16, (j - 1) as i16];
                out.inserted += 1;
                i -= 1;
            }
            3 => {
                *step = [3, -1, (j - 1) as i16];
                out.deleted += 1;
                j -= 1;
            }
            _ => return 1,
        }
    }
    out.reference_start = j as u32;
    out.path[..out.path_len as usize].reverse();
    for steps in out.path[..out.path_len as usize].windows(2) {
        let (left, right) = (steps[0], steps[1]);
        if left[0] != 1 || right[0] != 1 || right[1] != left[1] + 1 || right[2] != left[2] + 1 {
            continue;
        }
        let rows = [
            &a[left[1] as usize],
            &a[right[1] as usize],
            &b[left[2] as usize],
            &b[right[2] as usize],
        ];
        // Unknown/custom gap objects retain lazy Python truth/error semantics.
        if rows.iter().any(|row| row.gap > 1) {
            return 4;
        }
        if rows.iter().any(|row| row.gap == 1) {
            continue;
        }
        if rows.iter().all(|row| row.mask & 1 != 0) {
            let delta =
                (rows[1].values[0] - rows[0].values[0]) - (rows[3].values[0] - rows[2].values[0]);
            // Python **2 calls libm pow, which may differ from multiplication.
            let squared = delta.powf(std::hint::black_box(2.0));
            if squared.is_infinite() && delta.is_finite() {
                return 4;
            }
            out.motion_error[out.motion_count as usize] = squared;
            out.motion_count += 1;
        }
        if rows
            .iter()
            .all(|row| row.observed_sec > 0.0 && row.mask != 0)
        {
            let ratio = (rows[3].time - rows[2].time) / (rows[1].time - rows[0].time);
            if ratio <= 0.0 {
                return 4;
            }
            let delta = ratio.log2() - c.tempo_shift;
            let squared = delta.powf(std::hint::black_box(2.0));
            if squared.is_infinite() && delta.is_finite() {
                return 4;
            }
            out.interval_error[out.interval_count as usize] = squared;
            out.interval_count += 1;
        }
    }
    0
}
