//! Safe, bounded f64 algorithms migrated from the frozen M0 native kernel.
//! FFI, Python ownership and exception fallbacks stay in the research reference.

pub(super) const CAPACITY: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Error {
    InvalidInput,
    NumericalRange,
}

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub(super) struct Knot {
    pub(super) values: [f64; 10],
    pub(super) time: f64,
    pub(super) mask: u64,
    pub(super) timing: u64,
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) observed_sec: f64,
    pub(super) raw_start: f64,
    pub(super) raw_end: f64,
    pub(super) available_end: f64,
    pub(super) lineage_changed: u64,
    pub(super) gap: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Config {
    pub(super) anchor: u32,
    pub(super) band: i32,
    pub(super) shift: f64,
    pub(super) ratio: f64,
    pub(super) tempo_shift: f64,
    pub(super) scales: [f64; 10],
    pub(super) insertion: f64,
    pub(super) deletion: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Output {
    pub(super) total: f64,
    pub(super) endpoint: u32,
    pub(super) cells: u32,
    pub(super) time_comparisons: u32,
    pub(super) width: u32,
    pub(super) starts: [u16; CAPACITY],
    pub(super) ends: [u16; CAPACITY],
    pub(super) parents: [u8; CAPACITY * CAPACITY],
    pub(super) coordinate_error: [f64; 10],
    pub(super) coordinate_count: [u32; 10],
    pub(super) path: [[i16; 3]; CAPACITY * 2],
    pub(super) path_len: u32,
    pub(super) reference_start: u32,
    pub(super) observed: u32,
    pub(super) matched: u32,
    pub(super) missing: u32,
    pub(super) inserted: u32,
    pub(super) deleted: u32,
    pub(super) valid_coordinates: u32,
    pub(super) band_edge: u32,
    pub(super) motion_count: u32,
    pub(super) interval_count: u32,
    pub(super) motion_error: [f64; CAPACITY],
    pub(super) interval_error: [f64; CAPACITY],
}

impl Default for Output {
    fn default() -> Self {
        Self {
            total: 0.0,
            endpoint: 0,
            cells: 0,
            time_comparisons: 0,
            width: 0,
            starts: [0; CAPACITY],
            ends: [0; CAPACITY],
            parents: [0; CAPACITY * CAPACITY],
            coordinate_error: [0.0; 10],
            coordinate_count: [0; 10],
            path: [[0; 3]; CAPACITY * 2],
            path_len: 0,
            reference_start: 0,
            observed: 0,
            matched: 0,
            missing: 0,
            inserted: 0,
            deleted: 0,
            valid_coordinates: 0,
            band_edge: 0,
            motion_count: 0,
            interval_count: 0,
            motion_error: [0.0; CAPACITY],
            interval_error: [0.0; CAPACITY],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CoarseConfig {
    pub(super) spacing: u32,
    pub(super) limit: u32,
    pub(super) bounds: [f64; 2],
    pub(super) grid: f64,
    pub(super) scales: [f64; 10],
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Anchor {
    pub(super) cost: f64,
    pub(super) unrounded: [f64; 2],
    pub(super) applied: [f64; 2],
    pub(super) valid: u32,
    pub(super) pairs: u32,
    pub(super) comparisons: u32,
    pub(super) pitch_samples: u32,
    pub(super) interval_samples: u32,
    pub(super) out_of_range: u32,
    pub(super) bound_hit: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct AnchorDiagnostic {
    pub(super) index: i32,
    pub(super) evaluated: u32,
    pub(super) comparisons: u32,
    pub(super) bound_anchors: u32,
    pub(super) pitch_error: [f64; 8],
    pub(super) interval_error: [f64; 8],
}

pub(super) fn validate(rows: &[Knot], observed_end: f64) -> Result<(), Error> {
    if rows.len() > CAPACITY || !observed_end.is_finite() {
        return Err(Error::InvalidInput);
    }
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
            return Err(Error::InvalidInput);
        }
    }
    Ok(())
}
pub(super) fn coarse(
    a: &[Knot],
    b: &[Knot],
    c: &CoarseConfig,
    out: &mut [Anchor; CAPACITY],
    diag: &mut AnchorDiagnostic,
) -> Result<(), Error> {
    let (n, m, spacing, limit) = (a.len(), b.len(), c.spacing as usize, c.limit as usize);
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
        return Err(Error::InvalidInput);
    }
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
                return Err(Error::InvalidInput);
            }
        }
    }
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
                        return Err(Error::NumericalRange);
                    }
                    if ratio <= 0.0 {
                        return Err(Error::NumericalRange);
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
            // Reject arithmetic outside the finite numerical domain.
            if values[..count]
                .iter()
                .any(|sample| (*sample - value).abs() > f64::MAX.sqrt() / 8.0)
            {
                return Err(Error::NumericalRange);
            }
            let quotient = value / c.grid;
            if !quotient.is_finite() {
                return Err(Error::NumericalRange);
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
    Ok(())
}

pub(super) fn dtw(a: &[Knot], b: &[Knot], c: &Config, out: &mut Output) -> Result<(), Error> {
    let (n, m) = (a.len(), b.len());
    if n == 0
        || m == 0
        || n > CAPACITY
        || m > CAPACITY
        || c.anchor as usize >= m
        || c.band < -1
        || c.band > CAPACITY as i32
        || !c.shift.is_finite()
        || !c.tempo_shift.is_finite()
        || !c.ratio.is_finite()
        || c.ratio < 0.0
        || !c.insertion.is_finite()
        || c.insertion <= 0.0
        || !c.deletion.is_finite()
        || c.deletion <= 0.0
        || c.scales.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(Error::InvalidInput);
    }
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
                return Err(Error::InvalidInput);
            }
        }
    }
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
            if !predicted.is_finite() {
                return Err(Error::NumericalRange);
            }
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
            if !step.is_finite() {
                return Err(Error::NumericalRange);
            }
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
        return Ok(());
    }
    let (mut i, mut j) = (n, endpoint);
    while i > 0 {
        if out.path_len as usize == out.path.len() {
            return Err(Error::InvalidInput);
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
            return Err(Error::InvalidInput);
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
            _ => return Err(Error::InvalidInput),
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
        // Reject invalid gap flags before emitting diagnostics.
        if rows.iter().any(|row| row.gap > 1) {
            return Err(Error::NumericalRange);
        }
        if rows.iter().any(|row| row.gap == 1) {
            continue;
        }
        if rows.iter().all(|row| row.mask & 1 != 0) {
            let delta =
                (rows[1].values[0] - rows[0].values[0]) - (rows[3].values[0] - rows[2].values[0]);
            // Python **2 calls libm pow, which may differ from multiplication.
            let squared = delta.powf(std::hint::black_box(2.0));
            if !squared.is_finite() {
                return Err(Error::NumericalRange);
            }
            out.motion_error[out.motion_count as usize] = squared;
            out.motion_count += 1;
        }
        if rows
            .iter()
            .all(|row| row.observed_sec > 0.0 && row.mask != 0)
        {
            let ratio = (rows[3].time - rows[2].time) / (rows[1].time - rows[0].time);
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(Error::NumericalRange);
            }
            let delta = ratio.log2() - c.tempo_shift;
            let squared = delta.powf(std::hint::black_box(2.0));
            if !squared.is_finite() {
                return Err(Error::NumericalRange);
            }
            out.interval_error[out.interval_count as usize] = squared;
            out.interval_count += 1;
        }
    }
    Ok(())
}
