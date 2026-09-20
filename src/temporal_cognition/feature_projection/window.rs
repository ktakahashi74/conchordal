//! Physical-support window arithmetic shared by observed and hypothetical heads.

use super::Feature;

#[derive(Clone, Copy)]
pub(in crate::temporal_cognition) struct Frame {
    pub start: u64,
    pub end: u64,
    /// Original audio dependency and delivery clocks for observed coordinates.
    pub source_end: u64,
    pub available: u64,
    pub raw: [Feature; 10],
    pub energy: Feature,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(in crate::temporal_cognition) struct Summary {
    /// Rise, decline, flux, share; low-energy fraction, mean RMS, log-RMS slope.
    pub values: [Feature; 7],
    pub observed_fraction: [f64; 7],
    pub projected_fraction: [f64; 7],
}

/// The issue-time backoff coverage belongs to the caller and is never advanced here.
pub(in crate::temporal_cognition) fn summarize(
    start: u64,
    end: u64,
    issue: u64,
    rate: u32,
    rms_reference: f64,
    frames: impl Iterator<Item = Frame>,
) -> Result<Summary, &'static str> {
    if start > end || rate == 0 || !rms_reference.is_finite() || rms_reference <= 0. {
        return Err("invalid feature window");
    }
    let mut result = Summary {
        values: [Feature::Unsupported; 7],
        observed_fraction: [0.; 7],
        projected_fraction: [0.; 7],
    };
    if start == end {
        return Ok(result);
    }
    let mut counts = [[0_u64; 2]; 6];
    let mut totals = [0.; 6];
    let mut first = None;
    let mut last = None;
    let mut previous_end = None;
    for frame in frames {
        if frame.start >= frame.end {
            return Err("invalid feature frame interval");
        }
        if frame.end <= start || frame.start >= end {
            continue;
        }
        if previous_end.is_some_and(|prior| frame.start < prior) {
            return Err("overlapping or unordered feature frames");
        }
        previous_end = Some(frame.end);
        let a = start.max(frame.start);
        let b = end.min(frame.end);
        let n = b - a;
        // Only dependencies consumed by these seven coordinates constrain provenance.
        let dependencies = [
            frame.raw[3],
            frame.raw[4],
            frame.raw[5],
            frame.raw[6],
            frame.raw[2],
            frame.energy,
        ];
        for (i, feature) in dependencies.into_iter().enumerate() {
            let Some(value) = feature.value() else {
                continue;
            };
            if !value.is_finite() || (i != 4 && value < 0.) {
                return Err("invalid feature window value");
            }
            if (feature.projected() && frame.start < issue)
                || (!feature.projected()
                    && (frame.end > issue
                        || frame.source_end < frame.end
                        || frame.source_end > frame.available
                        || frame.available > issue))
            {
                return Err("feature support crosses its issue-time provenance");
            }
            counts[i][usize::from(feature.projected())] += n;
            let value = match i {
                4 => {
                    let endpoint = (a, b, value);
                    first.get_or_insert(endpoint);
                    last = Some(endpoint);
                    f64::from(2_f64.powf(value) <= 0.01 * rms_reference)
                }
                5 => value.sqrt(),
                _ => value,
            };
            totals[i] += value * n as f64;
        }
    }
    let duration = (end - start) as f64;
    for i in 0..6 {
        let count = counts[i][0] + counts[i][1];
        result.observed_fraction[i] = counts[i][0] as f64 / duration;
        result.projected_fraction[i] = counts[i][1] as f64 / duration;
        let covered = if i < 4 {
            count as f64 >= 0.9 * duration
        } else {
            count as f64 / duration >= 0.9
        };
        if count > 0 && covered {
            result.values[i] = Feature::from(Some(totals[i] / count as f64), counts[i][1] > 0);
        }
    }
    if let (Some((a, b, initial)), Some((c, d, final_value))) = (first, last)
        && a == start
        && d == end
        && c > a
        && result.values[4].value().is_some()
    {
        let seconds = ((c - a) as f64 + (d - b) as f64) / (2. * f64::from(rate));
        result.values[6] = Feature::from(Some((final_value - initial) / seconds), counts[4][1] > 0);
        result.observed_fraction[6] = result.observed_fraction[4];
        result.projected_fraction[6] = result.projected_fraction[4];
    }
    if result
        .values
        .iter()
        .filter_map(|v| v.value())
        .any(|v| !v.is_finite())
    {
        return Err("feature window accumulation overflow");
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
