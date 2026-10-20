//! core/erb.rs — ERB scale conversions (Glasberg & Moore).

use std::f32::consts::LN_10;

/// Converts frequency [Hz] to ERB-rate value.
#[inline]
pub fn hz_to_erb(f_hz: f32) -> f32 {
    // 21.4 * log10(4.37e-3 * f + 1)
    21.4 * ((4.37e-3 * f_hz + 1.0).ln() / LN_10)
}

/// Converts ERB-rate value to frequency [Hz].
#[inline]
pub fn erb_to_hz(e: f32) -> f32 {
    // (10^(e/21.4) - 1) / 4.37e-3
    (((e / 21.4) * LN_10).exp() - 1.0) / 4.37e-3
}

/// Returns ERB bandwidth in Hz (Glasberg & Moore 1990)
#[inline]
pub fn erb_bw_hz(f_hz: f32) -> f32 {
    24.7 * (4.37e-3 * f_hz + 1.0)
}

pub(crate) struct ErbCell {
    pub(crate) lo_erb: f32,
    pub(crate) hi_erb: f32,
    pub(crate) bins: std::ops::Range<usize>,
}

/// One-ERB cells; merge short trailing edges and empty cells as in Flora.
pub(crate) fn partition_erb_cells(
    space: &super::log2space::Log2Space,
    erb_scan: &[f32],
    extent_hz: (f32, f32),
) -> Vec<ErbCell> {
    space.assert_scan_len_named(erb_scan, "erb_scan");
    let start = space.centers_hz.partition_point(|&f| f < extent_hz.0);
    let end = space.centers_hz.partition_point(|&f| f <= extent_hz.1);
    let mut cells: Vec<ErbCell> = Vec::new();
    if start >= end {
        return cells;
    }
    let lo = hz_to_erb(extent_hz.0.max(space.fmin));
    let hi = hz_to_erb(extent_hz.1.min(space.fmax));
    let whole = (hi - lo).floor() as usize;
    let count = if hi - lo - (whole as f32) < 0.5 {
        whole.max(1)
    } else {
        whole + 1
    };
    let mut bin = start;
    let mut pending_lo = lo;
    for c in 0..count {
        let upper = if c + 1 == count {
            hi
        } else {
            lo + (c + 1) as f32
        };
        let first = bin;
        while bin < end && (c + 1 == count || erb_scan[bin] < upper) {
            bin += 1;
        }
        if bin == first {
            if let Some(prev) = cells.last_mut() {
                prev.hi_erb = upper;
            }
        } else {
            cells.push(ErbCell {
                lo_erb: pending_lo,
                hi_erb: upper,
                bins: first..bin,
            });
        }
        if !cells.is_empty() {
            pending_lo = upper;
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip test of hz_to_erb ↔ erb_to_hz conversions.
    #[test]
    fn test_erb_conversion_roundtrip() {
        // Check round-trip accuracy at multiple points.
        for f in [20.0, 100.0, 1000.0, 4000.0, 8000.0, 16000.0] {
            let e = hz_to_erb(f);
            let f2 = erb_to_hz(e);
            let rel_err = (f - f2).abs() / f;
            assert!(
                rel_err < 1e-4,
                "roundtrip mismatch: f={f}, f2={f2}, rel_err={rel_err}"
            );
        }
    }

    /// Validate that hz_to_erb is monotonic and increasing.
    #[test]
    fn test_hz_to_erb_monotonic() {
        let f: Vec<f32> = (1..20).map(|i| i as f32 * 500.0).collect();
        let e: Vec<f32> = f.iter().map(|&x| hz_to_erb(x)).collect();
        assert!(
            e.windows(2).all(|w| w[1] > w[0]),
            "hz_to_erb not strictly increasing"
        );
    }

    /// Verify erb_bw_hz matches Glasberg & Moore expected scaling.
    #[test]
    fn test_erb_bw_reference_values() {
        let bw_1k = erb_bw_hz(1000.0);
        let bw_4k = erb_bw_hz(4000.0);
        // Matches known reference values around 132.6 Hz and 456.4 Hz.
        assert!((bw_1k - 132.6).abs() < 1.0, "bw(1kHz) mismatch: {bw_1k}");
        assert!((bw_4k - 456.4).abs() < 1.0, "bw(4kHz) mismatch: {bw_4k}");
        assert!(bw_4k > bw_1k);
    }

    #[test]
    fn delta_erb_mapping_matches_exact() {
        use super::*;
        // Check several frequency deltas near 1 kHz.
        let fi = 1000.0f32;
        let steps_hz = [
            -300.0, -150.0, -75.0, -30.0, -15.0, 0.0, 15.0, 30.0, 75.0, 150.0, 300.0,
        ];

        for df_hz in steps_hz {
            let fj = (fi + df_hz).max(1.0);
            let d_exact = hz_to_erb(fj) - hz_to_erb(fi);
            let bw_mid = erb_bw_hz(0.5 * (fi + fj));
            let d_approx = (fj - fi) / bw_mid;

            // Relative error within 3% is OK.
            let denom = d_exact.abs().max(1e-6);
            let rel_err = (d_exact - d_approx).abs() / denom;
            assert!(
                rel_err < 0.03,
                "ΔERB approx mismatch at df={df_hz}: exact={d_exact}, approx={d_approx}, rel_err={rel_err}"
            );
        }
    }
}
