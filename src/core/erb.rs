//! core/erb.rs — ERB scale utilities and ErbSpace definition.

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

/// Represents a frequency axis linearly spaced in ERB domain.
#[derive(Clone, Debug)]
pub struct ErbSpace {
    pub f_min: f32,
    pub f_max: f32,
    pub erb_step: f32,
    pub erb_min: f32,
    pub erb_max: f32,
    pub freqs_hz: Vec<f32>,
}

impl ErbSpace {
    /// Create a new ERB-space axis.
    ///
    /// # Arguments
    /// * `f_min` - lowest frequency [Hz]
    /// * `f_max` - highest frequency [Hz]
    /// * `erb_step` - ERB step (smaller → denser sampling)
    pub fn new(f_min: f32, f_max: f32, erb_step: f32) -> Self {
        let erb_min = hz_to_erb(f_min);
        let erb_max = hz_to_erb(f_max);

        let n_points = ((erb_max - erb_min) / erb_step).floor() as usize + 1;

        let mut freqs_hz = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let e = erb_min + i as f32 * erb_step;
            freqs_hz.push(erb_to_hz(e));
        }

        Self {
            f_min,
            f_max,
            erb_step,
            erb_min,
            erb_max,
            freqs_hz,
        }
    }

    /// Convert frequency [Hz] to ERB coordinate.
    #[inline]
    pub fn to_erb(&self, f_hz: f32) -> f32 {
        hz_to_erb(f_hz)
    }

    /// Convert ERB coordinate to frequency [Hz].
    #[inline]
    pub fn to_hz(&self, e: f32) -> f32 {
        erb_to_hz(e)
    }

    /// Return number of ERB bins.
    #[inline]
    pub fn len(&self) -> usize {
        self.freqs_hz.len()
    }

    /// Returns true if there are no ERB bins.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.freqs_hz.is_empty()
    }

    /// Return reference to frequency vector [Hz].
    #[inline]
    pub fn freqs_hz(&self) -> &[f32] {
        &self.freqs_hz
    }

    pub fn index_of_freq(&self, f_hz: f32) -> usize {
        self.freqs_hz
            .iter()
            .position(|&f| f >= f_hz)
            .unwrap_or(self.freqs_hz.len() - 1)
    }
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

    /// Check monotonicity and range of ErbSpace.
    #[test]
    fn test_erbspace_creation() {
        let space = ErbSpace::new(50.0, 8000.0, 0.5);
        // Even with a coarse ERB step there should be ample bins.
        assert!(
            space.freqs_hz.len() > 10,
            "too few bins: {}",
            space.freqs_hz.len()
        );
        assert!(space.freqs_hz[0] >= 50.0);
        assert!(
            *space.freqs_hz.last().unwrap() <= 8000.0 + 1e-3 * 8000.0,
            "max freq exceeded"
        );

        // Frequency axis must be strictly increasing.
        let diffs: Vec<f32> = space.freqs_hz.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(diffs.iter().all(|&d| d > 0.0), "non-monotonic freqs");
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

    /// Extreme range test (very wide band)
    #[test]
    fn test_erbspace_extreme_range() {
        let space = ErbSpace::new(20.0, 20000.0, 0.25);
        let n = space.len();
        assert!(n > 100, "unexpectedly few bins for wide band: {n}");
        // Ensure ERB min/max are consistent.
        let e_min = hz_to_erb(20.0);
        let e_max = hz_to_erb(20000.0);
        assert!(
            (space.erb_min - e_min).abs() < 1e-4 && (space.erb_max - e_max).abs() < 1e-4,
            "erb range mismatch: min {} vs {}, max {} vs {}",
            space.erb_min,
            e_min,
            space.erb_max,
            e_max
        );
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
