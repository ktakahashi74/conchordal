//! core/log2.rs — Log2 frequency space (octave-based).
//!
//! Provides uniform log2(Hz) mapping used by NSGT and kernel convolution.
//! Example: 27.5 Hz → log2(27.5)=4.78, 55 Hz→5.78 (1 octave up).

/// Uniform log2(Hz) frequency space.
#[derive(Clone, Debug)]
pub struct Log2Space {
    pub fmin: f32,
    pub fmax: f32,
    pub bins_per_oct: u32,
    pub centers_hz: Vec<f32>,
    pub centers_log2: Vec<f32>,
    pub step_log2: f32,
}

/// Explicit log2 axis specification (for external DSP handoff or regeneration).
#[derive(Clone, Copy, Debug)]
pub struct Log2SpaceSpec {
    pub fmin: f32,
    pub fmax: f32,
    pub bins_per_oct: u32,
}

impl Log2Space {
    /// Create a log2-space grid between fmin..fmax (inclusive).
    pub fn new(fmin: f32, fmax: f32, bins_per_oct: u32) -> Self {
        assert!(fmin > 0.0 && fmax > fmin);
        assert!(bins_per_oct > 0);

        let lo = fmin.log2();
        let hi = fmax.log2();
        let step_log2 = 1.0 / bins_per_oct as f32;
        let n_bins = ((hi - lo) / step_log2).floor() as usize + 1;

        let centers_log2: Vec<f32> = (0..n_bins).map(|i| lo + i as f32 * step_log2).collect();
        let centers_hz: Vec<f32> = centers_log2.iter().map(|&x| 2f32.powf(x)).collect();

        Self {
            fmin,
            fmax,
            bins_per_oct,
            centers_hz,
            centers_log2,
            step_log2,
        }
    }

    /// Number of bins.
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.centers_hz.len()
    }

    #[inline]
    pub fn assert_scan_len<T>(&self, scan: &[T]) {
        debug_assert_eq!(scan.len(), self.n_bins());
    }

    #[inline]
    pub fn assert_scan_len_named<T>(&self, scan: &[T], name: &str) {
        debug_assert_eq!(scan.len(), self.n_bins(), "scan length mismatch: {name}");
    }

    /// Compact metadata for re-creating this log2 axis elsewhere.
    #[inline]
    pub fn spec(&self) -> Log2SpaceSpec {
        Log2SpaceSpec {
            fmin: self.fmin,
            fmax: self.fmax,
            bins_per_oct: self.bins_per_oct,
        }
    }

    /// Convert Hz → log2(Hz)
    #[inline]
    pub fn hz_to_log2(&self, hz: f32) -> f32 {
        hz.log2()
    }

    /// Convert log2(Hz) → Hz
    #[inline]
    pub fn log2_to_hz(&self, l: f32) -> f32 {
        2f32.powf(l)
    }

    /// Return Δlog2 per bin (1/bins_per_oct).
    #[inline]
    pub fn step(&self) -> f32 {
        self.step_log2
    }

    /// Find nearest bin index for given frequency.
    pub fn index_of_freq(&self, hz: f32) -> Option<usize> {
        if hz < self.fmin || hz > self.fmax {
            return None;
        }
        let l = hz.log2();
        let idx = ((l - self.centers_log2[0]) / self.step_log2).round() as usize;
        self.centers_hz.get(idx)?;
        Some(idx)
    }

    /// Map Hz to a continuous bin position (0..n_bins-1).
    pub fn bin_pos_of_freq(&self, hz: f32) -> Option<f32> {
        if !hz.is_finite() || hz < self.fmin || hz > self.fmax {
            return None;
        }
        let l = hz.log2();
        let pos = (l - self.centers_log2[0]) / self.step_log2;
        if pos.is_finite() { Some(pos) } else { None }
    }

    /// Find nearest bin index for a log2(Hz) coordinate.
    pub fn index_of_log2(&self, log2_hz: f32) -> Option<usize> {
        let hz = 2f32.powf(log2_hz);
        self.index_of_freq(hz)
    }

    pub fn freq_of_index(&self, i: usize) -> f32 {
        self.centers_hz[i]
    }

    /// Approximate linear bandwidth (Hz) of a log2-space bin.
    ///
    /// - Constant-Q spacing: Δlog2 = 1 / bins_per_oct.
    /// - Formula: Δf = f * (2^(Δlog2/2) − 2^(−Δlog2/2)).
    /// - Q = f / Δf = 1 / (2^(Δlog2/2) − 2^(−Δlog2/2)).
    ///
    /// Use `bandwidth_hz_at(i)` for bin index, or this for arbitrary frequency.
    #[inline]
    pub fn bandwidth_hz(&self, f_hz: f32) -> f32 {
        let half_step = 0.5 * self.step_log2;
        let delta = 2f32.powf(half_step) - 2f32.powf(-half_step);
        (f_hz * delta).max(1e-6)
    }

    /// Bandwidth (Hz) of the i-th bin center.
    #[inline]
    pub fn bandwidth_hz_at(&self, i: usize) -> f32 {
        if let Some(&f_hz) = self.centers_hz.get(i) {
            self.bandwidth_hz(f_hz)
        } else {
            0.0
        }
    }

    /// Return Δlog2 between two frequencies.
    #[inline]
    pub fn delta_log2(&self, f1: f32, f2: f32) -> f32 {
        f2.log2() - f1.log2()
    }
}

/// Sample a Log2Space-aligned scan by linear interpolation in log2-frequency.
pub fn sample_scan_linear_log2(space: &Log2Space, scan: &[f32], freq_hz: f32) -> f32 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if freq_hz < space.fmin || freq_hz > space.fmax {
        return f32::NEG_INFINITY;
    }
    debug_assert_eq!(scan.len(), space.n_bins());
    let pos = match space.bin_pos_of_freq(freq_hz) {
        Some(pos) => pos,
        None => return f32::NEG_INFINITY,
    };
    let idx_base = pos.floor();
    let idx = idx_base as isize;
    if idx < 0 {
        return f32::NEG_INFINITY;
    }
    let idx = idx as usize;
    let frac = pos - idx_base;
    if idx + 1 < scan.len() {
        let v0 = scan[idx];
        let v1 = scan[idx + 1];
        (v0 * (1.0 - frac)) + (v1 * frac)
    } else if idx < scan.len() {
        scan[idx]
    } else {
        f32::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2space_basic() {
        let s = Log2Space::new(27.5, 880.0, 12);
        assert!(s.n_bins() > 0);
        assert!(s.centers_hz[0] >= 27.5);
        assert!(s.centers_hz.last().unwrap() <= &880.1);
        let idx = s.index_of_freq(55.0).unwrap();
        let ratio = s.centers_hz[idx] / s.centers_hz[0];
        // One octave ≈ factor 2
        assert!(ratio > 1.9 && ratio < 2.1);
    }

    #[test]
    fn test_log2space_geometric_spacing() {
        let s = Log2Space::new(55.0, 3520.0, 24);
        let ratios: Vec<f32> = s.centers_hz.windows(2).map(|w| w[1] / w[0]).collect();
        let target = ratios[0];
        assert!(ratios.iter().all(|&r| (r / target - 1.0).abs() < 1e-6));
        assert!(s.centers_hz.windows(2).all(|w| w[1] > w[0]));
        assert!(s.centers_hz[0] >= s.fmin);
        assert!(s.centers_hz.last().unwrap() <= &s.fmax);
    }

    #[test]
    fn test_bandwidth_constant_q_relation() {
        let space = Log2Space::new(100.0, 6400.0, 24);
        let q_target = 1.0 / (2f32.powf(1.0 / (24.0 * 2.0)) - 2f32.powf(-1.0 / (24.0 * 2.0)));
        for &f in &[100.0, 400.0, 1600.0, 6400.0] {
            let bw = space.bandwidth_hz(f);
            let q = f / bw;
            assert!(
                (q / q_target - 1.0).abs() < 1e-5,
                "Q mismatch: got {q}, want {q_target}"
            );
        }
    }

    #[test]
    fn test_bandwidth_hz_at_matches_bandwidth_hz() {
        let space = Log2Space::new(100.0, 6400.0, 24);
        for i in [0, space.n_bins() / 2, space.n_bins() - 1] {
            let bw_f = space.bandwidth_hz(space.centers_hz[i]);
            let bw_i = space.bandwidth_hz_at(i);
            assert!((bw_f - bw_i).abs() / bw_f < 1e-6, "Mismatch at bin {i}");
        }
    }

    #[test]
    fn test_bandwidth_scales_linearly_with_frequency() {
        let space = Log2Space::new(100.0, 6400.0, 12);
        let bw1 = space.bandwidth_hz(100.0);
        let bw2 = space.bandwidth_hz(200.0);
        let bw4 = space.bandwidth_hz(400.0);
        let ratio12 = bw2 / bw1;
        let ratio24 = bw4 / bw2;
        assert!((ratio12 - 2.0).abs() < 0.05);
        assert!((ratio24 - 2.0).abs() < 0.05);
    }
}
