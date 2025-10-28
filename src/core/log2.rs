//! core/log2.rs — Log2 frequency space (octave-based).
//!
//! Provides uniform log2(Hz) mapping used by NSGT and kernel convolution.
//! Example: 27.5 Hz → log2(27.5)=4.78, 55 Hz→5.78 (1 octave up).

use std::f32::consts::LOG2_10;

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

    pub fn delta_hz_at(&self, f: f32) -> Option<f32> {
        let idx = self.index_of_freq(f)?;
        let df = if idx == 0 {
            self.centers_hz[1] - self.centers_hz[0]
        } else if idx + 1 >= self.centers_hz.len() {
            self.centers_hz[idx] - self.centers_hz[idx - 1]
        } else {
            (self.centers_hz[idx + 1] - self.centers_hz[idx - 1]) / 2.0
        };
        Some(df.max(10.0)) // lower bound
    }

    /// Return Δlog2 between two frequencies.
    #[inline]
    pub fn delta_log2(&self, f1: f32, f2: f32) -> f32 {
        f2.log2() - f1.log2()
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
}
