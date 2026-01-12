//! Resonator bank using a damped Modified Coupled Form (Hz, sec).
//! x = r*(x - e*y) + b1*u, y = r*(e*x + y) + b2*u

use crate::synth::modes::{compile_mode, ModeParams};
use crate::synth::util::flush_denorm;
use crate::synth::SynthError;

/// Bank of resonators in a struct-of-arrays layout.
#[derive(Debug, Clone)]
pub struct ResonatorBank {
    fs: f32,
    capacity: usize,
    active_len: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    e: Vec<f32>,
    r: Vec<f32>,
    b1: Vec<f32>,
    b2: Vec<f32>,
    gain: Vec<f32>,
}

impl ResonatorBank {
    /// Create a new bank with a fixed capacity.
    pub fn new(fs: f32, max_modes: usize) -> Result<Self, SynthError> {
        if !fs.is_finite() || fs <= 0.0 {
            return Err(SynthError::InvalidSampleRate);
        }

        let mut x = Vec::with_capacity(max_modes);
        x.resize(max_modes, 0.0);
        let mut y = Vec::with_capacity(max_modes);
        y.resize(max_modes, 0.0);
        let mut e = Vec::with_capacity(max_modes);
        e.resize(max_modes, 0.0);
        let mut r = Vec::with_capacity(max_modes);
        r.resize(max_modes, 0.0);
        let mut b1 = Vec::with_capacity(max_modes);
        b1.resize(max_modes, 0.0);
        let mut b2 = Vec::with_capacity(max_modes);
        b2.resize(max_modes, 0.0);
        let mut gain = Vec::with_capacity(max_modes);
        gain.resize(max_modes, 0.0);

        Ok(Self {
            fs,
            capacity: max_modes,
            active_len: 0,
            x,
            y,
            e,
            r,
            b1,
            b2,
            gain,
        })
    }

    /// Sample rate in Hz.
    pub fn fs(&self) -> f32 {
        self.fs
    }

    /// Maximum number of modes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Active number of modes.
    pub fn active_len(&self) -> usize {
        self.active_len
    }

    /// Reset all internal states to zero.
    pub fn reset_state(&mut self) {
        for v in &mut self.x {
            *v = 0.0;
        }
        for v in &mut self.y {
            *v = 0.0;
        }
    }

    /// Set mode parameters and compile coefficients, resetting state to zero.
    pub fn set_modes(&mut self, modes: &[ModeParams]) -> Result<(), SynthError> {
        if modes.len() > self.capacity {
            return Err(SynthError::TooManyModes {
                requested: modes.len(),
                capacity: self.capacity,
            });
        }

        self.active_len = modes.len();
        self.reset_state();

        for (i, params) in modes.iter().enumerate() {
            let coeffs = compile_mode(params, self.fs);
            self.e[i] = coeffs.e;
            self.r[i] = coeffs.r;
            self.b1[i] = coeffs.b1;
            self.b2[i] = coeffs.b2;
            self.gain[i] = coeffs.gain;
        }

        Ok(())
    }

    /// Process a single sample with the damped MCF update.
    pub fn process_sample(&mut self, u: f32) -> f32 {
        let mut out = 0.0;
        for i in 0..self.active_len {
            let e = self.e[i];
            let r = self.r[i];
            let b1 = self.b1[i];
            let b2 = self.b2[i];
            let gain = self.gain[i];

            let x0 = self.x[i];
            let y0 = self.y[i];

            let mut x1 = r * (x0 - e * y0) + b1 * u;
            x1 = flush_denorm(x1);

            let mut y1 = r * (e * x1 + y0) + b2 * u;
            y1 = flush_denorm(y1);

            self.x[i] = x1;
            self.y[i] = y1;

            out += gain * y1;
        }

        flush_denorm(out)
    }

    /// Process a mono block; input and output slices must be same length.
    pub fn process_block_mono(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (u, y) in input.iter().copied().zip(output.iter_mut()) {
            *y = self.process_sample(u);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_modes_rejects_over_capacity() {
        let mut bank = ResonatorBank::new(48_000.0, 1).unwrap();
        let modes = [
            ModeParams {
                freq_hz: 220.0,
                t60_s: 0.5,
                gain: 1.0,
                in_gain: 1.0,
            },
            ModeParams {
                freq_hz: 330.0,
                t60_s: 0.5,
                gain: 1.0,
                in_gain: 1.0,
            },
        ];
        let err = bank.set_modes(&modes).unwrap_err();
        assert_eq!(
            err,
            SynthError::TooManyModes {
                requested: 2,
                capacity: 1
            }
        );
    }

    #[test]
    fn process_block_is_deterministic() {
        let mut bank = ResonatorBank::new(48_000.0, 2).unwrap();
        let modes = [
            ModeParams {
                freq_hz: 220.0,
                t60_s: 0.3,
                gain: 0.8,
                in_gain: 0.5,
            },
            ModeParams {
                freq_hz: 440.0,
                t60_s: 0.2,
                gain: 0.4,
                in_gain: 0.3,
            },
        ];
        bank.set_modes(&modes).unwrap();

        let input: Vec<f32> = (0..256)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let mut out_a = vec![0.0; input.len()];
        let mut out_b = vec![0.0; input.len()];

        bank.process_block_mono(&input, &mut out_a);
        bank.reset_state();
        bank.process_block_mono(&input, &mut out_b);

        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn zero_input_does_not_grow() {
        let fs = 48_000.0;
        let mut bank = ResonatorBank::new(fs, 1).unwrap();
        let mode = ModeParams {
            freq_hz: 440.0,
            t60_s: 0.05,
            gain: 1.0,
            in_gain: 1.0,
        };
        bank.set_modes(&[mode]).unwrap();

        let _ = bank.process_sample(1.0);

        let mut max_abs: f32 = 0.0;
        let mut last: f32 = 0.0;
        for _ in 0..(fs as usize) {
            last = bank.process_sample(0.0);
            max_abs = max_abs.max(last.abs());
            assert!(last.is_finite());
        }

        assert!(max_abs < 2.0, "max_abs={max_abs}");
        assert!(last.abs() < 1.0e-2, "last={last}");
    }

    #[test]
    fn set_modes_does_not_reallocate() {
        let mut bank = ResonatorBank::new(48_000.0, 4).unwrap();
        let ptrs = (
            bank.x.as_ptr(),
            bank.y.as_ptr(),
            bank.e.as_ptr(),
            bank.r.as_ptr(),
            bank.b1.as_ptr(),
            bank.b2.as_ptr(),
            bank.gain.as_ptr(),
        );
        let caps = (
            bank.x.capacity(),
            bank.y.capacity(),
            bank.e.capacity(),
            bank.r.capacity(),
            bank.b1.capacity(),
            bank.b2.capacity(),
            bank.gain.capacity(),
        );

        let modes_a = [
            ModeParams {
                freq_hz: 220.0,
                t60_s: 0.5,
                gain: 1.0,
                in_gain: 0.1,
            },
            ModeParams {
                freq_hz: 330.0,
                t60_s: 0.3,
                gain: 0.5,
                in_gain: 0.2,
            },
        ];
        let modes_b = [ModeParams {
            freq_hz: 110.0,
            t60_s: 0.2,
            gain: 0.2,
            in_gain: 0.0,
        }];

        bank.set_modes(&modes_a).unwrap();
        assert_eq!(ptrs.0, bank.x.as_ptr());
        assert_eq!(ptrs.1, bank.y.as_ptr());
        assert_eq!(ptrs.2, bank.e.as_ptr());
        assert_eq!(ptrs.3, bank.r.as_ptr());
        assert_eq!(ptrs.4, bank.b1.as_ptr());
        assert_eq!(ptrs.5, bank.b2.as_ptr());
        assert_eq!(ptrs.6, bank.gain.as_ptr());
        assert_eq!(caps.0, bank.x.capacity());
        assert_eq!(caps.1, bank.y.capacity());
        assert_eq!(caps.2, bank.e.capacity());
        assert_eq!(caps.3, bank.r.capacity());
        assert_eq!(caps.4, bank.b1.capacity());
        assert_eq!(caps.5, bank.b2.capacity());
        assert_eq!(caps.6, bank.gain.capacity());

        bank.set_modes(&modes_b).unwrap();
        assert_eq!(ptrs.0, bank.x.as_ptr());
        assert_eq!(ptrs.1, bank.y.as_ptr());
        assert_eq!(ptrs.2, bank.e.as_ptr());
        assert_eq!(ptrs.3, bank.r.as_ptr());
        assert_eq!(ptrs.4, bank.b1.as_ptr());
        assert_eq!(ptrs.5, bank.b2.as_ptr());
        assert_eq!(ptrs.6, bank.gain.as_ptr());
        assert_eq!(caps.0, bank.x.capacity());
        assert_eq!(caps.1, bank.y.capacity());
        assert_eq!(caps.2, bank.e.capacity());
        assert_eq!(caps.3, bank.r.capacity());
        assert_eq!(caps.4, bank.b1.capacity());
        assert_eq!(caps.5, bank.b2.capacity());
        assert_eq!(caps.6, bank.gain.capacity());
    }

    #[test]
    fn new_rejects_invalid_sample_rate() {
        assert_eq!(
            ResonatorBank::new(0.0, 1).unwrap_err(),
            SynthError::InvalidSampleRate
        );
        assert_eq!(
            ResonatorBank::new(-1.0, 1).unwrap_err(),
            SynthError::InvalidSampleRate
        );
    }
}
