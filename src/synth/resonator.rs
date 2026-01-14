//! Resonator bank using a damped Modified Coupled Form (Hz, sec).
//! x = r*(x - e*y) + b1*u, y = r*(e*x + y) + b2*u

use crate::synth::SynthError;
use crate::synth::modes::{ModeParams, compile_mode};
use std::f32::consts::PI;
#[cfg(feature = "simd-wide")]
use wide::{f32x4, f32x8};

// Use fused mul-add only when the SIMD path can also fuse, to keep outputs aligned.
#[cfg(any(target_feature = "fma", target_feature = "neon"))]
#[inline(always)]
fn mul_add_fast(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg(not(any(target_feature = "fma", target_feature = "neon")))]
#[inline(always)]
fn mul_add_fast(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

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
    theta: Vec<f32>,
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
        let mut theta = Vec::with_capacity(max_modes);
        theta.resize(max_modes, 0.0);

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
            theta,
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
            self.theta[i] = 2.0 * PI * params.freq_hz / self.fs;
        }

        Ok(())
    }

    /// Set mode parameters and compile coefficients, preserving existing state where possible.
    pub fn set_modes_preserve_state(&mut self, modes: &[ModeParams]) -> Result<(), SynthError> {
        if modes.len() > self.capacity {
            return Err(SynthError::TooManyModes {
                requested: modes.len(),
                capacity: self.capacity,
            });
        }

        let old_len = self.active_len;
        let new_len = modes.len();
        self.active_len = new_len;

        if new_len > old_len {
            for i in old_len..new_len {
                self.x[i] = 0.0;
                self.y[i] = 0.0;
            }
        }

        for (i, params) in modes.iter().enumerate() {
            let coeffs = compile_mode(params, self.fs);
            self.e[i] = coeffs.e;
            self.r[i] = coeffs.r;
            self.b1[i] = coeffs.b1;
            self.b2[i] = coeffs.b2;
            self.gain[i] = coeffs.gain;
            self.theta[i] = 2.0 * PI * params.freq_hz / self.fs;
        }

        Ok(())
    }

    /// Blog baseline: naive per-sample sin/cos rotation with damping.
    #[allow(dead_code)]
    fn process_sample_naive_sin(&mut self, u: f32) -> f32 {
        let mut out = 0.0;
        for i in 0..self.active_len {
            let (s, c) = self.theta[i].sin_cos();
            let r = self.r[i];
            let b1 = self.b1[i];
            let b2 = self.b2[i];
            let gain = self.gain[i];

            let x0 = self.x[i];
            let y0 = self.y[i];

            let rot_x = mul_add_fast(c, x0, -s * y0);
            let rot_y = mul_add_fast(s, x0, c * y0);
            let x1 = mul_add_fast(r, rot_x, b1 * u);
            let y1 = mul_add_fast(r, rot_y, b2 * u);

            self.x[i] = x1;
            self.y[i] = y1;

            out = mul_add_fast(gain, y1, out);
        }

        out
    }

    /// Reference scalar MCF update (no denormal flushing).
    #[allow(dead_code)]
    fn process_sample_ref(&mut self, u: f32) -> f32 {
        let mut out = 0.0;
        for i in 0..self.active_len {
            let e = self.e[i];
            let r = self.r[i];
            let b1 = self.b1[i];
            let b2 = self.b2[i];
            let gain = self.gain[i];

            let x0 = self.x[i];
            let y0 = self.y[i];

            let x1 = r * (x0 - e * y0) + b1 * u;
            let y1 = r * (e * x1 + y0) + b2 * u;

            self.x[i] = x1;
            self.y[i] = y1;

            out += gain * y1;
        }

        out
    }

    /// Blog baseline: scalar MCF update with bounds-checked slices.
    #[allow(dead_code)]
    fn process_sample_magic_scalar_basic(&mut self, u: f32) -> f32 {
        let n = self.active_len;
        let x = &mut self.x[..n];
        let y = &mut self.y[..n];
        let e = &self.e[..n];
        let r = &self.r[..n];
        let b1 = &self.b1[..n];
        let b2 = &self.b2[..n];
        let gain = &self.gain[..n];

        let mut out = 0.0;
        for i in 0..n {
            let x0 = x[i];
            let y0 = y[i];

            let x1 = mul_add_fast(r[i], mul_add_fast(-e[i], y0, x0), b1[i] * u);
            let y1 = mul_add_fast(r[i], mul_add_fast(e[i], x1, y0), b2[i] * u);

            x[i] = x1;
            y[i] = y1;

            out = mul_add_fast(gain[i], y1, out);
        }

        out
    }

    #[allow(dead_code)]
    fn process_sample_scalar(&mut self, u: f32) -> f32 {
        self.process_sample_magic_scalar_basic(u)
    }

    /// Scalar MCF update with unchecked indexing to remove bounds checks.
    #[allow(dead_code)]
    fn process_sample_magic_scalar_unsafe(&mut self, u: f32) -> f32 {
        let n = self.active_len;
        let x_ptr = self.x.as_mut_ptr();
        let y_ptr = self.y.as_mut_ptr();
        let e_ptr = self.e.as_ptr();
        let r_ptr = self.r.as_ptr();
        let b1_ptr = self.b1.as_ptr();
        let b2_ptr = self.b2.as_ptr();
        let gain_ptr = self.gain.as_ptr();

        let mut out = 0.0;

        // Safety: all arrays have length >= active_len, and we only access 0..n.
        unsafe {
            for i in 0..n {
                let x0 = *x_ptr.add(i);
                let y0 = *y_ptr.add(i);
                let e = *e_ptr.add(i);
                let r = *r_ptr.add(i);
                let b1 = *b1_ptr.add(i);
                let b2 = *b2_ptr.add(i);
                let gain = *gain_ptr.add(i);

                let x1 = mul_add_fast(r, mul_add_fast(-e, y0, x0), b1 * u);
                let y1 = mul_add_fast(r, mul_add_fast(e, x1, y0), b2 * u);

                *x_ptr.add(i) = x1;
                *y_ptr.add(i) = y1;

                out = mul_add_fast(gain, y1, out);
            }
        }

        out
    }

    /// Blog baseline: safe SIMD with f32x8 and slice copies.
    #[cfg(feature = "simd-wide")]
    fn process_sample_simd_wide8(&mut self, u: f32) -> f32 {
        let n = self.active_len;
        let n8 = n & !7;
        let x = &mut self.x[..n];
        let y = &mut self.y[..n];
        let e = &self.e[..n];
        let r = &self.r[..n];
        let b1 = &self.b1[..n];
        let b2 = &self.b2[..n];
        let gain = &self.gain[..n];

        let mut out = 0.0;
        let u_vec = f32x8::splat(u);

        for i in (0..n8).step_by(8) {
            let x0_arr: [f32; 8] = x[i..i + 8].try_into().unwrap();
            let y0_arr: [f32; 8] = y[i..i + 8].try_into().unwrap();
            let e_arr: [f32; 8] = e[i..i + 8].try_into().unwrap();
            let r_arr: [f32; 8] = r[i..i + 8].try_into().unwrap();
            let b1_arr: [f32; 8] = b1[i..i + 8].try_into().unwrap();
            let b2_arr: [f32; 8] = b2[i..i + 8].try_into().unwrap();

            let x0 = f32x8::from(x0_arr);
            let y0 = f32x8::from(y0_arr);
            let e_vec = f32x8::from(e_arr);
            let r_vec = f32x8::from(r_arr);
            let b1_vec = f32x8::from(b1_arr);
            let b2_vec = f32x8::from(b2_arr);

            let rot = e_vec.mul_add(-y0, x0);
            let x1_vec = r_vec.mul_add(rot, b1_vec * u_vec);
            let y1_vec = r_vec.mul_add(e_vec.mul_add(x1_vec, y0), b2_vec * u_vec);

            let x1_arr = x1_vec.to_array();
            let y1_arr = y1_vec.to_array();

            x[i..i + 8].copy_from_slice(&x1_arr);
            y[i..i + 8].copy_from_slice(&y1_arr);

            for k in 0..8 {
                out = mul_add_fast(gain[i + k], y1_arr[k], out);
            }
        }

        for i in n8..n {
            let x0 = x[i];
            let y0 = y[i];

            let x1 = mul_add_fast(r[i], mul_add_fast(-e[i], y0, x0), b1[i] * u);
            let y1 = mul_add_fast(r[i], mul_add_fast(e[i], x1, y0), b2[i] * u);

            x[i] = x1;
            y[i] = y1;

            out = mul_add_fast(gain[i], y1, out);
        }

        out
    }

    /// SIMD with f32x8 plus a single f32x4 tail (safe slices).
    #[cfg(feature = "simd-wide")]
    fn process_sample_simd_wide8_tail4(&mut self, u: f32) -> f32 {
        let n = self.active_len;
        let n8 = n & !7;
        let x = &mut self.x[..n];
        let y = &mut self.y[..n];
        let e = &self.e[..n];
        let r = &self.r[..n];
        let b1 = &self.b1[..n];
        let b2 = &self.b2[..n];
        let gain = &self.gain[..n];

        let mut out = 0.0;
        let u8 = f32x8::splat(u);
        let u4 = f32x4::splat(u);

        for i in (0..n8).step_by(8) {
            let x0_arr: [f32; 8] = x[i..i + 8].try_into().unwrap();
            let y0_arr: [f32; 8] = y[i..i + 8].try_into().unwrap();
            let e_arr: [f32; 8] = e[i..i + 8].try_into().unwrap();
            let r_arr: [f32; 8] = r[i..i + 8].try_into().unwrap();
            let b1_arr: [f32; 8] = b1[i..i + 8].try_into().unwrap();
            let b2_arr: [f32; 8] = b2[i..i + 8].try_into().unwrap();

            let x0 = f32x8::from(x0_arr);
            let y0 = f32x8::from(y0_arr);
            let e_vec = f32x8::from(e_arr);
            let r_vec = f32x8::from(r_arr);
            let b1_vec = f32x8::from(b1_arr);
            let b2_vec = f32x8::from(b2_arr);

            let rot = e_vec.mul_add(-y0, x0);
            let x1_vec = r_vec.mul_add(rot, b1_vec * u8);
            let y1_vec = r_vec.mul_add(e_vec.mul_add(x1_vec, y0), b2_vec * u8);

            let x1_arr = x1_vec.to_array();
            let y1_arr = y1_vec.to_array();

            x[i..i + 8].copy_from_slice(&x1_arr);
            y[i..i + 8].copy_from_slice(&y1_arr);

            for k in 0..8 {
                out = mul_add_fast(gain[i + k], y1_arr[k], out);
            }
        }

        let mut idx = n8;
        if n - n8 >= 4 {
            let x0_arr: [f32; 4] = x[idx..idx + 4].try_into().unwrap();
            let y0_arr: [f32; 4] = y[idx..idx + 4].try_into().unwrap();
            let e_arr: [f32; 4] = e[idx..idx + 4].try_into().unwrap();
            let r_arr: [f32; 4] = r[idx..idx + 4].try_into().unwrap();
            let b1_arr: [f32; 4] = b1[idx..idx + 4].try_into().unwrap();
            let b2_arr: [f32; 4] = b2[idx..idx + 4].try_into().unwrap();

            let x0 = f32x4::from(x0_arr);
            let y0 = f32x4::from(y0_arr);
            let e_vec = f32x4::from(e_arr);
            let r_vec = f32x4::from(r_arr);
            let b1_vec = f32x4::from(b1_arr);
            let b2_vec = f32x4::from(b2_arr);

            let rot = e_vec.mul_add(-y0, x0);
            let x1_vec = r_vec.mul_add(rot, b1_vec * u4);
            let y1_vec = r_vec.mul_add(e_vec.mul_add(x1_vec, y0), b2_vec * u4);

            let x1_arr = x1_vec.to_array();
            let y1_arr = y1_vec.to_array();

            x[idx..idx + 4].copy_from_slice(&x1_arr);
            y[idx..idx + 4].copy_from_slice(&y1_arr);

            for k in 0..4 {
                out = mul_add_fast(gain[idx + k], y1_arr[k], out);
            }

            idx += 4;
        }

        for i in idx..n {
            let x0 = x[i];
            let y0 = y[i];

            let x1 = mul_add_fast(r[i], mul_add_fast(-e[i], y0, x0), b1[i] * u);
            let y1 = mul_add_fast(r[i], mul_add_fast(e[i], x1, y0), b2[i] * u);

            x[i] = x1;
            y[i] = y1;

            out = mul_add_fast(gain[i], y1, out);
        }

        out
    }

    /// SIMD fast path with unsafe pointer access to reduce bounds checks/copies.
    #[cfg(feature = "simd-wide")]
    fn process_sample_simd_wide_fast(&mut self, u: f32) -> f32 {
        let n = self.active_len;
        let n8 = n & !7;
        let u8 = f32x8::splat(u);
        let u4 = f32x4::splat(u);

        let x_ptr = self.x.as_mut_ptr();
        let y_ptr = self.y.as_mut_ptr();
        let e_ptr = self.e.as_ptr();
        let r_ptr = self.r.as_ptr();
        let b1_ptr = self.b1.as_ptr();
        let b2_ptr = self.b2.as_ptr();
        let gain_ptr = self.gain.as_ptr();

        let mut out = 0.0;

        // Safety: all arrays have length >= active_len; loops only access valid lanes.
        unsafe {
            for i in (0..n8).step_by(8) {
                let x0_arr = std::ptr::read_unaligned(x_ptr.add(i) as *const [f32; 8]);
                let y0_arr = std::ptr::read_unaligned(y_ptr.add(i) as *const [f32; 8]);
                let e_arr = std::ptr::read_unaligned(e_ptr.add(i) as *const [f32; 8]);
                let r_arr = std::ptr::read_unaligned(r_ptr.add(i) as *const [f32; 8]);
                let b1_arr = std::ptr::read_unaligned(b1_ptr.add(i) as *const [f32; 8]);
                let b2_arr = std::ptr::read_unaligned(b2_ptr.add(i) as *const [f32; 8]);
                let gain_arr = std::ptr::read_unaligned(gain_ptr.add(i) as *const [f32; 8]);

                let x0 = f32x8::from(x0_arr);
                let y0 = f32x8::from(y0_arr);
                let e_vec = f32x8::from(e_arr);
                let r_vec = f32x8::from(r_arr);
                let b1_vec = f32x8::from(b1_arr);
                let b2_vec = f32x8::from(b2_arr);

                let rot = e_vec.mul_add(-y0, x0);
                let x1_vec = r_vec.mul_add(rot, b1_vec * u8);
                let y1_vec = r_vec.mul_add(e_vec.mul_add(x1_vec, y0), b2_vec * u8);

                let x1_arr = x1_vec.to_array();
                let y1_arr = y1_vec.to_array();

                std::ptr::write_unaligned(x_ptr.add(i) as *mut [f32; 8], x1_arr);
                std::ptr::write_unaligned(y_ptr.add(i) as *mut [f32; 8], y1_arr);

                for k in 0..8 {
                    out = mul_add_fast(gain_arr[k], y1_arr[k], out);
                }
            }

            let mut idx = n8;
            if n - n8 >= 4 {
                let x0_arr = std::ptr::read_unaligned(x_ptr.add(idx) as *const [f32; 4]);
                let y0_arr = std::ptr::read_unaligned(y_ptr.add(idx) as *const [f32; 4]);
                let e_arr = std::ptr::read_unaligned(e_ptr.add(idx) as *const [f32; 4]);
                let r_arr = std::ptr::read_unaligned(r_ptr.add(idx) as *const [f32; 4]);
                let b1_arr = std::ptr::read_unaligned(b1_ptr.add(idx) as *const [f32; 4]);
                let b2_arr = std::ptr::read_unaligned(b2_ptr.add(idx) as *const [f32; 4]);
                let gain_arr = std::ptr::read_unaligned(gain_ptr.add(idx) as *const [f32; 4]);

                let x0 = f32x4::from(x0_arr);
                let y0 = f32x4::from(y0_arr);
                let e_vec = f32x4::from(e_arr);
                let r_vec = f32x4::from(r_arr);
                let b1_vec = f32x4::from(b1_arr);
                let b2_vec = f32x4::from(b2_arr);

                let rot = e_vec.mul_add(-y0, x0);
                let x1_vec = r_vec.mul_add(rot, b1_vec * u4);
                let y1_vec = r_vec.mul_add(e_vec.mul_add(x1_vec, y0), b2_vec * u4);

                let x1_arr = x1_vec.to_array();
                let y1_arr = y1_vec.to_array();

                std::ptr::write_unaligned(x_ptr.add(idx) as *mut [f32; 4], x1_arr);
                std::ptr::write_unaligned(y_ptr.add(idx) as *mut [f32; 4], y1_arr);

                for k in 0..4 {
                    out = mul_add_fast(gain_arr[k], y1_arr[k], out);
                }

                idx += 4;
            }

            for i in idx..n {
                let x0 = *x_ptr.add(i);
                let y0 = *y_ptr.add(i);
                let e = *e_ptr.add(i);
                let r = *r_ptr.add(i);
                let b1 = *b1_ptr.add(i);
                let b2 = *b2_ptr.add(i);
                let gain = *gain_ptr.add(i);

                let x1 = mul_add_fast(r, mul_add_fast(-e, y0, x0), b1 * u);
                let y1 = mul_add_fast(r, mul_add_fast(e, x1, y0), b2 * u);

                *x_ptr.add(i) = x1;
                *y_ptr.add(i) = y1;

                out = mul_add_fast(gain, y1, out);
            }
        }

        out
    }

    /// Process a single sample with the damped MCF update.
    #[cfg(all(
        feature = "simd-wide",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    pub fn process_sample(&mut self, u: f32) -> f32 {
        self.process_sample_simd_wide_fast(u)
    }

    /// Process a single sample with the damped MCF update.
    #[cfg(not(all(
        feature = "simd-wide",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    pub fn process_sample(&mut self, u: f32) -> f32 {
        self.process_sample_magic_scalar_unsafe(u)
    }

    /// Bench hook: naive per-sample trig baseline.
    #[cfg(any(test, feature = "bench-hooks"))]
    pub fn process_block_mono_naive_sin(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (u, y) in input.iter().copied().zip(output.iter_mut()) {
            *y = self.process_sample_naive_sin(u);
        }
    }

    /// Bench hook: magic-cycle scalar basic.
    #[cfg(any(test, feature = "bench-hooks"))]
    pub fn process_block_mono_magic_scalar_basic(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (u, y) in input.iter().copied().zip(output.iter_mut()) {
            *y = self.process_sample_magic_scalar_basic(u);
        }
    }

    /// Bench hook: magic-cycle scalar unchecked.
    #[cfg(any(test, feature = "bench-hooks"))]
    pub fn process_block_mono_magic_scalar_unsafe(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (u, y) in input.iter().copied().zip(output.iter_mut()) {
            *y = self.process_sample_magic_scalar_unsafe(u);
        }
    }

    /// Bench hook: SIMD safe f32x8 (initial version).
    #[cfg(all(any(test, feature = "bench-hooks"), feature = "simd-wide"))]
    pub fn process_block_mono_simd_safe8(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (u, y) in input.iter().copied().zip(output.iter_mut()) {
            *y = self.process_sample_simd_wide8(u);
        }
    }

    /// Bench hook: SIMD f32x8 with a single f32x4 tail.
    #[cfg(all(any(test, feature = "bench-hooks"), feature = "simd-wide"))]
    pub fn process_block_mono_simd_tail4(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (u, y) in input.iter().copied().zip(output.iter_mut()) {
            *y = self.process_sample_simd_wide8_tail4(u);
        }
    }

    /// Bench hook: SIMD fast unsafe path.
    #[cfg(all(any(test, feature = "bench-hooks"), feature = "simd-wide"))]
    pub fn process_block_mono_simd_fast(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (u, y) in input.iter().copied().zip(output.iter_mut()) {
            *y = self.process_sample_simd_wide_fast(u);
        }
    }

    /// Process a mono block with the scalar backend (bench hook).
    #[cfg(any(test, feature = "bench-hooks"))]
    pub fn process_block_mono_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        self.process_block_mono_magic_scalar_basic(input, output);
    }

    /// Process a mono block with the SIMD backend (bench hook).
    #[cfg(all(any(test, feature = "bench-hooks"), feature = "simd-wide"))]
    pub fn process_block_mono_simd(&mut self, input: &[f32], output: &mut [f32]) {
        self.process_block_mono_simd_safe8(input, output);
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

        let input: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01).sin()).collect();
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
            bank.theta.as_ptr(),
        );
        let caps = (
            bank.x.capacity(),
            bank.y.capacity(),
            bank.e.capacity(),
            bank.r.capacity(),
            bank.b1.capacity(),
            bank.b2.capacity(),
            bank.gain.capacity(),
            bank.theta.capacity(),
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
        assert_eq!(ptrs.7, bank.theta.as_ptr());
        assert_eq!(caps.0, bank.x.capacity());
        assert_eq!(caps.1, bank.y.capacity());
        assert_eq!(caps.2, bank.e.capacity());
        assert_eq!(caps.3, bank.r.capacity());
        assert_eq!(caps.4, bank.b1.capacity());
        assert_eq!(caps.5, bank.b2.capacity());
        assert_eq!(caps.6, bank.gain.capacity());
        assert_eq!(caps.7, bank.theta.capacity());

        bank.set_modes(&modes_b).unwrap();
        assert_eq!(ptrs.0, bank.x.as_ptr());
        assert_eq!(ptrs.1, bank.y.as_ptr());
        assert_eq!(ptrs.2, bank.e.as_ptr());
        assert_eq!(ptrs.3, bank.r.as_ptr());
        assert_eq!(ptrs.4, bank.b1.as_ptr());
        assert_eq!(ptrs.5, bank.b2.as_ptr());
        assert_eq!(ptrs.6, bank.gain.as_ptr());
        assert_eq!(ptrs.7, bank.theta.as_ptr());
        assert_eq!(caps.0, bank.x.capacity());
        assert_eq!(caps.1, bank.y.capacity());
        assert_eq!(caps.2, bank.e.capacity());
        assert_eq!(caps.3, bank.r.capacity());
        assert_eq!(caps.4, bank.b1.capacity());
        assert_eq!(caps.5, bank.b2.capacity());
        assert_eq!(caps.6, bank.gain.capacity());
        assert_eq!(caps.7, bank.theta.capacity());
    }

    #[test]
    fn preserve_state_keeps_xy_for_overlapping_modes() {
        let mut bank = ResonatorBank::new(48_000.0, 8).unwrap();
        let modes_old: Vec<ModeParams> = (0..8)
            .map(|i| ModeParams {
                freq_hz: 110.0 + i as f32 * 10.0,
                t60_s: 0.2,
                gain: 1.0,
                in_gain: 1.0,
            })
            .collect();
        bank.set_modes(&modes_old).unwrap();
        for i in 0..8 {
            bank.x[i] = i as f32 + 0.5;
            bank.y[i] = -(i as f32) - 0.25;
        }

        let modes_new: Vec<ModeParams> = (0..6)
            .map(|i| ModeParams {
                freq_hz: 220.0 + i as f32 * 20.0,
                t60_s: 0.1,
                gain: 0.5,
                in_gain: 0.5,
            })
            .collect();
        bank.set_modes_preserve_state(&modes_new).unwrap();

        for i in 0..6 {
            assert_eq!(bank.x[i], i as f32 + 0.5);
            assert_eq!(bank.y[i], -(i as f32) - 0.25);
        }
    }

    #[test]
    fn preserve_state_initializes_new_modes() {
        let mut bank = ResonatorBank::new(48_000.0, 8).unwrap();
        let modes_old: Vec<ModeParams> = (0..4)
            .map(|i| ModeParams {
                freq_hz: 110.0 + i as f32 * 10.0,
                t60_s: 0.2,
                gain: 1.0,
                in_gain: 1.0,
            })
            .collect();
        bank.set_modes(&modes_old).unwrap();
        for i in 0..4 {
            bank.x[i] = (i + 1) as f32;
            bank.y[i] = -((i + 1) as f32);
        }

        let modes_new: Vec<ModeParams> = (0..8)
            .map(|i| ModeParams {
                freq_hz: 220.0 + i as f32 * 20.0,
                t60_s: 0.1,
                gain: 0.5,
                in_gain: 0.5,
            })
            .collect();
        bank.set_modes_preserve_state(&modes_new).unwrap();

        for i in 0..4 {
            assert_eq!(bank.x[i], (i + 1) as f32);
            assert_eq!(bank.y[i], -((i + 1) as f32));
        }
        for i in 4..8 {
            assert_eq!(bank.x[i], 0.0);
            assert_eq!(bank.y[i], 0.0);
        }
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

    #[test]
    fn magic_circle_matches_sine_formula() {
        use std::f32::consts::PI;

        let fs = 48_000.0;
        let freq = 440.0;
        let theta = 2.0 * PI * freq / fs;
        let e = 2.0 * (0.5 * theta).sin();
        let denom = (0.5 * theta).cos();

        let mut bank = ResonatorBank::new(fs, 8).unwrap();
        bank.active_len = 8;
        for i in 0..8 {
            bank.x[i] = 1.0;
            bank.y[i] = 0.0;
            bank.e[i] = e;
            bank.r[i] = 1.0;
            bank.b1[i] = 0.0;
            bank.b2[i] = 0.0;
            bank.gain[i] = if i == 0 { 1.0 } else { 0.0 };
        }

        for n in 1..=256 {
            let y = bank.process_sample(0.0);
            let expected = (n as f32 * theta).sin() / denom;
            let err = (y - expected).abs();
            assert!(err < 1.0e-6, "n={n} y={y} expected={expected}");
        }
    }

    #[test]
    fn default_matches_reference_with_tolerance() {
        let mut bank = ResonatorBank::new(48_000.0, 16).unwrap();
        let mut modes = Vec::with_capacity(10);
        for i in 0..10 {
            modes.push(ModeParams {
                freq_hz: 110.0 + i as f32 * 27.5,
                t60_s: 0.05 + i as f32 * 0.01,
                gain: 0.2 + i as f32 * 0.01,
                in_gain: 0.1 + i as f32 * 0.005,
            });
        }
        bank.set_modes(&modes).unwrap();

        let mut bank_ref = bank.clone();
        let mut bank_def = bank.clone();
        for i in 0..1024 {
            let mut u = (i as f32 * 0.01).sin();
            if i == 0 {
                u += 1.0;
            }
            if i % 127 == 0 {
                u += 0.25;
            }
            let y_ref = bank_ref.process_sample_ref(u);
            let y_def = bank_def.process_sample(u);
            if cfg!(any(target_feature = "fma", target_feature = "neon")) {
                let err = (y_ref - y_def).abs();
                assert!(err < 1.0e-5, "i={i} y_ref={y_ref} y_def={y_def}");
            } else {
                assert_eq!(y_ref, y_def);
            }
        }
    }
}
