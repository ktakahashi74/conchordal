//! Compile Hz/sec parameters into per-sample coefficients (freq_hz in Hz, t60_s in sec).

use std::f32::consts::PI;

use super::{ModeCoeffs, ModeParams};

/// Compile parameters into coefficients using:
/// theta = 2*pi*freq_hz/fs, e = 2*sin(theta/2), r = 10^(-3/(t60_s*fs)).
/// Returns `ModeCoeffs::zero()` when `fs` is non-finite or <= 0.
pub fn compile_mode(params: &ModeParams, fs: f32) -> ModeCoeffs {
    if !fs.is_finite() || fs <= 0.0 {
        debug_assert!(fs.is_finite() && fs > 0.0, "invalid sample rate");
        return ModeCoeffs::zero();
    }

    let fs_safe = fs;
    let freq_max = (fs_safe * 0.49).max(1.0);

    let freq_hz = clamp_finite(params.freq_hz, 1.0, freq_max, 1.0);
    let t60_s = clamp_finite(params.t60_s, 0.005, 120.0, 0.005);

    let gain = if params.gain.is_finite() { params.gain } else { 0.0 };
    let in_gain = if params.in_gain.is_finite() {
        params.in_gain
    } else {
        0.0
    };

    let theta = 2.0 * PI * freq_hz / fs_safe;
    let e = 2.0 * (0.5 * theta).sin();

    let r_raw = 10.0_f32.powf(-3.0 / (t60_s * fs_safe));
    let r = if r_raw.is_finite() {
        r_raw.clamp(0.0, 0.999_999_9)
    } else {
        0.0
    };

    ModeCoeffs {
        e,
        r,
        b1: in_gain,
        b2: 0.0,
        gain,
    }
}

fn clamp_finite(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if !value.is_finite() {
        return fallback;
    }
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_r_matches_t60() {
        let fs = 48_000.0;
        let t60_s = 1.0;
        let params = ModeParams {
            freq_hz: 440.0,
            t60_s,
            gain: 0.0,
            in_gain: 0.0,
        };
        let coeffs = compile_mode(&params, fs);
        let n = (fs * t60_s) as f32;
        let decay = coeffs.r.powf(n);
        let expected = 1.0e-3_f32;
        let rel_err = ((decay / expected) - 1.0).abs();
        assert!(rel_err < 1.0e-3, "decay={decay} rel_err={rel_err}");
    }

    #[test]
    fn compile_sanitizes_non_finite_params() {
        let fs = 48_000.0;
        let safe_freq = 1.0;
        let safe_t60 = 0.005;

        let theta = 2.0 * PI * safe_freq / fs;
        let expected_e = 2.0 * (0.5 * theta).sin();
        let expected_r = 10.0_f32
            .powf(-3.0 / (safe_t60 * fs))
            .clamp(0.0, 0.999_999_9);

        let cases = [
            ModeParams {
                freq_hz: f32::NAN,
                t60_s: f32::NAN,
                gain: f32::NAN,
                in_gain: f32::NAN,
            },
            ModeParams {
                freq_hz: f32::INFINITY,
                t60_s: f32::INFINITY,
                gain: f32::INFINITY,
                in_gain: f32::INFINITY,
            },
        ];

        for params in cases.iter() {
            let coeffs = compile_mode(params, fs);
            assert!(coeffs.e.is_finite());
            assert!(coeffs.r.is_finite());
            assert!(coeffs.b1.is_finite());
            assert!(coeffs.gain.is_finite());
            assert_eq!(coeffs.b1, 0.0);
            assert_eq!(coeffs.gain, 0.0);
            assert!((coeffs.e - expected_e).abs() < 1.0e-6);
            assert!((coeffs.r - expected_r).abs() < 1.0e-6);
        }
    }
}
