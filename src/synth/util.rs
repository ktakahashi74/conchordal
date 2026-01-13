//! Small DSP helpers (denorm flush for audio-rate values).

const DENORM_THRESH: f32 = 1.0e-20;

/// Flush denormals and non-finite values to zero.
#[inline(always)]
pub fn flush_denorm(x: f32) -> f32 {
    if !x.is_finite() || x.abs() < DENORM_THRESH {
        0.0
    } else {
        x
    }
}
