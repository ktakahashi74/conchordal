//! Compiled per-sample coefficients (e = 2*sin(theta/2), r from T60 in sec).

/// Coefficients used by the sample processing loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeCoeffs {
    /// Tuning coefficient: e = 2*sin(theta/2).
    pub e: f32,
    /// Per-sample decay 0..1.
    pub r: f32,
    /// Input coupling gain.
    pub b1: f32,
    /// Secondary coupling (reserved).
    pub b2: f32,
    /// Output gain.
    pub gain: f32,
}

impl ModeCoeffs {
    /// Safe zero coefficients.
    pub fn zero() -> Self {
        Self {
            e: 0.0,
            r: 0.0,
            b1: 0.0,
            b2: 0.0,
            gain: 0.0,
        }
    }
}
