//! User-facing modal parameters (Hz, sec; theta = 2*pi*freq_hz/fs).

/// Parameters per mode in Hz and seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeParams {
    /// Mode frequency in Hz.
    pub freq_hz: f32,
    /// T60 decay time in seconds.
    pub t60_s: f32,
    /// Output gain contribution.
    pub gain: f32,
    /// Input coupling gain for u.
    pub in_gain: f32,
}

impl Default for ModeParams {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            t60_s: 1.0,
            gain: 0.0,
            in_gain: 0.0,
        }
    }
}
