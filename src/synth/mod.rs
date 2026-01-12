//! Synthesis primitives without semantic meaning (Hz, sec).

pub mod modes;
pub mod resonator;
pub mod util;

/// Errors returned by synth primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthError {
    /// Sample rate is non-finite or not positive.
    InvalidSampleRate,
    /// Provided modes exceed the bank capacity.
    TooManyModes { requested: usize, capacity: usize },
}
