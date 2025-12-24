//! dB conversion helpers with explicit power/amplitude semantics.
//! - Power uses 10*log10(·) and dB to ratio uses /10.
//! - Amplitude uses 20*log10(·) and dB to ratio uses /20.
//!   EPS_POWER is applied to avoid log10(0).

/// Minimum power floor for log conversions.
pub const EPS_POWER: f32 = 1e-20;

/// Convert dB to a power ratio.
pub fn db_to_power_ratio(db: f32) -> f32 {
    10.0_f32.powf(db / 10.0)
}

/// Convert dB to an amplitude ratio.
pub fn db_to_amp_ratio(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert power ratio to dB (10*log10).
pub fn power_to_db(p: f32) -> f32 {
    10.0 * (p.max(EPS_POWER)).log10()
}

/// Convert amplitude ratio to dB (20*log10).
pub fn amp_to_db(a: f32) -> f32 {
    20.0 * (a.max(EPS_POWER.sqrt())).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_to_power_ratio_basics() {
        assert!((db_to_power_ratio(0.0) - 1.0).abs() < 1e-4);
        assert!((db_to_power_ratio(10.0) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn db_to_amp_ratio_basics() {
        assert!((db_to_amp_ratio(0.0) - 1.0).abs() < 1e-4);
        assert!((db_to_amp_ratio(6.0206) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn power_to_db_basics() {
        assert!((power_to_db(1.0) - 0.0).abs() < 1e-4);
        assert!((power_to_db(10.0) - 10.0).abs() < 1e-4);
    }
}
