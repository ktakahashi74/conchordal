// core/erb.rs
// Utilities for ERB-rate scale conversion and uniform grid

/// Convert Hz to ERB-rate (Cam units, Glasberg & Moore 1990)
pub fn hz_to_erb(f_hz: f32) -> f32 {
    21.4 * (1.0 + 4.37 * f_hz / 1000.0).log10()
}

/// Convert ERB-rate (Cam) back to Hz
pub fn erb_to_hz(e_cam: f32) -> f32 {
    (10f32.powf(e_cam / 21.4) - 1.0) * 1000.0 / 4.37
}

/// ERB bandwidth in Hz (Glasberg & Moore 1990)
#[inline]
pub fn erb_bw_hz(f_hz: f32) -> f32 {
    24.7 * (4.37 * f_hz / 1000.0 + 1.0)
}


/// Construct a frequency vector (Hz) uniformly spaced in ERB-rate.
/// 
/// # Arguments
/// - `f_min`: minimum frequency in Hz
/// - `f_max`: maximum frequency in Hz
/// - `delta_e`: step size in ERB-rate (Cam units). Typical: 0.05
///
/// # Returns
/// Vector of frequencies in Hz, ascending.
pub fn erb_space(f_min: f32, f_max: f32, delta_e: f32) -> Vec<f32> {
    let e_min = hz_to_erb(f_min);
    let e_max = hz_to_erb(f_max);

    let mut freqs = Vec::new();
    let mut e = e_min;
    while e <= e_max {
        freqs.push(erb_to_hz(e));
        e += delta_e;
    }
    freqs
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_hz_erb() {
        let f = 1000.0;
        let e = hz_to_erb(f);
        let f2 = erb_to_hz(e);
        assert!(
            (f - f2).abs() < 1.0,
            "Round trip failed: f={} -> e={} -> f2={}",
            f,
            e,
            f2
        );
    }

    #[test]
    fn erb_space_monotonic() {
        let freqs = erb_space(100.0, 1000.0, 0.1);
        assert!(
            freqs.windows(2).all(|w| w[1] > w[0]),
            "erb_space not monotonic"
        );
    }

    #[test]
    fn erb_space_range() {
        let f_min = 100.0;
        let f_max = 1000.0;
        let freqs = erb_space(f_min, f_max, 0.1);
        assert!(
            *freqs.first().unwrap() >= f_min,
            "first freq too low: {}",
            freqs.first().unwrap()
        );
        assert!(
            *freqs.last().unwrap() <= f_max * 1.01,
            "last freq too high: {}",
            freqs.last().unwrap()
        );
    }
}
