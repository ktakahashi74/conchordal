pub fn hz_to_log2(hz: f32, ref_hz: f32) -> f32 {
    (hz / ref_hz).log2()
}

/// Generate sine wave samples
pub fn sine(fs: f32, f: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * f * (i as f32) / fs).sin())
        .collect()
}

/// Generate log-spaced values between start and stop (inclusive)
pub fn logspace(start: f32, stop: f32, num: usize) -> Vec<f32> {
    let log_start = start.ln();
    let log_stop = stop.ln();
    (0..num)
        .map(|i| {
            let t = i as f32 / (num - 1) as f32;
            (log_start + t * (log_stop - log_start)).exp()
        })
        .collect()
}
