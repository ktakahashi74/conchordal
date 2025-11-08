use rand::{Rng, SeedableRng};

// --- noise generators ---
pub fn white_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Pink noise via simple 3-pole filter approximation (Voss–McCartney not required)
pub fn pink_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let white = rng.random_range(-1.0..1.0);
        // Paul Kellet 3-pole filter (approx −3 dB/oct)
        b0 = 0.99765 * b0 + white * 0.0990460;
        b1 = 0.96300 * b1 + white * 0.2965164;
        b2 = 0.57000 * b2 + white * 1.0526913;
        let pink = b0 + b1 + b2 + white * 0.1848;
        out.push((pink * 0.03) as f32); // normalize
    }
    out
}

/// Brown (red) noise via single-pole low-pass filter (−6 dB/oct)
pub fn brown_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut y = 0.0f32;
    let mut out = Vec::with_capacity(n);

    // Simple 1-pole IIR with pole near DC
    // y[n] = a*y[n-1] + (1 - a)*x[n]
    // choose a close to 1.0 to get strong low-frequency emphasis
    let a = 0.995; // ≈ −6 dB/oct

    for _ in 0..n {
        let white = rng.random_range(-1.0..1.0);
        y = a * y + (1.0 - a) * white;
        out.push(y);
    }
    out
}
