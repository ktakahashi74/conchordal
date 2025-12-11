use rand::{Rng, SeedableRng};

/// Single-sample pink noise (Paul Kellet 3-pole filter).
pub fn pink_noise_tick<R: Rng + ?Sized>(rng: &mut R, b0: &mut f32, b1: &mut f32, b2: &mut f32) -> f32 {
    let white = rng.random_range(-1.0..1.0);
    *b0 = 0.99765 * *b0 + white * 0.099_046_0;
    *b1 = 0.96300 * *b1 + white * 0.296_516_4;
    *b2 = 0.57000 * *b2 + white * 1.052_691_3;
    let pink = *b0 + *b1 + *b2 + white * 0.1848;
    pink * 0.03
}

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
        out.push(pink_noise_tick(&mut rng, &mut b0, &mut b1, &mut b2));
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
