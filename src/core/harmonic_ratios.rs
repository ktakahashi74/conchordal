pub const HARMONIC_RATIOS: &[(u16, u16)] = &[
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 4),
    (6, 5),
    (5, 3),
    (8, 5),
];

#[inline]
pub fn ratio_to_f32((n, d): (u16, u16)) -> f32 {
    n as f32 / d as f32
}

pub fn fold_to_octave_near(freq: f32, base: f32, lo: f32, hi: f32) -> f32 {
    if !freq.is_finite() || !base.is_finite() || base <= 0.0 {
        return freq;
    }
    let mut f = freq;
    let lo = lo.max(1e-6);
    let hi = hi.max(lo);
    for _ in 0..64 {
        if f >= lo {
            break;
        }
        f *= 2.0;
    }
    for _ in 0..64 {
        if f <= hi {
            break;
        }
        f *= 0.5;
    }
    debug_assert!(f >= lo && f <= hi, "folded freq out of range");
    f
}
