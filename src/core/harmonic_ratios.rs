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
