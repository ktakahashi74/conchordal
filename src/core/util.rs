pub fn hz_to_log2(hz: f32, ref_hz: f32) -> f32 {
    (hz / ref_hz).log2()
}
