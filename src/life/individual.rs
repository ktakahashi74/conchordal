#[derive(Clone, Debug)]
pub struct PureTone {
    pub freq_hz: f32,
    pub amp: f32,
}

impl PureTone {
    pub fn new(freq_hz: f32, amp: f32) -> Self {
        Self { freq_hz, amp }
    }
}
