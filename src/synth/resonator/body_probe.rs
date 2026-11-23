use super::*;

impl ResonatorBank {
    pub(crate) fn reset_representative(&mut self) {
        self.active_len = 0;
        for buffer in [
            &mut self.x,
            &mut self.y,
            &mut self.e,
            &mut self.r,
            &mut self.b1,
            &mut self.b2,
            &mut self.gain,
        ] {
            buffer.fill(0.0);
        }
    }
}
