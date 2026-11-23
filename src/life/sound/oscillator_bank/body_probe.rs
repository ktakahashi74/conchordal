use super::*;

impl OscillatorBank {
    pub(in crate::life::sound) fn reset_representative(&mut self) {
        self.x.fill(1.0);
        self.y.fill(0.0);
        self.rot_c.fill(1.0);
        self.rot_s.fill(0.0);
        self.gain_mask.fill(0.0);
        self.active_lane_len = 0;
        self.last_modes_len = 0;
        self.last_built_pitch_hz = 0.0;
        self.pending_phase_seed = None;
        if let Some(r) = self.attack.as_mut() {
            r.seed = 0;
            r.noise = Default::default();
        }
        self.clock = MotionClock::default();
        self.motion_tick = 0;
        self.motion_block = None;
        self.drive_env = 0.0;
        self.spectral_env = 0.0;
        self.pitch_counter = 0;
        self.norm_counter = 0;
        #[cfg(test)]
        {
            self.rebuild_count = 0;
            self.motion_refresh_count = 0;
        }
    }
}
