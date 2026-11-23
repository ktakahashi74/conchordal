use super::*;

impl OscillatorBank {
    /// Keep native excitation separate while sharing the legacy f32 carrier kernel.
    /// The Tone stops oscillator radiation at effective Off before reaching this path.
    pub(crate) fn phase3_carrier(&mut self, pitch: f32, spectral_drive: f32, open: bool) -> f32 {
        if open {
            self.advance_spectral_env(spectral_drive);
            if self.pitch_counter == 0 || self.needs_pitch_state_refresh(pitch) {
                self.refresh_pitch_state(pitch);
                if self.spectral_damping_active {
                    self.apply_spectral_damping();
                }
            }
            if self.moving {
                self.refresh_motion_state(pitch);
            }
            self.pitch_counter = (self.pitch_counter + 1) % PITCH_REFRESH_PERIOD_SAMPLES;
        }
        self.process_sample()
    }
}
