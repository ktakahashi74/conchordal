use super::*;

impl Voice {
    /// The representative recipe of I11-1 §4.2, frozen from what this Voice would
    /// emit now. `None` unless a participation clock configures onset comparison.
    pub(crate) fn footprint_recipe(&self, fs: f32) -> Option<crate::life::onset_footprint::Recipe> {
        let (hold_sec, adsr) = self.phonation_engine.footprint_hold()?;
        let phonation_mode = self.phonation_engine.mode;
        Some(crate::life::onset_footprint::Recipe {
            renderer_phase3: false,
            body: self.body_snapshot(),
            freq_hz: self.body.base_freq_hz(),
            hold: if matches!(phonation_mode, PhonationMode::Hold) {
                Tick::MAX
            } else {
                (f64::from(hold_sec) * f64::from(fs)).round() as Tick
            },
            adsr,
            modulator: crate::life::onset_footprint::representative_modulator(
                self.articulation.render_modulator_spec(phonation_mode),
            ),
            smoothing_tau_sec: Self::PHONATION_UPDATE_SMOOTH_TAU_SEC,
            fs,
        })
    }
}
