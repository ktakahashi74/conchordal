use super::*;

impl Voice {
    pub(crate) fn representative_body_recipe(
        &self,
        fs: f32,
        hold: Tick,
    ) -> crate::life::onset_footprint::Recipe {
        let modulator = crate::life::onset_footprint::representative_modulator(
            self.articulation
                .render_modulator_spec(self.phonation_engine.mode),
        );
        let modulator = match modulator {
            crate::life::sound::RenderModulatorSpec::DroneSway { sway_rate, .. } => {
                crate::life::sound::RenderModulatorSpec::DroneSway {
                    phase: 0.0,
                    sway_rate,
                }
            }
            other => other,
        };
        crate::life::onset_footprint::Recipe {
            renderer_phase3: false,
            body: self.body_snapshot(),
            freq_hz: self.body.base_freq_hz(),
            hold,
            adsr: self.voice_adsr,
            modulator,
            smoothing_tau_sec: Self::PHONATION_UPDATE_SMOOTH_TAU_SEC,
            fs,
        }
    }
}
