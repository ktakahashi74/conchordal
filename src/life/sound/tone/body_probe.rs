#[cfg(feature = "research")]
#[path = "metabolism_modulator.rs"]
mod metabolism_modulator;

use super::*;

impl Tone {
    /// Reset a probe of this same body without rebuilding its allocated backend.
    pub(crate) fn reset_representative(&mut self, template: &Self, pitch_hz: f32) {
        assert!(!template.started && template.pending_updates.is_empty());
        self.phrase = template.phrase;
        self.phrase_on = template.phrase_on;
        self.phrase_off = template.phrase_off;
        self.phrase_pitch_changed = template.phrase_pitch_changed;
        self.phrase_off_origin = template.phrase_off_origin;
        self.phrase_gain = template.phrase_gain;
        self.backend.reset_representative();
        self.prototype = template.prototype.clone();
        self.render_modulator = template.render_modulator.clone();
        self.pending_impulse_energy = template.pending_impulse_energy;
        self.envelope = template.envelope;
        self.planned_kick_pending = template.planned_kick_pending;
        self.pending_updates.clear();
        self.pending_trigger = template.pending_trigger;
        self.current_amp = template.current_amp;
        self.target_amp = template.target_amp;
        self.current_pitch_hz = pitch_hz;
        self.target_pitch_hz = pitch_hz;
        self.amp_tau_sec = template.amp_tau_sec;
        self.pitch_tau_sec = template.pitch_tau_sec;
        self.amp_alpha = template.amp_alpha;
        self.pitch_alpha = template.pitch_alpha;
        self.sample_dt = template.sample_dt;
        self.continuous_drive = template.continuous_drive;
        self.noise_state = template.noise_state;
        self.started = false;
        self.sine_impulse_boost = template.sine_impulse_boost;
    }
}
