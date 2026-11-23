use super::*;

impl ModalEngine {
    pub(in crate::life::sound) fn reset_representative(&mut self) {
        self.bank.reset_representative();
        self.coefficients_seeded = false;
        self.scratch.clear();
        self.last_built_pitch_hz = 0.0;
        self.counter = 0;
        self.last_modes_len = 0;
        self.pending_modal_phase_seed = None;
    }
}
