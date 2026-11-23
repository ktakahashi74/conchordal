use super::*;

impl AnyBackend {
    pub(in crate::life::sound) fn reset_representative(&mut self) {
        match self {
            Self::Oscillator(bank) => bank.reset_representative(),
            Self::Resonator(engine) => engine.reset_representative(),
        }
    }
}
