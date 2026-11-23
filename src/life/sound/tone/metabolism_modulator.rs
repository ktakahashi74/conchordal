use super::*;

impl Tone {
    pub(crate) fn set_representative_modulator(&mut self, spec: RenderModulatorSpec) {
        assert!(!self.started);
        self.render_modulator = Some(RenderModulator::from_spec(spec));
    }
}
