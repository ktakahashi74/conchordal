use super::*;

pub(crate) struct SpectralFrameView<'a> {
    pub subjective_intensity: &'a [f32],
}

impl AuditoryFrontEnd {
    // Unselected bodies still advance the filter and hop-statistic history.
    pub(crate) fn skip(&mut self, pcm: &[f32]) {
        self.bank
            .read_complex_hop(pcm.iter().map(|&x| Complex32::new(x, 0.0)));
        self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
    }
}
