use super::AnalysisStream;
use crate::temporal_cognition::{
    observation::Tap,
    t2::{
        Frame,
        runtime_input::{Input, Output},
    },
};

#[derive(Default)]
pub(crate) struct AnalysisTemporalInput {
    tap: Option<Tap>,
    raw_output: Option<Output>,
}

pub(super) struct AnalysisTemporal {
    raw: Option<Input>,
    tap: Option<Tap>,
}

pub(super) struct AnalysisTemporalFrame(Option<Frame>);

impl AnalysisTemporalInput {
    #[cfg(feature = "native")]
    pub(crate) fn new(tap: Option<Tap>, raw_output: Option<Output>) -> Self {
        Self { tap, raw_output }
    }

    pub(super) fn start(self, stream: &AnalysisStream) -> AnalysisTemporal {
        let raw = (self.raw_output.is_some() || self.tap.as_ref().is_some_and(|t| t.t2_enabled))
            .then(|| {
                Input::new(
                    stream.sample_rate() as u32,
                    stream.hop_samples() as u64,
                    self.raw_output,
                )
                .expect("validated raw T2 geometry")
            });
        AnalysisTemporal { raw, tap: self.tap }
    }
}

impl AnalysisTemporal {
    pub(super) fn observe_raw(&mut self, frame: u64, hop: &[f32]) -> AnalysisTemporalFrame {
        AnalysisTemporalFrame(
            self.raw
                .as_mut()
                .and_then(|input| input.observe(frame, hop)),
        )
    }

    pub(super) fn observe_spectrum(
        &mut self,
        frame: u64,
        hop: &[f32],
        power: &[f32],
        t2: AnalysisTemporalFrame,
    ) {
        #[cfg(feature = "native")]
        if let Some(tap) = self.tap.as_mut() {
            let mono_energy =
                hop.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / hop.len() as f64;
            tap.observe_frame(frame, power, mono_energy, t2.0);
        }
        #[cfg(not(feature = "native"))]
        let _ = (frame, hop, power, t2.0);
    }

    pub(super) fn finish(&mut self) {
        drop(self.raw.take());
        drop(self.tap.take());
    }
}
