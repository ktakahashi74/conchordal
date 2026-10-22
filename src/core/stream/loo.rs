//! PCM self-removal before the shared nonlinear perception pipeline.
use super::analysis::AnalysisStream;
use crate::core::landscape::{Landscape, LandscapeParams};
use crossbeam_channel::Sender;
use std::sync::Arc;

pub(crate) struct SourcePcm {
    pub id: u64,
    pub generation: u32,
    pub habitat: Vec<f32>,
}

pub(crate) struct PcmPacket {
    pub sources: Vec<SourcePcm>,
    pub recycle: Sender<Vec<SourcePcm>>,
}

impl Drop for PcmPacket {
    fn drop(&mut self) {
        let _ = self.recycle.try_send(std::mem::take(&mut self.sources));
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Observation {
    pub id: u64,
    pub generation: u32,
    pub h_scan: Vec<f32>,
    pub r_scan: Vec<f32>,
    pub score_scan: Arc<[f32]>,
}

impl Observation {
    pub(crate) fn revalue(
        &mut self,
        space: &crate::core::log2space::Log2Space,
        params: &LandscapeParams,
    ) {
        space.assert_scan_len_named(&self.h_scan, "loo_h_scan");
        space.assert_scan_len_named(&self.r_scan, "loo_r_scan");
        space.assert_scan_len_named(&self.score_scan, "loo_score_scan");
        let kernel = params.consonance_field_kernel();
        for ((out, &h), &r) in Arc::make_mut(&mut self.score_scan)
            .iter_mut()
            .zip(&self.h_scan)
            .zip(&self.r_scan)
        {
            *out = kernel.score(h, r);
        }
    }
}

pub(crate) struct Observer {
    pub id: u64,
    pub generation: u32,
    stream: AnalysisStream,
    pcm: Vec<f32>,
}

impl Observer {
    pub(crate) fn new(id: u64, generation: u32, before_birth: &AnalysisStream) -> Self {
        Self {
            id,
            generation,
            stream: before_birth.clone(),
            pcm: vec![0.; before_birth.hop_samples()],
        }
    }

    pub(crate) fn process(&mut self, mix: &[f32], own: &[f32]) -> &Landscape {
        assert_eq!(mix.len(), self.pcm.len(), "LOO complete mix hop");
        assert_eq!(own.len(), self.pcm.len(), "LOO complete own hop");
        for ((out, &mix), &own) in self.pcm.iter_mut().zip(mix).zip(own) {
            *out = mix - own;
        }
        self.stream.process(&self.pcm)
    }

    #[cfg(test)]
    pub(crate) fn last(&self) -> &Landscape {
        self.stream.last()
    }

    pub(crate) fn reset(&mut self) {
        self.stream.reset();
    }
    pub(crate) fn update(&mut self, update: crate::core::landscape::LandscapeUpdate) {
        self.stream.apply_update(update);
    }

    pub(crate) fn snapshot(&self) -> Observation {
        let frame = self.stream.last();
        let h_scan: Vec<_> = frame
            .harmonicity
            .iter()
            .map(|&x| crate::core::float::sanitize01(x))
            .collect();
        let r_scan = frame.roughness01.clone();
        let score_scan = h_scan
            .iter()
            .zip(&r_scan)
            .map(|(&h, &r)| frame.consonance_kernel.score(h, r))
            .collect();
        Observation {
            id: self.id,
            generation: self.generation,
            h_scan,
            r_scan,
            score_scan,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn pcm_loo_steady_state_allocates_nothing() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let template = AnalysisStream::new(params, nsgt);
        let mut observer = Observer::new(7, 2, &template);
        let mix: Vec<_> = (0..template.hop_samples())
            .map(|i| 0.2 * (i as f32 * 0.13).sin())
            .collect();
        let own: Vec<_> = mix.iter().map(|x| x * 0.3).collect();
        for _ in 0..64 {
            observer.process(&mix, &own);
        }
        let start = std::time::Instant::now();
        crate::runtime_profile::begin_allocations();
        for _ in 0..1024 {
            observer.process(&mix, &own);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
        assert!(observer.last().subjective_intensity.iter().any(|x| *x > 0.));
        println!(
            "L1_PCM_ALLOC {counts:?} mean_us={}",
            start.elapsed().as_secs_f64() * 1e6 / 1024.
        );
    }

    #[test]
    #[should_panic(expected = "loo_score_scan")]
    fn score_geometry_boundary_is_checked() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let template = AnalysisStream::new(params.clone(), nsgt);
        let mut observation = Observer::new(7, 0, &template).snapshot();
        observation.score_scan = vec![0.].into();
        observation.revalue(&template.last().space, &params);
    }
}
