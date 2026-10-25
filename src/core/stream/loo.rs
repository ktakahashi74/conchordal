//! PCM self-removal before the shared nonlinear perception pipeline.
use super::analysis::AnalysisStream;
use crate::core::landscape::LandscapeParams;
use crossbeam_channel::Sender;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceKind {
    Voice,
    Flora,
}

pub(crate) struct SourcePcm {
    pub kind: SourceKind,
    pub analyze_hr: bool,
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

#[derive(Clone, Debug)]
pub(crate) struct FloraObservation {
    pub id: u64,
    pub generation: u32,
    pub power_scan: Option<Vec<f32>>,
    pub perception: Option<Observation>,
    pub level_scan: Vec<f32>,
}

impl FloraObservation {
    pub(crate) fn revalue(
        &mut self,
        space: &crate::core::log2space::Log2Space,
        params: &LandscapeParams,
    ) {
        if let Some(perception) = &mut self.perception {
            perception.revalue(space, params);
            space.assert_scan_len_named(&self.level_scan, "flora_level_scan");
            for (level, &score) in self.level_scan.iter_mut().zip(perception.score_scan.iter()) {
                *level = params.consonance_representation.level(score);
            }
        }
    }
}

pub(crate) struct Observer {
    pub kind: SourceKind,
    pub id: u64,
    pub generation: u32,
    stream: Option<AnalysisStream>,
    power: Option<crate::core::nsgt_rt::RtNsgtKernelLog2>,
    pcm: Vec<f32>,
}

impl Observer {
    pub(crate) fn new(id: u64, generation: u32, before_birth: &AnalysisStream) -> Self {
        Self {
            id,
            generation,
            kind: SourceKind::Voice,
            stream: Some(before_birth.voice_observer()),
            power: None,
            pcm: vec![0.; before_birth.hop_samples()],
        }
    }

    pub(crate) fn flora(
        id: u64,
        generation: u32,
        before_birth: &mut AnalysisStream,
        analyze_hr: bool,
    ) -> Self {
        Self {
            id,
            generation,
            kind: SourceKind::Flora,
            pcm: vec![0.0; before_birth.hop_samples()],
            power: Some(before_birth.resource_observer()),
            stream: analyze_hr.then(|| before_birth.voice_observer()),
        }
    }

    pub(crate) fn process(&mut self, mix: &[f32], own: &[f32]) {
        assert_eq!(mix.len(), self.pcm.len(), "LOO complete mix hop");
        assert_eq!(own.len(), self.pcm.len(), "LOO complete own hop");
        for ((out, &mix), &own) in self.pcm.iter_mut().zip(mix).zip(own) {
            *out = mix - own;
        }
        if let Some(stream) = &mut self.stream {
            stream.process(&self.pcm);
        }
        if let Some(power) = &mut self.power {
            power.process_resource(&self.pcm);
        }
    }

    #[cfg(test)]
    pub(crate) fn last(&self) -> &crate::core::landscape::Landscape {
        self.stream.as_ref().unwrap().last()
    }

    pub(crate) fn reset(&mut self) {
        if let Some(stream) = &mut self.stream {
            stream.reset();
        }
        if let Some(power) = &mut self.power {
            power.reset();
        }
    }
    pub(crate) fn update(&mut self, update: crate::core::landscape::LandscapeUpdate) {
        if let Some(stream) = &mut self.stream {
            stream.apply_update(update);
        }
    }

    pub(crate) fn resource_snapshot(&self) -> FloraObservation {
        let power = self.power.as_ref().expect("Flora power observer");
        FloraObservation {
            id: self.id,
            generation: self.generation,
            power_scan: power.resource_power().map(<[f32]>::to_vec),
            perception: self.stream.as_ref().map(|_| self.snapshot()),
            level_scan: self
                .stream
                .as_ref()
                .map_or_else(Vec::new, |stream| vec![0.0; stream.last().space.n_bins()]),
        }
    }

    pub(crate) fn snapshot(&self) -> Observation {
        let frame = self.stream.as_ref().unwrap().last();
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
    #[ignore = "F-19b cost acquisition requires the machine timing lock"]
    fn flora_voice_observer_cost() {
        use std::hint::black_box;
        use std::time::Instant;
        let (params, kernel, hop) = crate::runtime::birth_surrogate_assay_core(
            &crate::config::AppConfig::default(),
            48_000,
        );
        let mut mixed = AnalysisStream::new(params.clone(), kernel);
        mixed.enable_resource_power();
        let mut mix: Vec<_> = (0..hop)
            .map(|i| 0.1 * (i as f32 * 0.13).sin() + 0.03 * (i as f32 * 0.17).sin())
            .collect();
        let mut own: Vec<_> = mix.iter().map(|x| 0.3 * x).collect();
        for _ in 0..64 {
            mixed.process(&mix);
        }
        let mut voice = Observer::new(1, 0, &mixed);
        let mut flora = Observer::flora(2, 0, &mut mixed, true);
        let input = std::env::var("F19B_INPUT").unwrap_or_else(|_| "repeated".into());
        let capture = std::env::var("F19B_PCM")
            .ok()
            .map(|path| std::fs::read(path).unwrap());
        let record_bytes = 4 + hop * 8;
        if let Some(bytes) = &capture {
            assert_eq!(bytes.len() % record_bytes, 0);
            mixed.reset();
        }
        let steps = capture.as_ref().map_or(1088, |b| b.len() / record_bytes);
        let mut active = false;
        let mut measured = 0;
        let mut times = [0.0; 9];
        let mut rows = Vec::new();
        for step in 0..steps {
            if let Some(bytes) = &capture {
                let record = &bytes[step * record_bytes..(step + 1) * record_bytes];
                for (sample, bytes) in mix
                    .iter_mut()
                    .chain(&mut own)
                    .zip(record[4..].as_chunks::<4>().0)
                {
                    *sample = f32::from_le_bytes(*bytes);
                }
                if record[0] == 0 {
                    mixed.process(&mix);
                    active = false;
                    continue;
                }
                if !active {
                    voice = Observer::new(1, 0, &mixed);
                    flora = Observer::flora(2, 0, &mut mixed, true);
                    active = true;
                }
            } else if input != "repeated" {
                for (i, (mix, own)) in mix.iter_mut().zip(&mut own).enumerate() {
                    let t = ((step + 64) * hop + i) as f32 / params.fs;
                    *mix = if input == "tail" && step > 128 {
                        0.0
                    } else {
                        0.1 * (std::f32::consts::TAU * 220.0 * t).sin()
                            + 0.03 * (std::f32::consts::TAU * 330.0 * t).sin()
                    };
                    *own = 0.3 * *mix;
                }
            }
            let mut elapsed = [0.0; 9];
            let start = Instant::now();
            voice.process(&mix, &own);
            elapsed[0] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            for ((out, &mix), &own) in flora.pcm.iter_mut().zip(&mix).zip(&own) {
                *out = mix - own;
            }
            elapsed[1] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            flora.stream.as_mut().unwrap().process(&flora.pcm);
            elapsed[2] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            flora.power.as_mut().unwrap().process_resource(&flora.pcm);
            elapsed[3] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut v = voice.snapshot();
            elapsed[4] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut f = flora.resource_snapshot();
            elapsed[5] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            v.revalue(&mixed.last().space, &params);
            elapsed[6] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            f.revalue(&mixed.last().space, &params);
            elapsed[7] = start.elapsed().as_secs_f64();
            let start = Instant::now();
            black_box(mixed.last().clone());
            elapsed[8] = start.elapsed().as_secs_f64();
            assert_eq!(voice.last().harmonicity, flora.last().harmonicity);
            assert_eq!(voice.last().roughness01, flora.last().roughness01);
            black_box((v, f));
            rows.push(serde_json::json!({"step":step,"elapsed_us":elapsed.map(|t|t*1e6),
                "density_subnormal":flora.last().subjective_intensity.iter().filter(|x|x.is_subnormal()).count(),
                "h_subnormal":flora.last().harmonicity.iter().filter(|x|x.is_subnormal()).count(),
                "r_subnormal":flora.last().roughness.iter().filter(|x|x.is_subnormal()).count()}));
            if capture.is_some() || step >= 64 {
                measured += 1;
                for (total, elapsed) in times.iter_mut().zip(elapsed) {
                    *total += elapsed;
                }
            }
            if capture.is_some() {
                mixed.process(&mix);
            }
        }
        assert!(measured > 0);
        if let Ok(path) = std::env::var("F19B_ROWS") {
            serde_json::to_writer(std::fs::File::create(path).unwrap(), &rows).unwrap();
        }
        println!(
            "F19B_COST {}",
            serde_json::json!({
                "input": input,
                "mean_us": times.map(|t| t * 1e6 / measured as f64),
                "measured_hops": measured,
                "stages": ["voice_process", "pcm_subtract", "flora_hr", "flora_power", "voice_snapshot", "flora_snapshot", "voice_revalue", "flora_revalue", "landscape_clone"],
                "hr_ratio": times[2] / times[0],
                "hop": hop,
            })
        );
    }

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
    #[cfg(feature = "profile-alloc")]
    fn flora_pcm_hr_and_power_steady_state_allocate_nothing() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut mixed = AnalysisStream::new(params, nsgt);
        let mut observer = Observer::flora(7, 0, &mut mixed, true);
        let mix: Vec<_> = (0..mixed.hop_samples())
            .map(|i| 0.03 * (i as f32 * 0.17).sin())
            .collect();
        let own: Vec<_> = mix.iter().map(|x| 0.3 * x).collect();
        for _ in 0..64 {
            observer.process(&mix, &own);
        }
        crate::runtime_profile::begin_allocations();
        for _ in 0..128 {
            observer.process(&mix, &own);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
        assert!(observer.last().harmonicity.iter().any(|&x| x > 0.0));
        assert!(observer.resource_snapshot().power_scan.is_some());
        println!("F19_OBSERVER_ALLOC {counts:?}");
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
