//! PCM self-removal before the shared nonlinear perception pipeline.
use super::analysis::AnalysisStream;
use crate::core::landscape::LandscapeParams;
#[cfg(feature = "native")]
use crate::handoff::Sender;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SourceKind {
    Voice,
    Flora,
    /// Rain impacts only; the Rain's resonant objects stay in the observed field.
    Rain,
}

pub(crate) struct SourcePcm {
    pub kind: SourceKind,
    pub site_id: u32,
    pub analyze_hr: bool,
    /// The Voice observer also owns an attack detector (`respond_to_onsets`).
    pub onset_history: bool,
    pub tonal_context: Option<bool>,
    pub id: u64,
    pub generation: u32,
    pub habitat: Vec<f32>,
}

pub(crate) struct SitePcm {
    pub id: u32,
    pub flora: bool,
    pub voice: bool,
    pub onset: bool,
    pub tonal: bool,
    pub samples: Vec<f32>,
}

pub(crate) type CaptureStorage = (Vec<SourcePcm>, Vec<SitePcm>);

pub(crate) struct PcmPacket {
    pub sites: Vec<SitePcm>,
    pub sources: Vec<SourcePcm>,
    #[cfg(feature = "native")]
    pub recycle: Sender<CaptureStorage>,
}

#[cfg(feature = "native")]
impl Drop for PcmPacket {
    fn drop(&mut self) {
        let _ = self.recycle.try_send((
            std::mem::take(&mut self.sources),
            std::mem::take(&mut self.sites),
        ));
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Observation {
    pub(crate) support: Option<Arc<crate::core::harmonic_support::Snapshot>>,
    pub id: u64,
    pub generation: u32,
    pub h_scan: Vec<f32>,
    pub r_scan: Vec<f32>,
    pub score_scan: Arc<[f32]>,
    /// Attack history without own PCM; `None` unless the source requested it.
    pub onset_habituation_state_scan: Option<Arc<[f32]>>,
    pub onset_habituation_end: u64,
    pub onset_habituation_known: bool,
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
pub(crate) struct TonalObservation {
    pub id: u64,
    pub generation: u32,
    pub shared: bool,
    pub excluded: bool,
    pub snapshot: Option<Arc<crate::core::tonal_context::Snapshot>>,
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

/// H/R of the habitat without one Rain's impacts, with its score and density mass.
#[derive(Clone, Debug)]
pub(crate) struct RainObservation {
    pub id: u64,
    pub h_scan: Vec<f32>,
    pub r_scan: Vec<f32>,
    pub score_scan: Vec<f32>,
    pub mass_scan: Vec<f32>,
}

impl RainObservation {
    pub(crate) fn revalue(
        &mut self,
        space: &crate::core::log2space::Log2Space,
        params: &LandscapeParams,
    ) {
        space.assert_scan_len_named(&self.h_scan, "rain_h_scan");
        space.assert_scan_len_named(&self.r_scan, "rain_r_scan");
        space.assert_scan_len_named(&self.score_scan, "rain_score_scan");
        space.assert_scan_len_named(&self.mass_scan, "rain_mass_scan");
        let kernel = params.consonance_field_kernel();
        for ((out, &h), &r) in self
            .score_scan
            .iter_mut()
            .zip(&self.h_scan)
            .zip(&self.r_scan)
        {
            *out = kernel.score(h, r);
        }
        crate::core::landscape::consonance_density_mass_scan(
            params,
            &self.h_scan,
            &self.r_scan,
            &mut self.mass_scan,
        );
    }
}

pub(crate) struct Observer {
    pub kind: SourceKind,
    pub id: u64,
    pub generation: u32,
    stream: Option<AnalysisStream>,
    power: Option<crate::core::nsgt_rt::RtNsgtKernelLog2>,
    pcm: Vec<f32>,
    tonal: Option<crate::core::tonal_context::Context>,
    tonal_ownership: Option<bool>,
    tonal_shared: bool,
}

impl Observer {
    pub(crate) fn new(
        id: u64,
        generation: u32,
        before_birth: &AnalysisStream,
        onset_history: bool,
    ) -> Self {
        Self {
            id,
            generation,
            kind: SourceKind::Voice,
            tonal: None,
            tonal_ownership: None,
            tonal_shared: false,
            stream: Some(before_birth.voice_observer(onset_history)),
            power: None,
            pcm: vec![0.; before_birth.hop_samples()],
        }
    }

    pub(crate) fn flora(
        id: u64,
        generation: u32,
        before_birth: &mut AnalysisStream,
        analyze_hr: bool,
        onset_history: bool,
    ) -> Self {
        Self {
            id,
            generation,
            kind: SourceKind::Flora,
            tonal: None,
            tonal_ownership: None,
            tonal_shared: false,
            pcm: vec![0.0; before_birth.hop_samples()],
            power: Some(before_birth.resource_observer()),
            stream: analyze_hr.then(|| before_birth.voice_observer(onset_history)),
        }
    }

    /// H/R observer of one Rain's impacts, built like a Voice observer.
    pub(crate) fn rain(id: u64, before_birth: &AnalysisStream) -> Self {
        Self {
            kind: SourceKind::Rain,
            ..Self::new(id, 0, before_birth, false)
        }
    }

    pub(crate) fn configure_tonal(
        &mut self,
        before: &AnalysisStream,
        ownership: Option<bool>,
        birth: bool,
    ) {
        if ownership == self.tonal_ownership {
            return;
        }
        self.tonal_ownership = ownership;
        self.tonal_shared = birth && ownership == Some(false);
        self.tonal = if ownership.is_some() && !self.tonal_shared {
            before.new_tonal_observer(birth)
        } else {
            None
        };
    }

    pub(crate) fn tonal_snapshot(&self) -> Option<TonalObservation> {
        self.tonal_ownership.map(|excluded| TonalObservation {
            id: self.id,
            generation: self.generation,
            shared: self.tonal_shared,
            excluded,
            snapshot: self.tonal.as_ref().and_then(|s| s.snapshot()),
        })
    }

    pub(crate) fn process_external(&mut self, full: &[f32], external: &[f32]) {
        self.process_input(full, external, true);
    }

    pub(crate) fn process(&mut self, mix: &[f32], own: &[f32]) {
        assert_eq!(mix.len(), self.pcm.len(), "LOO complete mix hop");
        assert_eq!(own.len(), self.pcm.len(), "LOO complete own hop");
        for ((out, &mix), &own) in self.pcm.iter_mut().zip(mix).zip(own) {
            *out = mix - own;
        }
        self.process_input(mix, own, false);
    }

    fn process_input(&mut self, mix: &[f32], own: &[f32], spatial: bool) {
        if spatial {
            self.pcm.copy_from_slice(own);
        }
        if let Some(tonal) = &mut self.tonal {
            tonal.process(if self.tonal_ownership == Some(true) {
                &self.pcm
            } else {
                mix
            });
        }
        if let Some(stream) = &mut self.stream {
            #[cfg(all(test, feature = "native", feature = "loo-diagnostic"))]
            if self.kind == SourceKind::Voice
                && crate::runtime::room_performance::oracle::mode() == "power"
            {
                stream.oracle_own(own, true);
                stream.process(mix);
                return;
            }
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

    pub(crate) fn gap_to(&mut self, sample: u64) {
        if let Some(tonal) = &mut self.tonal {
            tonal.invalidate();
        }
        if let Some(power) = &mut self.power {
            power.reset();
        }
        if let Some(stream) = &mut self.stream {
            stream.gap_to(sample);
        }
    }

    pub(crate) fn reset(&mut self) {
        if let Some(tonal) = &mut self.tonal {
            tonal.invalidate();
        }
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

    pub(crate) fn rain_snapshot(&self) -> RainObservation {
        let Observation {
            h_scan,
            r_scan,
            score_scan,
            ..
        } = self.snapshot();
        RainObservation {
            id: self.id,
            h_scan,
            r_scan,
            score_scan: score_scan.to_vec(),
            mass_scan: self
                .stream
                .as_ref()
                .unwrap()
                .last()
                .consonance_density_mass
                .clone(),
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
            support: frame.harmonic.support.clone(),
            id: self.id,
            generation: self.generation,
            h_scan,
            r_scan,
            score_scan,
            onset_habituation_state_scan: (!frame.onset_habituation_state_scan.is_empty())
                .then(|| frame.onset_habituation_state_scan.clone()),
            onset_habituation_end: frame.onset_habituation_end,
            onset_habituation_known: frame.onset_habituation_known,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tonal_birth_live_ownership_and_gap() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut mixed = AnalysisStream::new(params, nsgt).with_nsgt_power(false);
        mixed.enable_tonal_context(crate::core::tonal_context::Decoder::unqueried_for_test());
        let pcm = vec![0.01; mixed.hop_samples()];
        let own = vec![0.005; mixed.hop_samples()];
        for _ in 0..24 {
            mixed.process(&pcm);
        }
        let before = mixed.tonal_snapshot().unwrap();
        let mut observer = Observer::new(7, 2, &mixed, false);
        observer.configure_tonal(&mixed, Some(true), true);
        observer.process(&pcm, &own);
        let born = observer.tonal_snapshot().unwrap();
        assert!(born.excluded && !born.shared);
        assert_eq!(born.snapshot.unwrap().epoch_start, before.epoch_start);
        assert_eq!(mixed.tonal_snapshot().unwrap().end, before.end);
        mixed.process(&pcm);
        let start = mixed.tonal_snapshot().unwrap().end;
        observer.configure_tonal(&mixed, Some(false), false);
        assert!(observer.tonal_snapshot().unwrap().snapshot.is_none());
        for _ in 0..24 {
            observer.process(&pcm, &own);
        }
        let changed = observer.tonal_snapshot().unwrap();
        assert!(!changed.excluded && !changed.shared);
        assert_eq!(changed.snapshot.unwrap().epoch_start, start);
        observer.reset();
        for _ in 0..24 {
            observer.process(&pcm, &own);
        }
        assert!(observer.tonal_snapshot().unwrap().snapshot.is_none());
        observer.configure_tonal(&mixed, None, false);
        assert!(observer.tonal_snapshot().is_none());
    }

    #[test]
    #[ignore = "F-19b cost acquisition requires the machine timing lock"]
    fn flora_voice_observer_cost() {
        use crate::clock::Instant;
        use std::hint::black_box;
        let (params, kernel, hop) = crate::runtime::analysis_runtime_test_core(
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
        let mut voice = Observer::new(1, 0, &mixed, false);
        let mut flora = Observer::flora(2, 0, &mut mixed, true, false);
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
                    voice = Observer::new(1, 0, &mixed, false);
                    flora = Observer::flora(2, 0, &mut mixed, true, false);
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
        let mut observer = Observer::new(7, 2, &template, false);
        let mix: Vec<_> = (0..template.hop_samples())
            .map(|i| 0.2 * (i as f32 * 0.13).sin())
            .collect();
        let own: Vec<_> = mix.iter().map(|x| x * 0.3).collect();
        for _ in 0..64 {
            observer.process(&mix, &own);
        }
        let start = crate::clock::Instant::now();
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
        let mut observer = Observer::flora(7, 0, &mut mixed, true, false);
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

    /// Tone bursts (10 ms attack, 0.25 s hold, 50 ms release) every `period` seconds.
    fn bursts(hz: f64, offset: f64, period: f64, start: u64, out: &mut [f32], fs: f64) {
        for (i, x) in out.iter_mut().enumerate() {
            let t = (start + i as u64) as f64 / fs;
            let u = (t - offset).rem_euclid(period);
            let env = if t < offset {
                0.0
            } else if u < 0.01 {
                u / 0.01
            } else if u < 0.26 {
                1.0
            } else if u < 0.31 {
                (0.31 - u) / 0.05
            } else {
                0.0
            };
            *x += (0.05 * env * (std::f64::consts::TAU * hz * t).sin()) as f32;
        }
    }

    fn onset_mixed() -> AnalysisStream {
        let (nsgt, mut params) = crate::runtime::body_metabolism_test_core();
        params.habituation.enabled = true;
        let mut mixed = AnalysisStream::new(params, nsgt).with_nsgt_power(false);
        mixed.enable_onset_habituation(0.5);
        let silence = vec![0.0; mixed.hop_samples()];
        for _ in 0..400 {
            mixed.process(&silence);
        }
        assert!(mixed.last().onset_habituation_known);
        mixed
    }

    #[test]
    fn onset_history_observer_hears_everything_except_its_own_pcm() {
        let template = onset_mixed();
        let hop = template.hop_samples();
        let fs = f64::from(template.sample_rate());
        let space = template.last().space.clone();
        let near = |hz: f32| {
            let c = space.nearest_index(hz);
            c.saturating_sub(4)..(c + 5).min(space.n_bins())
        };
        for (name, other) in [
            ("own only", None),
            ("other pitch", Some((660.0, 0.5))),
            ("same pitch", Some((440.0, 0.5))),
        ] {
            let mut mixed = template.clone();
            let mut observer = Observer::new(7, 2, &mixed, true);
            // The observer starts from the mixed pre-birth history.
            let born = observer.snapshot();
            assert_eq!(
                born.onset_habituation_state_scan.as_deref(),
                Some(&*mixed.last().onset_habituation_state_scan)
            );
            assert!(born.onset_habituation_known);
            let (mut own, mut mix) = (vec![0.0; hop], vec![0.0; hop]);
            let (mut own_max, mut mixed_max) = (0.0_f32, 0.0_f32);
            for frame in 0..600_u64 {
                let start = (400 + frame) * hop as u64;
                own.fill(0.0);
                bursts(440.0, 0.0, 1.0, frame * hop as u64, &mut own, fs);
                mix.copy_from_slice(&own);
                if let Some((hz, offset)) = other {
                    bursts(hz, offset, 1.0, frame * hop as u64, &mut mix, fs);
                }
                observer.process(&mix, &own);
                mixed.process(&mix);
                let heard = observer.snapshot();
                assert_eq!(heard.onset_habituation_end, start + hop as u64);
                let scan = heard.onset_habituation_state_scan.unwrap();
                let region = if name == "other pitch" {
                    near(660.0)
                } else {
                    near(440.0)
                };
                own_max = own_max.max(scan[region.clone()].iter().copied().fold(0.0, f32::max));
                mixed_max = mixed_max.max(
                    mixed.last().onset_habituation_state_scan[region]
                        .iter()
                        .copied()
                        .fold(0.0, f32::max),
                );
            }
            println!("onset observer {name}: observer_max={own_max} mixed_max={mixed_max}");
            assert!(
                mixed_max > 0.1,
                "{name}: the mixed stream hears the attacks"
            );
            if other.is_none() {
                assert_eq!(own_max, 0.0, "{name}: own attacks are subtracted");
            } else {
                assert!(own_max > 0.1, "{name}: external attacks are heard");
            }
        }
    }

    #[test]
    fn onset_history_observer_reset_and_unrequested_observers_keep_their_contracts() {
        let mut template = onset_mixed();
        let hop = template.hop_samples();
        let fs = f64::from(template.sample_rate());
        let mut pcm = vec![0.0; hop];
        for frame in 0..200_u64 {
            pcm.fill(0.0);
            bursts(523.0, 0.0, 1.0, frame * hop as u64, &mut pcm, fs);
            template.process(&pcm);
        }
        assert!(
            template
                .last()
                .onset_habituation_state_scan
                .iter()
                .any(|&x| x > 0.0)
        );
        let silence = vec![0.0; hop];
        let mut observer = Observer::new(7, 2, &template, true);
        observer.process(&silence, &silence);
        observer.reset();
        let reset = observer.snapshot();
        // Reset follows the mixed detector: unknown, with the history cleared.
        assert!(!reset.onset_habituation_known);
        assert!(
            reset
                .onset_habituation_state_scan
                .is_none_or(|scan| scan.iter().all(|&x| x == 0.0))
        );
        let mut frames = 0;
        while !observer.snapshot().onset_habituation_known {
            observer.process(&silence, &silence);
            frames += 1;
            assert!(frames < 2000);
        }
        // Observers that did not request the detector are unchanged.
        let mut plain = Observer::new(7, 2, &template, false);
        let mut flora = Observer::flora(8, 0, &mut template, true, false);
        for observer in [&mut plain, &mut flora] {
            observer.process(&pcm, &silence);
            assert!(observer.last().onset_habituation_state_scan.is_empty());
            assert!(!observer.last().onset_habituation_known);
            assert!(observer.snapshot().onset_habituation_state_scan.is_none());
        }
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn onset_history_observer_steady_state_allocates_nothing() {
        let template = onset_mixed();
        let mut observer = Observer::new(7, 2, &template, true);
        let mix: Vec<_> = (0..template.hop_samples())
            .map(|i| 0.2 * (i as f32 * 0.13).sin())
            .collect();
        let own: Vec<_> = mix.iter().map(|x| x * 0.3).collect();
        for _ in 0..64 {
            observer.process(&mix, &own);
        }
        crate::runtime_profile::begin_allocations();
        for _ in 0..1024 {
            observer.process(&mix, &own);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
        assert!(observer.last().onset_habituation_known);
        println!("RESPOND_OBSERVER_ALLOC {counts:?}");
    }

    #[test]
    #[should_panic(expected = "loo_score_scan")]
    fn score_geometry_boundary_is_checked() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let template = AnalysisStream::new(params.clone(), nsgt);
        let mut observation = Observer::new(7, 0, &template, false).snapshot();
        observation.score_scan = vec![0.].into();
        observation.revalue(&template.last().space, &params);
    }
}
