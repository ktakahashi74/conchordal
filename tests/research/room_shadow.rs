//! Two causal readings of the same pre-room habitat, on the renderer's worker thread.
use super::*;
use crate::core::habituation::HabituationField;
use crate::core::room::{RoomKernel, RoomStream, york_response};
use crate::core::stream::loo::{Observer, SourceKind, SourcePcm};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;

type Key = (u64, u32);
type Scans = [Vec<f32>; 2];
pub(crate) type Field = (String, Arc<[f32]>, Vec<f32>);

struct Reading {
    mix: AnalysisStream,
    h: HabituationField,
    observers: BTreeMap<Key, Observer>,
    last_peak: Option<(usize, f32)>,
    peak_changes: usize,
    positive_pairs: usize,
    peak_bin_travel: usize,
    amplitude_travel: f64,
}

impl Reading {
    fn new(core: &wiring::AnalysisRuntimeCore) -> Self {
        Self {
            mix: AnalysisStream::new(core.lparams.clone(), core.nsgt.clone()),
            h: HabituationField::new(
                &core.lparams.habituation,
                core.lparams.consonance_representation.theta,
                core.landscape.acoustic.space.n_bins(),
            ),
            observers: BTreeMap::new(),
            last_peak: None,
            peak_changes: 0,
            positive_pairs: 0,
            peak_bin_travel: 0,
            amplitude_travel: 0.,
        }
    }

    fn advance_mix(&mut self, pcm: &[f32], params: &LandscapeParams, dt: f32) {
        let land = self.mix.process(pcm);
        self.h.advance_from_channels(
            &land.acoustic.subjective_intensity,
            &land.acoustic.space,
            &params.harmonicity_kernel.params,
            dt,
        );
        let (bin, &amp) = land
            .acoustic
            .subjective_intensity
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        if let Some((prev, old)) = self.last_peak {
            self.amplitude_travel += f64::from((amp - old).abs());
            if old > 0. && amp > 0. {
                self.positive_pairs += 1;
                self.peak_changes += usize::from(prev != bin);
                self.peak_bin_travel += prev.abs_diff(bin);
            }
        }
        self.last_peak = Some((bin, amp));
    }
}

struct Shadow {
    readings: [Reading; 2],
    kernel: RoomKernel,
    bus: RoomStream,
    own: BTreeMap<Key, RoomStream>,
    wet_mix: Vec<f32>,
    wet_own: Vec<f32>,
    last: BTreeMap<Key, ([Scans; 2], u64)>,
    params: LandscapeParams,
    dt: f32,
    end: u64,
    next: u64,
    wet_live: usize,
}

impl Shadow {
    fn new() -> Self {
        let config =
            AppConfig::load_or_default(Some(&std::env::var("ROOM_CONFIG").unwrap())).unwrap();
        let core = build_analysis_runtime_core(&config, config.audio.sample_rate);
        let kernel = RoomKernel::new(&york_response(), core.hop);
        Self {
            readings: std::array::from_fn(|_| Reading::new(&core)),
            bus: kernel.stream_pool(1, 1).unwrap().pop().unwrap(),
            kernel,
            own: BTreeMap::new(),
            wet_mix: vec![0.; core.hop],
            wet_own: vec![0.; core.hop],
            last: BTreeMap::new(),
            params: core.lparams,
            dt: core.hop_duration.as_secs_f32(),
            end: std::env::var("ROOM_END").unwrap().parse().unwrap(),
            next: 0,
            wet_live: usize::from(
                config
                    .room
                    .as_ref()
                    .is_some_and(|r| r.mode == crate::config::RoomMode::Both),
            ),
        }
    }

    fn advance(&mut self, tick: u64, mix: &[f32], sources: &[SourcePcm]) {
        if tick >= self.end {
            return;
        }
        assert_eq!(tick, self.next);
        self.next += mix.len() as u64;
        self.bus.process(mix, &mut self.wet_mix);
        let voices = || sources.iter().filter(|s| s.kind == SourceKind::Voice);
        let live = |k: &Key| voices().any(|s| (s.id, s.generation) == *k);
        self.own.retain(|k, _| live(k));
        self.last.retain(|k, _| live(k));
        for reading in &mut self.readings {
            reading.observers.retain(|k, _| live(k));
        }
        for s in voices() {
            let key = (s.id, s.generation);
            self.own
                .entry(key)
                .or_insert_with(|| self.kernel.stream_pool(1, 1).unwrap().pop().unwrap())
                .process(&s.habitat, &mut self.wet_own);
            for (i, reading) in self.readings.iter_mut().enumerate() {
                let (bus, own) = if i == 0 {
                    (mix, s.habitat.as_slice())
                } else {
                    (self.wet_mix.as_slice(), self.wet_own.as_slice())
                };
                reading
                    .observers
                    .entry(key)
                    .or_insert_with(|| Observer::new(s.id, s.generation, &reading.mix, false))
                    .process(bus, own);
            }
        }
        self.readings[0].advance_mix(mix, &self.params, self.dt);
        self.readings[1].advance_mix(&self.wet_mix, &self.params, self.dt);
        if self.next == self.end {
            let stats: Vec<_> = self
                .readings
                .iter()
                .map(|r| {
                    json!([
                        r.positive_pairs,
                        r.peak_changes,
                        r.peak_bin_travel,
                        r.amplitude_travel
                    ])
                })
                .collect();
            std::fs::write(PathBuf::from(std::env::var("ROOM_OUTPUT").unwrap()).join("speed.json"),json!({"hops":self.end/mix.len() as u64,"bins_per_oct":self.readings[0].mix.last().acoustic.space.bins_per_oct,"dry_wet":stats}).to_string()).unwrap();
        }
    }

    fn fields(&mut self, key: Key, tick: u64, land: &Landscape) -> Option<Vec<Field>> {
        if tick >= self.end {
            return None;
        }
        // A lookahead decision can schedule an onset in a later audio hop.
        assert_eq!(land.harmonic.now_sample, self.next);
        let now: [Scans; 2] = std::array::from_fn(|i| {
            let obs = self.readings[i].observers[&key].snapshot();
            [obs.h_scan, obs.r_scan]
        });
        let live = &self.readings[self.wet_live];
        assert_eq!(live.h.state(), land.acoustic.perc_habituation_state_scan);
        let native = land
            .ecology
            .loo
            .iter()
            .find(|o| (o.id, o.generation) == key)
            .unwrap();
        assert_eq!(now[self.wet_live][0], native.h_scan);
        assert_eq!(now[self.wet_live][1], native.r_scan);
        let mut out = Vec::new();
        let mut add =
            |scans: &[Scans; 2], hi: usize, ri: usize, hab: usize, zero: bool, label: String| {
                let mut params = self.params.clone();
                if zero {
                    params.set_roughness_aversion(0.);
                }
                let k = params.consonance_field_kernel();
                let raw = scans[hi][0]
                    .iter()
                    .zip(&scans[ri][1])
                    .map(|(&h, &r)| k.score(h, r))
                    .collect::<Vec<_>>();
                out.push((label, Arc::from(raw), self.readings[hab].h.state().to_vec()));
            };
        for h in 0..2 {
            for r in 0..2 {
                for a in 0..2 {
                    add(&now, h, r, a, false, format!("{h}{r}{a}"));
                    if h == r {
                        add(&now, h, r, a, true, format!("zero{h}{r}{a}"));
                    }
                }
            }
        }
        if let Some((old, _)) = self.last.get(&key) {
            for i in 0..2 {
                add(old, i, i, i, false, format!("prev{i}{i}{i}"));
            }
        }
        self.last.insert(key, (now, tick));
        Some(out)
    }
}

thread_local! { static SHADOW: RefCell<Option<Shadow>> = const { RefCell::new(None) }; }

pub(crate) fn observe_dry(tick: Tick, mix: &[f32], sources: &[SourcePcm]) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("ROOM_RESCORE").is_some()) {
        SHADOW.with_borrow_mut(|slot| {
            slot.get_or_insert_with(Shadow::new)
                .advance(tick, mix, sources)
        });
    }
}

pub(crate) fn room_fields(
    id: u64,
    generation: u32,
    tick: Tick,
    land: &Landscape,
) -> Option<Vec<Field>> {
    if super::oracle::mode() == "fixed" {
        return super::oracle::fields(id, generation, land);
    }
    SHADOW.with_borrow_mut(|slot| slot.as_mut()?.fields((id, generation), tick, land))
}
