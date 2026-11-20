//! Native research only. Runs are isolated processes, one mode per process.
use super::*;
use crate::core::room::{RoomKernel, RoomStream, york_response};
use crate::core::stream::loo::{SourceKind, SourcePcm};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

type Owner = (u64, u32);
type ToneId = (u64, u32, u64);

pub(crate) fn mode() -> &'static str {
    static MODE: OnceLock<String> = OnceLock::new();
    MODE.get_or_init(|| std::env::var("ROOM_ORACLE").unwrap_or_default())
}

pub(crate) fn active() -> bool {
    !mode().is_empty()
}
pub(crate) fn waveform() -> bool {
    matches!(mode(), "off-boundary" | "release-boundary")
}

static REJECT: Mutex<[f64; 4]> = Mutex::new([0.; 4]);
pub(crate) fn rejections(values: [f64; 4]) {
    for (a, b) in REJECT.lock().unwrap().iter_mut().zip(values) {
        *a += b;
    }
}
pub(super) fn save_rejections(dir: &std::path::Path) {
    if active() {
        std::fs::write(
            dir.join("guard.json"),
            json!(*REJECT.lock().unwrap()).to_string(),
        )
        .unwrap();
    }
}

struct Fixed {
    mixed: AnalysisStream,
    observers: BTreeMap<Owner, AnalysisStream>,
    end: Tick,
}

impl Fixed {
    fn new() -> Self {
        let config =
            AppConfig::load_or_default(Some(&std::env::var("ROOM_CONFIG").unwrap())).unwrap();
        let core = build_analysis_runtime_core(&config, config.audio.sample_rate);
        Self {
            mixed: AnalysisStream::new(core.lparams, core.nsgt),
            observers: BTreeMap::new(),
            end: 0,
        }
    }

    fn advance(&mut self, now: Tick, mix: &[f32], sources: &[SourcePcm]) {
        assert_eq!(self.end, now);
        self.end += mix.len() as u64;
        self.observers.retain(|&(id, g), _| {
            sources
                .iter()
                .any(|s| s.kind == SourceKind::Voice && (s.id, s.generation) == (id, g))
        });
        for s in sources.iter().filter(|s| s.kind == SourceKind::Voice) {
            let stream = self
                .observers
                .entry((s.id, s.generation))
                .or_insert_with(|| self.mixed.voice_observer(false));
            stream.oracle_own(&s.habitat, true);
            stream.process(mix);
        }
        self.mixed.process(mix);
    }
}

thread_local! {
    static SOUNDING: RefCell<(Tick,BTreeMap<Owner,Vec<f32>>)> = const {RefCell::new((0,BTreeMap::new()))};
    static FIXED: RefCell<Option<Fixed>> = const { RefCell::new(None) };
    static WAVES: RefCell<Option<Waves>> = const { RefCell::new(None) };
}

pub(crate) fn count_pcm(owner: Owner, now: Tick, pcm: &[f32]) {
    SOUNDING.with_borrow_mut(|(tick, sums)| {
        if *tick != now {
            sums.clear();
            *tick = now;
        }
        let sum = sums.entry(owner).or_insert_with(|| vec![0.; pcm.len()]);
        for (a, b) in sum.iter_mut().zip(pcm) {
            *a += b;
        }
    });
}

pub(crate) fn retiring(live: &std::collections::BTreeSet<Owner>, now: Tick) -> usize {
    SOUNDING.with_borrow(|(tick, sums)| {
        if *tick != now {
            return 0;
        }
        sums.iter()
            .filter(|(key, pcm)| !live.contains(key) && pcm.iter().any(|&x| x != 0.))
            .count()
    })
}

pub(super) fn fields(id: u64, g: u32, land: &Landscape) -> Option<Vec<room_shadow::Field>> {
    FIXED.with_borrow(|slot| {
        let fixed = slot.as_ref()?;
        let native = land.loo.iter().find(|s| (s.id, s.generation) == (id, g))?;
        assert_eq!(fixed.end, land.harmonic_now_sample);
        let frame = fixed.observers[&(id, g)].last();
        let score = frame
            .harmonicity
            .iter()
            .zip(&frame.roughness01)
            .map(|(&h, &r)| {
                land.consonance_kernel
                    .score(crate::core::float::sanitize01(h), r)
            })
            .collect();
        let hab = &land.perc_habituation_state_scan;
        Some(vec![
            ("strict".into(), native.score_scan.clone(), hab.clone()),
            ("power".into(), score, hab.clone()),
        ])
    })
}

struct Waves {
    kernel: RoomKernel,
    streams: BTreeMap<ToneId, (Tick, RoomStream)>,
    own: BTreeMap<Owner, Vec<f32>>,
    tick: Option<Tick>,
    wet: Vec<f32>,
}

impl Waves {
    fn new(hop: usize) -> Self {
        Self {
            kernel: RoomKernel::new(&york_response(), hop),
            streams: BTreeMap::new(),
            own: BTreeMap::new(),
            tick: None,
            wet: vec![0.; hop],
        }
    }

    fn begin(&mut self, now: Tick) {
        if self.tick != Some(now) {
            self.own.clear();
            self.streams.retain(|_, (end, _)| *end >= now);
            self.tick = Some(now);
        }
    }
}

pub(crate) fn tone(key: ToneId, now: Tick, pcm: &[f32], on: Tick, off: Tick, release: Tick) {
    if !waveform() {
        return;
    }
    WAVES.with_borrow_mut(|slot| {
        let waves = slot.get_or_insert_with(|| Waves::new(pcm.len()));
        waves.begin(now);
        let close = if mode() == "off-boundary" {
            off
        } else {
            release
        };
        let all = waves
            .own
            .entry((key.0, key.1))
            .or_insert_with(|| vec![0.; pcm.len()]);
        // When g is zero the rest of this tone's FIR can no longer be read.
        if now < close {
            let (end, stream) = waves
                .streams
                .entry(key)
                .or_insert_with(|| (now, waves.kernel.stream_pool(1, 1).unwrap().pop().unwrap()));
            *end = now + pcm.len() as Tick;
            stream.process(pcm, &mut waves.wet);
            for (i, a) in all.iter_mut().enumerate() {
                *a += if (on..close).contains(&(now + i as Tick)) {
                    waves.wet[i]
                } else {
                    pcm[i]
                };
            }
        } else {
            for (a, b) in all.iter_mut().zip(pcm) {
                *a += b;
            }
        }
    });
}

pub(crate) fn wet(now: Tick, mix: &[f32], sources: &[SourcePcm]) {
    if mode() == "fixed" {
        FIXED.with_borrow_mut(|slot| {
            slot.get_or_insert_with(Fixed::new)
                .advance(now, mix, sources)
        });
    }
    if waveform() {
        WAVES.with_borrow_mut(|slot| slot.get_or_insert_with(|| Waves::new(mix.len())).begin(now));
    }
}

pub(crate) fn copy(owner: Owner, pcm: &mut [f32]) {
    WAVES.with_borrow(|slot| {
        if let Some(own) = slot.as_ref().unwrap().own.get(&owner) {
            pcm.copy_from_slice(own);
        } else {
            pcm.fill(0.);
        }
    })
}

#[test]
#[ignore = "offline upper bound on measured wet self power"]
fn isolated_tones() {
    use crate::life::voice::pitch_controller::OnsetResponse;
    let out = PathBuf::from(std::env::var("ROOM_OUTPUT").unwrap());
    let mut file = BufWriter::new(File::create(out.join("tones.tsv")).unwrap());
    writeln!(file,"other_hz\tobserver\thop\tmeasure\tpeak\tfirst_hz\tsecond_hz\tthird_hz\town_value\trise_sum\textra_sec").unwrap();
    let mut attacks = BufWriter::new(File::create(out.join("attacks.tsv")).unwrap());
    writeln!(attacks, "other_hz\tobserver\thop\tchannel").unwrap();
    let (nsgt, mut params) = crate::runtime::body_metabolism_test_core();
    params.habituation.enabled = true;
    let mut template = AnalysisStream::new(params, nsgt).with_nsgt_power(false);
    template.enable_onset_habituation(0.5);
    let hop = template.hop_samples();
    let silence = vec![0.; hop];
    for _ in 0..400 {
        template.process(&silence);
    }
    let kernel = RoomKernel::new(&york_response(), hop);
    let mut response_control_active = false;
    for other in [0., 440., 446., 660.] {
        let mut streams = kernel.stream_pool(2, 2).unwrap();
        let mut observers = std::array::from_fn::<_, 3, _>(|_| template.voice_observer(true));
        let mut response = std::array::from_fn::<_, 3, _>(|_| {
            OnsetResponse::new(16., template.last().space.n_bins())
        });
        let mut own = vec![0.; hop];
        let mut mix = own.clone();
        let mut wet = own.clone();
        let mut self_wet = own.clone();
        let mut ext = own.clone();
        for k in 0..800 {
            for i in 0..hop {
                let t = (k * hop + i) as f64 / 48000.;
                own[i] = if t >= 0.5 {
                    (0.08 * (std::f64::consts::TAU * 440. * t).sin()) as f32
                } else {
                    0.
                };
                let added = if other > 0. && t >= 2. {
                    (0.004 * (std::f64::consts::TAU * other * t + 0.73).sin()) as f32
                } else {
                    0.
                };
                mix[i] = own[i] + added;
            }
            streams[0].process(&mix, &mut wet);
            streams[1].process(&own, &mut self_wet);
            for ((v, m), s) in ext.iter_mut().zip(&wet).zip(&self_wet) {
                *v = m - s;
            }
            for (idx, obs) in observers.iter_mut().enumerate() {
                let frame = if idx == 0 {
                    obs.process(&ext)
                } else {
                    obs.oracle_own(&self_wet, idx == 1);
                    obs.process(&wet)
                };
                response[idx].oracle_step(frame, hop as f32 / 48000.);
                response_control_active |= idx == 0 && other > 0. && response[idx].extra_sec > 0.;
                let hz = &frame.space.centers_hz;
                for (name, scan) in [
                    ("S6", &*frame.subjective_intensity),
                    ("H", &*frame.harmonicity),
                    ("R", &*frame.roughness01),
                    ("onset", &*frame.onset_habituation_state_scan),
                ] {
                    let mut peaks: Vec<_> = (0..scan.len())
                        .filter(|&i| {
                            scan[i] > 0.
                                && (i == 0 || scan[i] > scan[i - 1])
                                && (i + 1 == scan.len() || scan[i] >= scan[i + 1])
                        })
                        .collect();
                    peaks.sort_by(|&a, &b| scan[b].total_cmp(&scan[a]));
                    let pitch = |j| peaks.get(j).map_or(0., |&i: &usize| hz[i]);
                    let peak = peaks.first().map_or(0., |&i| scan[i]);
                    if other == 0. {
                        assert_eq!(peak, 0.);
                    }
                    writeln!(
                        file,
                        "{other}\t{}\t{k}\t{name}\t{peak}\t{}\t{}\t{}\t{}\t{}\t{}",
                        ["strict", "guard", "unguarded"][idx],
                        pitch(0),
                        pitch(1),
                        pitch(2),
                        scan[frame.space.nearest_index(440.)],
                        response[idx].rise_sum,
                        response[idx].extra_sec
                    )
                    .unwrap();
                }
                for channel in obs.oracle_attacks() {
                    assert_ne!(other, 0., "attack during own-only condition");
                    writeln!(
                        attacks,
                        "{other}\t{}\t{k}\t{channel}",
                        ["strict", "guard", "unguarded"][idx]
                    )
                    .unwrap();
                }
            }
        }
    }
    assert!(
        response_control_active,
        "weak companion must drive the strict onset clock"
    );
    save_rejections(&out);
}
