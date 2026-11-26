use super::*;
use crate::config::RoomMode;
use crate::core::{
    landscape::Landscape,
    log2space::Log2Space,
    room::{york_response, york_right_response},
};
use crate::life::{
    flora::Flora,
    rain::Rain,
    sound::{BodyKind, ToneAdsr},
};
use crate::scenario::{FieldTarget, FloraSeeding, FloraSpec, RainSpec, Site};

const FS: f32 = 48000.;
const HOP: usize = 512;
const TIME: Timebase = Timebase { fs: FS, hop: HOP };

fn site(y: f64) -> Site {
    Site { y, ..Site::DEFAULT }
}

fn reflection(stem: &[f64], ir: &[f32], n: usize) -> f64 {
    stem[..n]
        .iter()
        .rev()
        .zip(&ir[1..])
        .map(|(s, &h)| s * f64::from(h))
        .sum::<f64>()
        / 23.5
}

fn close(got: f32, expected: f64) {
    assert!(
        (f64::from(got) - expected).abs() < 1e-5,
        "{got} vs {expected}"
    );
}

#[test]
fn routed_sources_keep_placement_through_release_on_both_render_orders() {
    use crate::life::phrase_boundary::PhraseTone;
    let geography = [12., 0., 1., 12.97, 23.5, 47.].map(site);
    let ir = [york_response(), york_right_response()];
    let land = Landscape::new(Log2Space::new(110., 8000., 48));
    let flora = || {
        Flora::new(
            21,
            FloraSpec {
                site_id: 4,
                fade_sec: 0.,
                residual: true,
                ..Default::default()
            },
            (220., 880.),
            FloraSeeding::Uniform,
            42,
            &land,
            FS,
            HOP,
        )
    };
    let rain = || {
        Rain::new(
            22,
            RainSpec {
                site_id: 3,
                rate: 500.,
                hit: 1.,
                ring: 1.,
                objects: 4,
                fade_sec: 0.,
                leave_drops_out: true,
                ..Default::default()
            },
            (300., 6000.),
            FieldTarget::Consonance,
            21,
            &land,
            FS,
            HOP,
        )
    };
    for phrase in [false, true] {
        let mut renderer = ScheduleRenderer::new(TIME);
        renderer.sites = SiteBuses::prepare(&geography, HOP);
        use crate::core::stream::loo::{SitePcm, SourceKind};
        renderer.sites.as_mut().unwrap().full = (0..geography.len())
            .map(|id| SitePcm {
                id: id as u32,
                voice: false,
                flora: false,
                onset: false,
                tonal: false,
                samples: vec![0.; HOP],
            })
            .collect();
        renderer.room = RoomBuses::prepare(RoomMode::Both, HOP, 3, true).unwrap();
        let captured = |kind, id, site_id| SourcePcm {
            kind,
            id,
            site_id,
            generation: 0,
            analyze_hr: true,
            onset_history: false,
            tonal_context: None,
            habitat: vec![0.; HOP],
        };
        renderer.loo_pcm = vec![
            captured(SourceKind::Voice, 2, 2),
            captured(SourceKind::Flora, 21, 4),
            captured(SourceKind::Rain, 22, 3),
        ];
        renderer.loo_pcm[0].generation = 7;
        let mut batches: Vec<_> = [0, 1, 2, 3, 4, 5, 2]
            .into_iter()
            .enumerate()
            .map(|(j, id)| {
                let mut b = tests::outcome_batch(BodyKind::Sine);
                b.body_policy = Some(crate::life::action_candidates::PolicySnapshot {
                    site_id: id,
                    at: 0,
                    is_alive: true,
                    gate_allows_onset: true,
                });
                b.source_id = j as u64;
                b.routing = Routing {
                    to_presentation: j != 4,
                    to_habitat: j != 5,
                };
                b.tones[0].freq_hz += j as f32 * 27.;
                b.tones[0].amp = if j == 6 { 0.000025 } else { 0.025 };
                if j == 6 {
                    b.tones[0].freq_hz = 337. + 2. * 27.;
                }
                b.tones[0].hold_ticks = Some(490);
                b.tones[0].adsr = Some(ToneAdsr {
                    attack_sec: 0.,
                    decay_sec: 0.,
                    sustain_level: 1.,
                    release_sec: 0.05,
                });
                b.tones[0].phrase = phrase.then_some(PhraseTone { reservation: None });
                b
            })
            .collect();
        let mut solo: Vec<_> = batches
            .iter()
            .map(|_| ScheduleRenderer::new(TIME))
            .collect();
        let (mut plants, mut rains) = ([flora()], [rain()]);
        let (mut dry_plant, mut dry_rain) = (flora(), rain());
        let mut stems = [Vec::new(), Vec::new()];
        let mut external_stems: [Vec<f64>; 3] = Default::default();
        let mut heard = [false; 3];
        for hop in 0..6 {
            let now = (hop * HOP) as u64;
            // Reuse the first voice's identity at a new site while the original tone releases.
            if hop == 1 {
                batches[0].body_policy.as_mut().unwrap().site_id = 5;
                batches[0].source_generation += 1;
                batches[0].tones[0].onset = now;
                solo.push(ScheduleRenderer::new(TIME));
            }
            let commands = match hop {
                0 => batches.as_slice(),
                1 => &batches[..1],
                _ => &[],
            };
            // Sum the independent stems again, omitting the named owner before mixing.
            let mut external = [[[0f64; HOP]; 2]; 3]; // direct at receiver, unweighted reflection send
            let mut include = |owner: usize, source: usize, i: usize, sample: f64| {
                let d = (geography[[2, 4, 3][owner]].y - geography[source].y)
                    .abs()
                    .max(1.);
                external[owner][0][i] += sample / d;
                external[owner][1][i] += sample;
            };
            let mut expected = vec![[[0f64; HOP]; 2]; geography.len()];
            for (j, r) in solo.iter_mut().enumerate() {
                let id = if j == 0 {
                    0
                } else {
                    batches[j % batches.len()].body_policy.unwrap().site_id as usize
                };
                let b = if hop == 0 {
                    &batches[j..=j]
                } else if j == batches.len() && hop == 1 {
                    &batches[..1]
                } else {
                    &[]
                };
                let f = r.render(b, now, &NeuralRhythms::default()).unwrap();
                for (i, (&p, &h)) in f.presentation.iter().zip(f.habitat).enumerate() {
                    expected[id][0][i] += f64::from(p);
                    expected[id][1][i] += f64::from(h);
                    for owner in 0..3 {
                        if owner != 0 || j != 2 {
                            include(owner, id, i, h.into());
                        }
                    }
                    heard[0] |= j == 0 && hop == 1 && p != 0.;
                }
            }
            plants[0].advance(&land);
            dry_plant.advance(&land);
            for (id, is_rain) in [(4, false), (3, true)] {
                let (mut p, mut h) = ([0.; HOP], [0.; HOP]);
                if is_rain {
                    dry_rain.render_add(&mut p, &mut h);
                } else {
                    dry_plant.render_add(&mut p, &mut h);
                }
                heard[if is_rain { 2 } else { 1 }] |= p.iter().any(|&x| x != 0.);
                for i in 0..HOP {
                    expected[id][0][i] += f64::from(p[i]);
                    expected[id][1][i] += f64::from(h[i]);
                    include(0, id, i, h[i].into());
                    if is_rain {
                        include(1, id, i, h[i].into());
                        include(
                            2,
                            id,
                            i,
                            f64::from(h[i]) - f64::from(dry_rain.drops_habitat_pcm()[i]),
                        );
                    } else {
                        include(2, id, i, h[i].into());
                    }
                }
            }
            let f = renderer
                .render_with_source_pcm(
                    commands,
                    now,
                    &NeuralRhythms::default(),
                    &mut [],
                    &mut plants,
                    &mut rains,
                    |_, _, _, _| {},
                )
                .unwrap();
            for i in 0..HOP {
                let mut direct = [0.; 2];
                for bus in 0..2 {
                    stems[bus].push(expected.iter().map(|s| s[bus][i]).sum());
                    direct[bus] = expected
                        .iter()
                        .zip(geography)
                        .map(|(s, loc)| s[bus][i] / loc.y.max(1.))
                        .sum();
                }
                let n = hop * HOP + i;
                let p = |ch: usize| 12. * (direct[0] + reflection(&stems[0], &ir[ch], n));
                let diffuse = reflection(&stems[1], &ir[0], n);
                for full in &f.sites.unwrap().full {
                    let receiver = geography[full.id as usize];
                    let direct: f64 = expected
                        .iter()
                        .zip(geography)
                        .map(|(s, loc)| s[1][i] / (receiver.y - loc.y).abs().max(1.))
                        .sum();
                    close(full.samples[i], direct + diffuse);
                }
                close(f.presentation[i], p(0));
                close(f.stereo.unwrap()[2 * i], p(0));
                close(f.stereo.unwrap()[2 * i + 1], p(1));
                close(
                    f.habitat[i],
                    12. * (direct[1] + reflection(&stems[1], &ir[0], n)),
                );
            }
            for (owner, capture) in renderer.loo_pcm.iter().enumerate() {
                for i in 0..HOP {
                    external_stems[owner].push(external[owner][1][i]);
                    close(
                        capture.habitat[i],
                        external[owner][0][i]
                            + reflection(&external_stems[owner], &ir[0], hop * HOP + i),
                    );
                }
            }
            for (actual, expected) in renderer.sites.as_ref().unwrap().dry.iter().zip(&expected) {
                for i in 0..HOP {
                    close(actual.presentation[i], expected[0][i]);
                    close(actual.habitat[i], expected[1][i]);
                }
            }
            assert_eq!(!renderer.phrase_sources.is_empty(), phrase);
        }
        assert_eq!(heard, [true; 3]);
    }
}

#[test]
fn implicit_presentation_impulse_matches_both_12m_responses() {
    let ir = [york_response(), york_right_response()];
    let mut room = RoomBuses::prepare(RoomMode::Presentation, HOP, 0, false)
        .unwrap()
        .unwrap();
    let mut dry = [0.; HOP];
    dry[0] = 1.;
    let mut left = [0.; HOP];
    let mut max_error = [0f64; 2];
    for offset in (0..ir[0].len()).step_by(HOP) {
        room.process_presentation(&dry, &mut left);
        for (i, pair) in room.interleaved.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(left[i], pair[0]);
            let t = offset + i;
            for ch in 0..2 {
                let expected = if t == 0 {
                    1.
                } else {
                    12. / 23.5 * f64::from(ir[ch].get(t).copied().unwrap_or(0.))
                };
                max_error[ch] = max_error[ch].max((f64::from(pair[ch]) - expected).abs());
            }
        }
        dry.fill(0.);
    }
    assert!(room.drained());
    assert!(max_error.into_iter().all(|e| e < 1e-5), "{max_error:?}");
    println!("12 m impulse max absolute error L/R: {max_error:?}");
}

// Replay a single ecological performance to compare the two audience paths
// independently of the site-specific ears that can change a live performance.
#[test]
#[cfg(feature = "native")]
#[ignore = "full Voice, Flora and Rain room transfer comparison"]
fn default_site_and_implicit_presentation_replay_the_same_performance() {
    use crate::audio::limiter::{Limiter, LimiterMode};
    use crate::{StepEngine, StepStatus};

    let score = include_str!("../../../tests/research/room_dry_load.rhai");
    let mut performance = StepEngine::new(score, "", Some(735), 48000).unwrap();
    assert_eq!(performance.hop_len(), HOP);
    let mut implicit = RoomBuses::prepare(RoomMode::Presentation, HOP, 0, false)
        .unwrap()
        .unwrap();
    let mut spatial = RoomBuses::prepare(RoomMode::Presentation, HOP, 0, true)
        .unwrap()
        .unwrap();
    let mut sites = SiteBuses::prepare(&[Site::DEFAULT], HOP).unwrap();
    let config = crate::config::AppConfig::from_toml("[room]\nmode='presentation'").unwrap();
    let [mut guard_implicit, mut guard_spatial] = [false, true].map(|has_sites| {
        Limiter::new(LimiterMode::default(), 48000, 2)
            .with_input_gain(config.output_gain(0., has_sites))
    });
    let mut dry = [0.; HOP];
    let (mut p, mut h, mut scratch) = ([0.; HOP], [0.; HOP], [0.; HOP]);
    let mut peak = [0f32; 2];
    let mut error = [0f32; 2];
    let mut hops = 0;
    while !performance.finished() || !implicit.drained() || !spatial.drained() {
        dry.fill(0.);
        if !performance.finished() {
            assert!(matches!(
                performance.step(&mut dry),
                StepStatus::Running | StepStatus::Finished
            ));
        }
        implicit.process_presentation(&dry, &mut scratch);
        sites.dry[0].presentation.copy_from_slice(&dry);
        sites.mix(Some(&mut spatial), &mut p, &mut h, &mut scratch);
        let a = &mut implicit.interleaved;
        let b = &mut spatial.interleaved;
        for stage in 0..2 {
            if stage == 1 {
                guard_implicit.process_interleaved(a, 2);
                guard_spatial.process_interleaved(b, 2);
            }
            for (&x, &y) in a.iter().zip(b.iter()) {
                peak[stage] = peak[stage].max(y.abs());
                error[stage] = error[stage].max((x - y).abs());
            }
        }
        hops += 1;
    }
    for stage in 0..2 {
        assert!(peak[stage] > 0. && error[stage] / peak[stage] < 1e-5);
    }
    println!("12 m mixed-score replay: hops={hops} pre/post peaks={peak:?} errors={error:?}");
}

#[test]
fn site_impulses_calibrate_direct_diffuse_energy_and_final_partial_hop() {
    let ir = [york_response(), york_right_response()];
    for d in [0., 1., 12., 12.97, 23.5, 47.] {
        let mut sites = SiteBuses::prepare(&[site(d)], HOP).unwrap();
        let mut room = RoomBuses::prepare(RoomMode::Presentation, HOP, 0, true)
            .unwrap()
            .unwrap();
        assert_eq!(room.counts().reserved, 2);
        let mut energy = [0.; 2];
        let start = 17;
        let end = start + ir[0].len();
        for offset in (0..end + HOP).step_by(HOP) {
            sites.clear();
            if offset == 0 {
                sites.dry[0].presentation[start] = 1.;
            }
            // A dry H-only sample after the P tail cannot extend this room.
            sites.dry[0].habitat[HOP - 1] = 1.;
            let (mut p, mut h, mut scratch) = ([0.; HOP], [0.; HOP], [0.; HOP]);
            sites.mix(Some(&mut room), &mut p, &mut h, &mut scratch);
            close(h[HOP - 1], 12. / d.max(1.));
            for (i, pair) in room.interleaved.as_chunks::<2>().0.iter().enumerate() {
                for ch in 0..2 {
                    let tap = (offset + i).checked_sub(start);
                    let expected = match tap {
                        Some(0) => 12. / d.max(1.),
                        Some(t) if t < ir[ch].len() => 12. / 23.5 * f64::from(ir[ch][t]),
                        _ => 0.,
                    };
                    close(pair[ch], expected);
                    if tap == Some(121) {
                        assert_ne!(pair[ch], 0.);
                    }
                    if tap.is_some_and(|t| t > 0) {
                        energy[ch] += f64::from(pair[ch]).powi(2);
                    }
                }
            }
            assert_eq!(room.drained(), offset + HOP >= end);
        }
        for e in energy {
            let drr = 10. * ((12. / d.max(1.)).powi(2) / e).log10();
            assert!((drr - (-5.16 + 20. * (23.5 / d.max(1.)).log10())).abs() < 0.01);
            if d == 12.97 {
                assert!(drr.abs() < 0.02);
            }
        }
    }
}

use crate::core::{
    meter::{MeterBand, MeterNetwork},
    stream::dorsal::DorsalStream,
};
use crate::listener_twin::{ContourEvent, ListenerTwin};

struct Readers {
    dorsal: DorsalStream,
    flora_meter: MeterNetwork,
    listener: ListenerTwin,
    temporal: AcousticTemporalExpectation,
}

impl Readers {
    fn new() -> Self {
        let mut listener = ListenerTwin::with_sample_rate(FS, Default::default());
        assert!(listener.enable_contour_reporting(FS));
        Self {
            dorsal: DorsalStream::new(FS),
            flora_meter: MeterNetwork::new(),
            listener,
            temporal: AcousticTemporalExpectation::new(FS as u32).unwrap(),
        }
    }

    fn observe(&mut self, hop: usize, h: &[f32], p: &[f32]) -> Observation {
        let now = (hop * HOP) as u64;
        self.dorsal.process(h);
        let flux = self.dorsal.last_metrics().flux;
        let drive = (500. * flux.max(0.)).tanh();
        let beat = self.flora_meter.process_at(now, HOP, FS, Some(drive)).beat;
        let fired = self
            .flora_meter
            .onset_age()
            .is_some_and(|age| age < HOP as f32 / FS);
        let fast = self.listener.observe_presentation_audio(now, hop as u64, p);
        let mut episodes = Vec::new();
        self.listener.observe_contour_audio(now, p, |e| {
            if e.kind == "periodic_episode" {
                episodes.push(e)
            }
        });
        let mut windows = Vec::new();
        self.temporal.process(now, h, |o| {
            windows.push((o.end_frame, o.band, o.prediction))
        });
        Observation {
            flux,
            drive,
            beat,
            fired,
            left_flux: fast.attention_metrics.flux,
            left: fast.meter_state.beat,
            attention: fast.attention_level,
            episodes,
            windows,
        }
    }
}

struct Observation {
    flux: f32,
    drive: f32,
    beat: MeterBand,
    fired: bool,
    left_flux: f32,
    left: MeterBand,
    attention: f32,
    episodes: Vec<ContourEvent>,
    windows: Vec<(u64, u8, crate::core::temporal_expectation::Prediction)>,
}

// Independent, zero-padded f64 convolution for the long pulse train.
fn diffuse_reference(input: &[f32], ir: &[f32]) -> Vec<f64> {
    use rustfft::{FftPlanner, num_complex::Complex64};
    let n = (input.len() + ir.len() - 1).next_power_of_two();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut x = vec![Complex64::default(); n];
    for (c, &v) in x.iter_mut().zip(input) {
        c.re = f64::from(v);
    }
    let mut y = vec![Complex64::default(); n];
    for (c, &v) in y[1..].iter_mut().zip(&ir[1..]) {
        c.re = f64::from(v);
    }
    fft.process(&mut x);
    fft.process(&mut y);
    for (a, b) in x.iter_mut().zip(y) {
        *a *= b;
    }
    planner.plan_fft_inverse(n).process(&mut x);
    x[..input.len()]
        .iter()
        .map(|c| c.re / (n as f64 * 23.5))
        .collect()
}

#[test]
fn acoustic_readers_at_stage_distances() {
    const ISOLATED: usize = 24000;
    // Finish the isolated note and drain its complete response before the train.
    const RESET: usize = (ISOLATED + 19200 + 351360).div_ceil(HOP);
    const TRAIN: usize = (RESET + 48) * HOP;
    let hops = (TRAIN + 16 * 24000).div_ceil(HOP);
    let mut input = vec![0.; hops * HOP];
    for (i, x) in input.iter_mut().enumerate() {
        let (j, f, len) = if i >= TRAIN {
            let n = i - TRAIN;
            (
                n % 24000,
                if (n / 24000).is_multiple_of(2) {
                    375.
                } else {
                    500.
                },
                9600,
            )
        } else if i >= ISOLATED {
            (i - ISOLATED, 375., 19200)
        } else {
            continue;
        };
        if j < len {
            *x = ((std::f64::consts::TAU * f * j as f64 / 48000.).sin() * (j as f64 / 960.).min(1.))
                as f32;
        }
    }
    let mut silence = Readers::new();
    for hop in 0..hops {
        let o = silence.observe(hop, &[0.; HOP], &[0.; HOP]);
        assert!(
            !o.fired
                && o.left.confidence == 0.
                && o.episodes.is_empty()
                && o.windows.iter().all(|v| v.1 == 0)
        );
    }
    let il = york_response();
    let ir = york_right_response();
    for amp in [0.03, 0.04] {
        let stem: Vec<_> = input.iter().map(|x| x * amp).collect();
        // The nonlinear meter can choose a different period for sub-ULP differences
        // in the waveform. Feed the reference readers an independent f32 stream
        // from the same stem; validate both channels against f64 below.
        let mut no_direct = il.clone();
        no_direct[0] = 0.;
        let mut reference_room = crate::core::room::RoomKernel::new(&no_direct, HOP)
            .stream_pool(1, 1)
            .unwrap()
            .pop()
            .unwrap();
        let mut left_reflection = vec![0.; stem.len()];
        for (x, y) in stem
            .as_chunks::<HOP>()
            .0
            .iter()
            .zip(left_reflection.as_chunks_mut::<HOP>().0)
        {
            reference_room.process(&x.map(|s| s * (1f32 / 23.5)), y);
        }
        let reflections = [diffuse_reference(&stem, &il), diffuse_reference(&stem, &ir)];
        for d in [1., 12., 23.5] {
            for mode in [RoomMode::Off, RoomMode::Presentation, RoomMode::Both] {
                let mut buses = SiteBuses::prepare(&[site(d)], HOP).unwrap();
                let mut room = RoomBuses::prepare(mode, HOP, 0, true).unwrap();
                let (mut actual, mut reference) = (Readers::new(), Readers::new());
                let mut onsets = Vec::new();
                let mut episode = Vec::new();
                let mut leading_period = None;
                let mut prediction_weight = 0f32;
                let mut peak_flux = [0f32; 2];
                let (mut band_count, mut train_episodes) = (0, 0);
                let (mut confidence, mut attention, mut peak_drive) = (0f32, 0f32, 0f32);
                for (hop, chunk) in stem.as_chunks::<HOP>().0.iter().enumerate() {
                    buses.dry[0].presentation.copy_from_slice(chunk);
                    buses.dry[0].habitat.copy_from_slice(chunk);
                    let (mut p, mut h, mut scratch) = ([0.; HOP], [0.; HOP], [0.; HOP]);
                    buses.mix(room.as_mut(), &mut p, &mut h, &mut scratch);
                    let mut rp = [0.; HOP];
                    let mut rh = [0.; HOP];
                    for i in 0..HOP {
                        let n = hop * HOP + i;
                        let direct = f64::from(chunk[i]) / d;
                        let expected = |wet, ch: usize| {
                            12. * (direct + if wet { reflections[ch][n] } else { 0. })
                        };
                        close(p[i], expected(mode != RoomMode::Off, 0));
                        close(h[i], expected(mode == RoomMode::Both, 0));
                        let dry = chunk[i] * (1. / d) as f32;
                        let wet = 12. * (dry + left_reflection[n]);
                        rp[i] = if mode == RoomMode::Off {
                            12. * dry
                        } else {
                            wet
                        };
                        rh[i] = if mode == RoomMode::Both {
                            wet
                        } else {
                            12. * dry
                        };
                        assert_eq!((p[i], h[i]), (rp[i], rh[i]));
                        if let Some(room) = &room {
                            close(room.interleaved[2 * i + 1], expected(true, 1));
                        }
                    }
                    // Isolated-pitch detection and acquisition of a beat from silence are
                    // separate observations. Start the latter with two fresh receivers.
                    if hop == RESET {
                        assert!(room.as_ref().is_none_or(RoomBuses::drained));
                        actual = Readers::new();
                        reference = Readers::new();
                    }
                    let a = actual.observe(hop, &h, &p);
                    let r = reference.observe(hop, &rh, &rp);
                    assert_eq!(a.fired, r.fired);
                    close(a.flux, f64::from(r.flux));
                    close(a.left_flux, f64::from(r.left_flux));
                    close(a.drive, f64::from(r.drive));
                    close(a.attention, f64::from(r.attention));
                    for (b, rb) in [(a.beat, r.beat), (a.left, r.left)] {
                        assert!(
                            crate::core::phase::wrap_pm_pi(b.phase - rb.phase).abs() < 0.001
                                && (b.freq_hz - rb.freq_hz).abs() < 0.001
                                && (b.confidence - rb.confidence).abs() < 0.001,
                            "mode={mode:?} d={d} amp={amp} hop={hop} {b:?} / {rb:?}"
                        );
                    }
                    assert_eq!(a.windows, r.windows);
                    assert_eq!(a.episodes.len(), r.episodes.len());
                    for (e, re) in a.episodes.iter().zip(r.episodes) {
                        assert_eq!(
                            (e.available_sec, e.onset_sec),
                            (re.available_sec, re.onset_sec)
                        );
                        let hz = e.periodic_frequency_hz.unwrap();
                        assert!((hz - re.periodic_frequency_hz.unwrap()).abs() < 0.1);
                        if hop * HOP < TRAIN {
                            episode.push((e.onset_sec, e.available_sec, hz));
                        } else {
                            train_episodes += 1;
                        }
                    }
                    if hop * HOP >= TRAIN {
                        if a.fired {
                            onsets.push(hop);
                        }
                        band_count += a.windows.iter().filter(|w| w.1 != 0).count();
                        for (_, _, prediction) in &a.windows {
                            if prediction.leading_period_sec.is_some() {
                                leading_period = prediction.leading_period_sec;
                            }
                            prediction_weight = prediction_weight.max(prediction.recurrence_weight);
                        }
                        confidence = confidence.max(a.left.confidence);
                        attention = attention.max(a.attention);
                        peak_drive = peak_drive.max(a.drive);
                        peak_flux[0] = peak_flux[0].max(a.flux);
                        peak_flux[1] = peak_flux[1].max(a.left_flux);
                    }
                }
                eprintln!(
                    "READERS {mode:?} {d} {amp} onsets={onsets:?} bands={band_count} period={leading_period:?} weight={prediction_weight:.4} confidence={confidence:.4} attention={attention:.4} drive={peak_drive:.4} flux_OL={peak_flux:?} isolated={episode:?} train_episodes={train_episodes}"
                );
                assert!(onsets.len() >= 8 && band_count >= 8 && confidence > 0. && attention > 0.);
                assert!(episode.iter().any(|v| (v.2 - 375.).abs() < 10.));
            }
        }
    }
}

#[test]
fn linked_limiter_on_near_far_stems() {
    use crate::audio::limiter::{Limiter, LimiterMode};
    let ir = [york_response(), york_right_response()];
    for mode in [RoomMode::Off, RoomMode::Presentation, RoomMode::Both] {
        let channels = if mode == RoomMode::Off { 1 } else { 2 };
        let mut buses = SiteBuses::prepare(&[site(1.), site(23.5)], HOP).unwrap();
        let mut room = RoomBuses::prepare(mode, HOP, 0, true).unwrap();
        let mut stems = [Vec::new(), Vec::new()];
        let mut raw = Vec::new();
        let mut oracle = Vec::new();
        for hop in 0..12 {
            for i in 0..HOP {
                let n = hop * HOP + i;
                let far = (0.04 * (std::f64::consts::TAU * 375. * n as f64 / 48000.).sin()) as f32;
                let near = if (3..8).contains(&hop) { far } else { 0. };
                for ((bus, stem), x) in buses.dry.iter_mut().zip(&mut stems).zip([near, far]) {
                    bus.presentation[i] = x;
                    bus.habitat[i] = x;
                    stem.push(f64::from(x));
                }
            }
            let (mut p, mut h, mut scratch) = ([0.; HOP], [0.; HOP], [0.; HOP]);
            buses.mix(room.as_mut(), &mut p, &mut h, &mut scratch);
            raw.extend_from_slice(room.as_ref().map_or(p.as_slice(), |r| &r.interleaved));
            for i in 0..HOP {
                let n = hop * HOP + i;
                for filter in &ir[..channels] {
                    let mut sum = 0.;
                    for (stem, d) in stems.iter().zip([1., 23.5]) {
                        sum += stem[n] / d;
                        if channels == 2 {
                            sum += reflection(stem, filter, n);
                        }
                    }
                    oracle.push((12. * sum) as f32);
                }
            }
        }
        for db in [0., 9.] {
            let gain = 10f32.powf(db / 20.);
            let make =
                || Limiter::new(LimiterMode::default(), 48000, channels).with_input_gain(gain);
            let mut guard = make();
            let mut expected = oracle.clone();
            let mut out = raw.clone();
            make().process_interleaved(&mut expected, channels);
            guard.process_interleaved(&mut out, channels);
            let mut none = raw.clone();
            Limiter::new(LimiterMode::None, 48000, channels)
                .with_input_gain(gain)
                .process_interleaved(&mut none, channels);
            let mut pre = [[0f32; 2]; 3];
            let mut post = pre;
            let mut active = None;
            let mut last = None;
            let mut affected = 0usize;
            let mut over = 0;
            for (n, ((&x, &y), &z)) in raw.iter().zip(&out).zip(&expected).enumerate() {
                close(x, f64::from(oracle[n]));
                close(y, f64::from(z));
                close(x * gain, f64::from(oracle[n] * gain));
                assert_eq!(none[n], x * gain);
                let frame = n / channels;
                let ch = n % channels;
                let phase = usize::from(frame >= 3 * HOP) + usize::from(frame >= 8 * HOP);
                pre[phase][ch] = pre[phase][ch].max((x * gain).abs());
                post[phase][ch] = post[phase][ch].max(y.abs());
                if (x * gain).abs() > 0.98 {
                    over += 1;
                }
                if ch == 0 && (x * gain - y).abs() > 1e-7 {
                    active.get_or_insert(frame);
                    last = Some(frame);
                    affected += 1;
                }
            }
            let stats = guard.stats();
            assert_eq!(stats.num_over, over);
            if channels == 1 {
                close(pre[1][0], 0.04 * (12. + 12. / 23.5) * f64::from(gain));
                if db == 0. {
                    assert_eq!(pre, post);
                    assert_eq!(stats.max_reduction_db, 0.);
                } else {
                    assert!(post[1][0] <= 0.98 && affected > 0 && post[2][0] < pre[2][0]);
                }
            }
            eprintln!(
                "LIMIT {mode:?} db={db} pre={pre:?} post={post:?} over={over} reduction_db={} left_first_last={active:?},{last:?} affected_frames={affected}",
                stats.max_reduction_db
            );
        }
    }
}
