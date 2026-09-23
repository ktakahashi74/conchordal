use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};
use tracing::*;

use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use ringbuf::traits::Observer;

use crate::audio::limiter::{Limiter, LimiterMeter, LimiterMode};
use crate::core::analysis_worker;
use crate::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeParams, LandscapeUpdate};
use crate::core::log2space::Log2Space;
use crate::core::meter::MeterNetwork;
use crate::core::modulation::NeuralRhythms;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
use crate::core::nsgt_rt::{RtConfig, RtNsgtKernelLog2};
use crate::core::phase::wrap_pm_pi;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::core::stream::{analysis::AnalysisStream, dorsal::DorsalStream};
use crate::core::temporal_expectation::TemporalObservation;
use crate::core::timebase::Tick;
use crate::dcc_coupler::{DccCoupler, ListenerPressure};
use crate::life::community::Community;
use crate::life::conductor::Conductor;
use crate::life::phonation_engine::ParticipationPrediction;
use crate::life::report::{
    HopTimingSample, JsonlReporter, ListenerContourSample, ListenerStateSample,
    ParticipationOutcomeSample, RhythmObservation, onset_samples_from_batches, scaffold_phase_0_1,
    summarize_populations,
};
use crate::life::schedule_renderer::ScheduleRenderer;
use crate::life::voice::{PhonationBatch, SoundBody};
use crate::listener_twin::{ListenerFastState, ListenerState, ListenerTwin};
use crate::runtime_profile::{HopProfile, RunProfile, begin_allocations, finish_allocations};
use crate::scenario::{Action, ScaffoldConfig, Scenario};
use crate::scripting::ScriptHost;
use crate::viewdata::{
    DorsalFrame, ListenerFrame, PlaybackState, PredictionFrame, SimulationMeta, SpecFrame, UiFrame,
    VoiceStateInfo, WaveFrame,
};
use crate::{
    audio::output::AudioOutput, config::AppConfig, core::harmonicity_kernel::HarmonicityParams,
};

const MAX_LANDSCAPE_LAG_FRAMES: u64 = 1;
const UI_MIN_INTERVAL: Duration = Duration::from_millis(33);

struct AudioMonitor {
    min_occupancy: Option<usize>,
    max_peak: f32,
    slow_chunks: u32,
    max_analysis_wait: Duration,
    max_listener_wait: Duration,
    last_underrun_frames: u64,
    last_stats_log: Instant,
    last_clip_log: Instant,
    last_lag_warn: Instant,
}

impl AudioMonitor {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            min_occupancy: None,
            max_peak: 0.0,
            slow_chunks: 0,
            max_analysis_wait: Duration::ZERO,
            max_listener_wait: Duration::ZERO,
            last_underrun_frames: 0,
            last_stats_log: now,
            last_clip_log: now,
            last_lag_warn: now,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        current_time: f32,
        frame_idx: u64,
        hop: usize,
        hop_duration: Duration,
        buffer_capacity: usize,
        buffer_occupancy: usize,
        chunk_peak: f32,
        chunk_elapsed: Duration,
        analysis_wait: Duration,
        listener_wait: Duration,
        underrun_frames: Option<&AtomicU64>,
        analysis_lag: Option<u64>,
        conductor_done: bool,
    ) {
        self.min_occupancy = Some(
            self.min_occupancy
                .map_or(buffer_occupancy, |m| m.min(buffer_occupancy)),
        );
        self.max_peak = self.max_peak.max(chunk_peak);
        self.max_analysis_wait = self.max_analysis_wait.max(analysis_wait);
        self.max_listener_wait = self.max_listener_wait.max(listener_wait);

        if self.last_clip_log.elapsed() > Duration::from_millis(200) {
            if chunk_peak > 0.98 {
                warn!(
                    "[t={:.6}] Audio peak high: {:.3} at frame_idx={}. Consider more headroom.",
                    current_time, chunk_peak, frame_idx
                );
                self.last_clip_log = Instant::now();
            } else if chunk_peak > 0.9 {
                warn!(
                    "[t={:.6}] Audio peak nearing clip: {:.3} at frame_idx={}",
                    current_time, chunk_peak, frame_idx
                );
                self.last_clip_log = Instant::now();
            } else if conductor_done && chunk_peak > 1e-4 {
                warn!(
                    "[t={:.6}] Scenario done but audio active: peak={:.4}",
                    current_time, chunk_peak
                );
                self.last_clip_log = Instant::now();
            }
        }

        if chunk_elapsed > hop_duration {
            self.slow_chunks += 1;
            warn!(
                "[t={:.6}] Audio hop slow: {:?} (budget {:?}, analysis_wait {:?}, listener_wait {:?}) frame_idx={}",
                current_time, chunk_elapsed, hop_duration, analysis_wait, listener_wait, frame_idx
            );
        }

        let should_warn = analysis_lag.is_some_and(|lag| lag >= 2);
        if should_warn && self.last_lag_warn.elapsed() > Duration::from_secs(1) {
            let lag = analysis_lag
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into());
            warn!(
                "[t={:.3}] Analysis lag (frames): A={} (Audio(gen)={})",
                current_time, lag, frame_idx
            );
            self.last_lag_warn = Instant::now();
        }

        if self.last_stats_log.elapsed() > Duration::from_secs(1) {
            let underrun_total = underrun_frames.map_or(0, |count| count.load(Ordering::Relaxed));
            let underrun_frames = underrun_total.saturating_sub(self.last_underrun_frames);
            self.last_underrun_frames = underrun_total;
            if underrun_frames > 0 {
                warn!(
                    "Audio output underrun: {underrun_frames} missing mono frames in stats interval"
                );
            }
            if let Some(min_occ) = self.min_occupancy.take() {
                debug!(
                    "[t={:.6}] Audio stats: min_occ={}, cap={}, hop={}, max_peak={:.3}, slow_chunks={}, max_analysis_wait={:?}, max_listener_wait={:?}, underrun_frames={}",
                    current_time,
                    min_occ,
                    buffer_capacity,
                    hop,
                    self.max_peak,
                    self.slow_chunks,
                    self.max_analysis_wait,
                    self.max_listener_wait,
                    underrun_frames
                );
            }
            self.max_peak = 0.0;
            self.slow_chunks = 0;
            self.max_analysis_wait = Duration::ZERO;
            self.max_listener_wait = Duration::ZERO;
            self.last_stats_log = Instant::now();
        }
    }
}

fn analysis_ok(frame_idx: u64, last_analysis: Option<u64>, max_lag: u64) -> bool {
    if frame_idx == 0 {
        return true;
    }
    match last_analysis {
        Some(id) => frame_idx.saturating_sub(id) <= max_lag,
        None => false,
    }
}

struct RuntimeUiFrameInput<'a> {
    body: Option<&'a crate::temporal_cognition::body::Snapshot>,
    self_sound: Option<crate::life::action_observation::Snapshot>,
    temporal: &'a [crate::temporal_cognition::observation::Snapshot; 2],
    scenario_name: &'a str,
    conductor: &'a Conductor,
    pop: &'a Community,
    generator_model: &'a crate::life::generator_model::GeneratorModel,
    current_landscape: &'a LandscapeFrame,
    log_space: &'a Log2Space,
    presentation_chunk: Arc<[f32]>,
    listener_fast_state: Option<ListenerFastState>,
    listener_state: Option<ListenerState>,
    current_time: f32,
    playback_state: PlaybackState,
    peak_level: f32,
    channel_peak: [f32; 2],
    now_tick: Tick,
    hop: usize,
    fs: f32,
}

fn build_initial_ui_frame(
    scenario_name: &str,
    conductor: &Conductor,
    pop: &Community,
    current_landscape: &LandscapeFrame,
    current_time: f32,
    playback_state: PlaybackState,
    fs: f32,
) -> UiFrame {
    UiFrame {
        self_sound: None,
        body: None,
        wave: WaveFrame {
            fs,
            samples: Arc::from(Vec::<f32>::new()),
        },
        spec: SpecFrame {
            spec_hz: current_landscape.space.centers_hz.clone(),
            amps: vec![0.0; current_landscape.space.n_bins()],
        },
        listener: ListenerFrame::default(),
        temporal: Default::default(),
        landscape: current_landscape.clone(),
        meta: SimulationMeta {
            time_sec: current_time,
            duration_sec: conductor.total_duration(),
            voice_count: pop.voices.len(),
            event_queue_len: conductor.remaining_events(),
            peak_level: 0.0,
            scenario_name: scenario_name.to_string(),
            scene_name: conductor.current_scene_name(current_time),
            playback_state,
            channel_peak: [0.0; 2],
            window_peak: [0.0; 2],
            kuramoto_order_r: None,
            kuramoto_active_count: 0,
            entrain_phases: Vec::new(),
        },
        prediction: PredictionFrame::default(),
        voices: Vec::new(),
    }
}

fn build_runtime_ui_frame(input: RuntimeUiFrameInput<'_>) -> UiFrame {
    let wave = WaveFrame {
        fs: input.fs,
        samples: input.presentation_chunk,
    };
    let spec = SpecFrame {
        spec_hz: input.log_space.centers_hz.clone(),
        amps: input
            .current_landscape
            .nsgt_power
            .iter()
            .map(|&x| x.sqrt())
            .collect(),
    };
    let voices: Vec<VoiceStateInfo> = input
        .pop
        .voices
        .iter()
        .map(|agent| {
            let freq_hz = agent.body.base_freq_hz();
            VoiceStateInfo {
                id: agent.id,
                freq_hz,
                target_freq: 2.0f32.powf(agent.target_pitch_log2()),
                integration_window: agent.integration_window(),
                consonance: input.current_landscape.evaluate_pitch_level(freq_hz),
            }
        })
        .collect();
    let pred_stats = input.pop.last_pred_gate_stats();
    let (entrain_phases, kuramoto_order_r) = input.pop.entrain_phases_and_order();
    let theta_hz = positive_hz(input.current_landscape.rhythm.theta.freq_hz);
    let delta_hz = positive_hz(input.current_landscape.rhythm.delta.freq_hz);
    let (pred_tau_tick, pred_horizon_tick) = input
        .generator_model
        .predictor_tau_horizon_ticks(&input.current_landscape.rhythm);
    let frame_end = input.now_tick.saturating_add((input.hop as Tick).max(1));
    let pred_next_gate = input.generator_model.last_pred_next_gate();
    let pred_available_in_hop = pred_next_gate
        .as_ref()
        .is_some_and(|(gate_tick, _)| *gate_tick >= input.now_tick && *gate_tick < frame_end);
    let pred_c_field_level_next_gate = pred_next_gate.map(|(_, scan)| scan);

    UiFrame {
        wave,
        self_sound: input.self_sound,
        body: input.body.map(|b| Arc::new(*b)),
        temporal: Arc::new(*input.temporal),
        spec,
        listener: listener_frame_from_states(input.listener_fast_state, input.listener_state),
        landscape: input.current_landscape.clone(),
        meta: SimulationMeta {
            time_sec: input.current_time,
            duration_sec: input.conductor.total_duration(),
            voice_count: input.pop.voices.len(),
            event_queue_len: input.conductor.remaining_events(),
            peak_level: input.peak_level,
            scenario_name: input.scenario_name.to_string(),
            scene_name: input.conductor.current_scene_name(input.current_time),
            playback_state: input.playback_state,
            channel_peak: input.channel_peak,
            window_peak: input.channel_peak,
            kuramoto_order_r,
            kuramoto_active_count: entrain_phases.len(),
            entrain_phases,
        },
        prediction: PredictionFrame {
            next_gate_tick_est: input.generator_model.next_gate_tick_est,
            theta_hz,
            delta_hz,
            pred_n_theta_per_delta: Some(
                input
                    .generator_model
                    .predictor_n_theta_per_delta(&input.current_landscape.rhythm),
            ),
            pred_tau_tick: Some(pred_tau_tick),
            pred_horizon_tick: Some(pred_horizon_tick),
            pred_c_field_level_next_gate,
            pred_gain_raw_mean: pred_stats.map(|stats| stats.raw_mean),
            pred_gain_raw_min: pred_stats.map(|stats| stats.raw_min),
            pred_gain_raw_max: pred_stats.map(|stats| stats.raw_max),
            pred_gain_mixed_mean: pred_stats.map(|stats| stats.mixed_mean),
            pred_gain_mixed_min: pred_stats.map(|stats| stats.mixed_min),
            pred_gain_mixed_max: pred_stats.map(|stats| stats.mixed_max),
            pred_sync_mean: pred_stats.map(|stats| stats.sync_mean),
            gate_boundary_in_hop: input.pop.last_gate_boundary_in_hop(),
            pred_available_in_hop: Some(pred_available_in_hop),
            phonation_onsets_in_hop: input.pop.last_phonation_onsets_in_hop(),
        },
        voices,
    }
}

fn positive_hz(hz: f32) -> Option<f32> {
    (hz.is_finite() && hz > 0.0).then_some(hz)
}

fn listener_frame_from_states(
    fast_state: Option<ListenerFastState>,
    state: Option<ListenerState>,
) -> ListenerFrame {
    let mut frame = ListenerFrame::default();
    if let Some(fast) = fast_state {
        frame.time_sec = fast.time_sec;
        frame.generated_frame_id = fast.generated_frame_id;
        frame.attention_level = fast.attention_level;
        frame.attention = DorsalFrame {
            e_low: fast.attention_metrics.e_low,
            e_mid: fast.attention_metrics.e_mid,
            e_high: fast.attention_metrics.e_high,
            flux: fast.attention_metrics.flux,
        };
        frame.meter = fast.meter_state;
        frame.has_fast_state = fast.has_state;
    }
    if let Some(state) = state {
        frame.time_sec = state.time_sec;
        frame.generated_frame_id = state.generated_frame_id;
        frame.analysis_frame_id = state.analysis_frame_id;
        frame.analysis_lag_frames = state.analysis_lag_frames;
        frame.stability_level = state.stability_level;
        frame.resolvability_level = state.resolvability_level;
        frame.tension_level = state.tension_level;
        frame.has_state = true;
    }
    frame
}

/// Returns `(updated, unavailable)`. `unavailable` means the analysis thread is
/// gone or invalidated: fixed-lag field analysis cannot generate the recovery
/// window while waiting, so the caller must end the run instead of waiting.
fn merge_latest_analysis_results(
    analysis_result_rx: &Receiver<analysis_worker::AnalysisResult>,
    current_landscape: &mut LandscapeFrame,
    log_space: &mut Log2Space,
    generator_model: &mut crate::life::generator_model::GeneratorModel,
    lparams: &LandscapeParams,
    last_analysis_frame: &mut Option<u64>,
    frame_idx: u64,
) -> (bool, bool) {
    let mut latest_audio: Option<(u64, Landscape)> = None;
    let mut unavailable = false;
    loop {
        match analysis_result_rx.try_recv() {
            Ok((analyzed_id, Some(frame))) => {
                *last_analysis_frame = Some(analyzed_id);
                latest_audio = Some((analyzed_id, frame));
            }
            Ok((_, None)) => {
                unavailable = true;
                current_landscape.spectral_history = None;
                *last_analysis_frame = None;
                latest_audio = None;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                unavailable = true;
                break;
            }
        }
    }
    let Some((_analysis_id, frame)) = latest_audio else {
        return (false, unavailable);
    };

    let space_changed = current_landscape.space.n_bins() != frame.space.n_bins()
        || current_landscape.space.fmin != frame.space.fmin
        || current_landscape.space.fmax != frame.space.fmax
        || current_landscape.space.bins_per_oct != frame.space.bins_per_oct;
    if space_changed {
        current_landscape.resize_to_space(frame.space.clone());
        *log_space = current_landscape.space.clone();
        generator_model.set_space(log_space.clone());
    }
    current_landscape.roughness_suppress_sigma_erb = frame.roughness_suppress_sigma_erb;
    current_landscape.roughness_kernel_params = frame.roughness_kernel_params;
    current_landscape.roughness = frame.roughness;
    current_landscape.roughness_shape_raw = frame.roughness_shape_raw;
    current_landscape.roughness01 = frame.roughness01;
    current_landscape.harmonicity = frame.harmonicity;
    current_landscape.roughness_total = frame.roughness_total;
    current_landscape.roughness_scalar_raw = frame.roughness_scalar_raw;
    current_landscape.roughness_norm = frame.roughness_norm;
    current_landscape.roughness01_scalar = frame.roughness01_scalar;
    current_landscape.loudness_mass = frame.loudness_mass;
    current_landscape.pitch_objective_mode = frame.pitch_objective_mode;
    current_landscape.harmonicity_params = frame.harmonicity_params;
    current_landscape.consonance_kernel = frame.consonance_kernel;
    current_landscape.roughness_k = frame.roughness_k;
    current_landscape.roughness_ref_peak = frame.roughness_ref_peak;
    current_landscape.roughness_ref_eps = frame.roughness_ref_eps;
    current_landscape.subjective_intensity = frame.subjective_intensity;
    current_landscape.nsgt_power = frame.nsgt_power;
    current_landscape.spectral_history = frame.spectral_history;
    current_landscape.recompute_consonance(lparams);

    if cfg!(debug_assertions) && frame_idx.is_multiple_of(30) {
        let mut max_r = 0.0f32;
        let mut max_i = 0usize;
        for (i, &r) in current_landscape.roughness01.iter().enumerate() {
            if r.is_finite() && r > max_r {
                max_r = r;
                max_i = i;
            }
        }
        let h = current_landscape
            .harmonicity01
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let r = current_landscape
            .roughness01
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let c_score = current_landscape
            .consonance_field_score_eff
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let c_level = current_landscape
            .consonance_field_level_eff
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let (c_score_pred, c_level_pred) =
            compose_consonance_field_score_level_with_params(h, r, lparams);
        debug!(
            "c_score_check bin={} h={:.4} r={:.4} c_score={:.4} c_score_pred={:.4} c_level={:.4} c_level_pred={:.4}",
            max_i, h, r, c_score, c_score_pred, c_level, c_level_pred
        );
    }

    (true, unavailable)
}

/// Advance a habituation field one hop from the landscape's raw views, then
/// write the effective views. No-op when disabled: `recompute_consonance`
/// already seeded the effective views bit-exact to raw.
fn drive_and_apply_habituation(
    landscape: &mut Landscape,
    hab: &mut crate::core::habituation::HabituationField,
    lparams: &LandscapeParams,
    dt_sec: f32,
) {
    if !hab.is_enabled() {
        return;
    }
    hab.ensure_len(landscape.space.n_bins());
    let (proj, _max) = lparams
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(&landscape.subjective_intensity, &landscape.space);
    hab.advance_from_parts(&landscape.consonance_field_level, &proj, dt_sec);
    landscape.apply_habituation(hab.state(), hab.theta(), &lparams.consonance_representation);
}

#[allow(clippy::too_many_arguments)]
fn observe_listener_analysis_results(
    listener_result_rx: Option<&Receiver<analysis_worker::AnalysisResult>>,
    listener_twin: &mut ListenerTwin,
    lparams: &LandscapeParams,
    hab_listener: &mut crate::core::habituation::HabituationField,
    timebase: crate::core::timebase::Timebase,
    generated_frame_id: u64,
    min_valid_frame: u64,
    last_listener_analysis_frame: &mut Option<u64>,
) -> (Option<Option<ListenerState>>, bool) {
    let Some(rx) = listener_result_rx else {
        return (None, false);
    };
    let mut latest_state = None;
    let mut disconnected = false;
    // Finish the queued batch without chasing an active producer on the audio thread.
    let pending = rx.len().max(1);
    for _ in 0..pending {
        match rx.try_recv() {
            Ok((analyzed_id, Some(_))) if analyzed_id < min_valid_frame => continue,
            Ok((analyzed_id, frame)) => {
                *last_listener_analysis_frame = frame.as_ref().map(|_| analyzed_id);
                let Some(mut frame) = frame else {
                    listener_twin.spectral_history = None;
                    latest_state = Some(None);
                    continue;
                };
                frame.recompute_consonance(lparams);
                // Each observed hop advances memory once, including observed silence.
                drive_and_apply_habituation(
                    &mut frame,
                    hab_listener,
                    lparams,
                    timebase.hop as f32 / timebase.fs,
                );
                let analysis_time_sec = timebase.tick_to_sec(timebase.frame_end_tick(analyzed_id));
                latest_state = Some(Some(listener_twin.observe_presentation_landscape(
                    analysis_time_sec,
                    generated_frame_id,
                    analyzed_id,
                    &frame,
                )));
            }
            Err(TryRecvError::Empty) => break,
            // Same hazard as the main analysis path: under deterministic render the
            // caller waits on `analysis_ok`, so a dead listener thread would hang it.
            Err(TryRecvError::Disconnected) => {
                disconnected = true;
                break;
            }
        }
    }
    (latest_state, disconnected)
}

pub(crate) struct RuntimeInit {
    pub(crate) ui_frame_rx: Receiver<UiFrame>,
    pub(crate) report_error_rx: Option<Receiver<String>>,
    pub(crate) worker_handle: Option<std::thread::JoinHandle<()>>,
    pub(crate) analysis_handle: Option<std::thread::JoinHandle<()>>,
    pub(crate) listener_analysis_handle: Option<std::thread::JoinHandle<()>>,
    pub(crate) start_flag: Arc<AtomicBool>,
    pub(crate) audio_out: Option<AudioOutput>,
    pub(crate) audio_init_error: Option<String>,
    pub(crate) visual_delay_frames: usize,
}

#[inline]
fn cmp_time_order(a_time: f32, a_order: u64, b_time: f32, b_order: u64) -> std::cmp::Ordering {
    a_time
        .partial_cmp(&b_time)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a_order.cmp(&b_order))
}

fn report_try<F>(reporter: &mut Option<JsonlReporter>, label: &str, f: F)
where
    F: FnOnce(&mut JsonlReporter) -> Result<(), String>,
{
    let Some(writer) = reporter.as_mut() else {
        return;
    };
    if writer.failure.is_some() {
        return;
    }
    if let Err(err) = f(writer) {
        warn!("report {label} failed: {err}");
        writer.failure = Some(err);
    }
}

fn apply_scaffold(
    rhythms: &mut crate::core::modulation::NeuralRhythms,
    scaffold: ScaffoldConfig,
    time_sec: f32,
    frame_idx: u64,
) {
    let Some(phase_0_1) = scaffold_phase_0_1(scaffold, time_sec, frame_idx) else {
        return;
    };
    let freq_hz = match scaffold {
        ScaffoldConfig::Off => return,
        ScaffoldConfig::Shared { freq_hz } | ScaffoldConfig::Scrambled { freq_hz, .. } => {
            freq_hz.max(0.0)
        }
    };
    rhythms.theta.freq_hz = freq_hz;
    rhythms.theta.phase = wrap_pm_pi(std::f32::consts::TAU * phase_0_1);
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.theta.beta = 0.0;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
}

struct AnalysisRuntimeCore {
    fs: f32,
    hop: usize,
    hop_duration: Duration,
    lparams: LandscapeParams,
    nsgt: RtNsgtKernelLog2,
    landscape: Landscape,
    dorsal: DorsalStream,
}

fn build_analysis_runtime_core(
    config: &AppConfig,
    runtime_sample_rate: u32,
) -> AnalysisRuntimeCore {
    let fs = runtime_sample_rate as f32;
    let space = Log2Space::new(55.0, 8000.0, 96);
    let lparams = LandscapeParams {
        fs,
        max_hist_cols: 256,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
        harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
        consonance_kernel: ConsonanceKernel {
            a: config.psychoacoustics.consonance.field.kernel.a,
            b: config.psychoacoustics.consonance.field.kernel.b,
            c: config.psychoacoustics.consonance.field.kernel.c,
            d: config.psychoacoustics.consonance.field.kernel.d,
        },
        consonance_representation: ConsonanceRepresentationParams {
            beta: config.psychoacoustics.consonance.field.level.beta,
            theta: config.psychoacoustics.consonance.field.level.theta,
        },
        consonance_density_roughness_gain: config.psychoacoustics.consonance.density.roughness_gain,
        habituation: crate::core::habituation::HabituationParams {
            enabled: config.psychoacoustics.habituation.enabled,
            satiation_sec: config.psychoacoustics.habituation.satiation_sec,
            recovery_sec: config.psychoacoustics.habituation.recovery_sec,
            ref_drive: config.psychoacoustics.habituation.ref_drive,
        },
        loudness_exp: config.psychoacoustics.loudness_exp,
        tau_ms: config.analysis.tau_ms,
        ref_power: 1e-4,
        roughness_k: config.psychoacoustics.roughness_k,
        roughness_ref_f0_hz: 1000.0,
        roughness_ref_sep_erb: 0.25,
        roughness_ref_mass_split: 0.5,
        roughness_ref_eps: 1e-12,
    };
    let nfft = config.analysis.nfft;
    let hop = config.analysis.hop_size;
    let overlap = 1.0 - (hop as f32 / nfft as f32);
    let power_mode = if config.psychoacoustics.use_incoherent_power {
        PowerMode::Incoherent
    } else {
        PowerMode::Coherent
    };
    let nsgt_kernel = NsgtKernelLog2::new(
        NsgtLog2Config {
            fs,
            overlap,
            nfft_override: Some(nfft),
            kernel_align: config.analysis.kernel_align,
        },
        space,
        None,
        power_mode,
    );
    let nsgt = RtNsgtKernelLog2::with_config(nsgt_kernel.clone(), RtConfig::default());
    let hop_duration = Duration::from_secs_f32(hop as f32 / fs);
    let landscape = Landscape::new(nsgt.space().clone());
    let dorsal = DorsalStream::new(fs);

    AnalysisRuntimeCore {
        fs,
        hop,
        hop_duration,
        lparams,
        nsgt,
        landscape,
        dorsal,
    }
}

fn spawn_analysis_worker(
    name: &'static str,
    analysis_stream: AnalysisStream,
    audio_to_analysis_rx: Receiver<(u64, Arc<[f32]>)>,
    analysis_result_tx: Sender<analysis_worker::AnalysisResult>,
    analysis_update_rx: Receiver<LandscapeUpdate>,
    delivery: analysis_worker::AnalysisDelivery,
    temporal_tap: Option<crate::temporal_cognition::observation::Tap>,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            analysis_worker::run(
                analysis_stream,
                audio_to_analysis_rx,
                analysis_result_tx,
                analysis_update_rx,
                delivery,
                temporal_tap,
            )
        })
        .expect("spawn analysis worker")
}

pub fn validate_scenario_script_extension(script_path: &Path) -> Result<(), String> {
    let ext = script_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext != "rhai" {
        return Err(format!(
            "Scenario must be a .rhai script: {}",
            script_path.display()
        ));
    }
    Ok(())
}

pub fn compile_scenario_from_script(
    script_path: &Path,
    args: &crate::cli::Args,
    _config: &AppConfig,
) -> Result<Scenario, String> {
    crate::life::modal::register_modal();
    validate_scenario_script_extension(script_path)?;
    let path_str = script_path.to_string_lossy();
    ScriptHost::load_script(&path_str, args.seed).map_err(|e| {
        let pos = e
            .position
            .map(|pos| format!(" (line {})", pos.line().unwrap_or(0)))
            .unwrap_or_default();
        format!(
            "Failed to run scenario script {}: {}{pos}",
            script_path.display(),
            e.message
        )
    })
}

pub(crate) fn validate_scenario(scenario: &Scenario) -> Result<(), String> {
    if scenario.events.is_empty() {
        return Err("Scenario has no events".to_string());
    }

    let mut has_finish = false;
    for event in &scenario.events {
        for action in &event.actions {
            match action {
                Action::Finish => {
                    has_finish = true;
                }
                Action::Spawn { .. }
                | Action::UpdatePopulation { .. }
                | Action::ReleasePopulation { .. }
                | Action::SetRespawnPolicy { .. }
                | Action::SetPopulationCrowdingTarget { .. } => {}
                Action::SetHarmonicityParams { .. }
                | Action::SetGlobalCoupling { .. }
                | Action::SetRoughnessTolerance { .. } => {}
            }
        }
    }

    if !has_finish {
        return Err("Scenario has no Finish action".to_string());
    }

    if scenario.duration_sec <= 0.0 {
        return Err("Scenario duration_sec must be > 0".to_string());
    }
    if let Some(max_time) = scenario
        .events
        .iter()
        .map(|ev| ev.time)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        && scenario.duration_sec + f32::EPSILON < max_time
    {
        return Err("Scenario duration_sec is before last event".to_string());
    }

    let mut prev_order = None;
    for event in &scenario.events {
        let order = event.order;
        if let Some(prev) = prev_order
            && order <= prev
        {
            return Err("Event order is not strictly increasing".to_string());
        }
        prev_order = Some(order);
    }

    Ok(())
}

pub fn run_compile_only(args: crate::cli::Args, config: AppConfig) {
    let path = Path::new(&args.scenario_path);
    let scenario = compile_scenario_from_script(path, &args, &config).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if let Err(e) = validate_scenario(&scenario) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let mut markers = scenario.scene_markers.clone();
    markers.sort_by(|a, b| cmp_time_order(a.time, a.order, b.time, b.order));
    for marker in &markers {
        eprintln!(
            "scene t={:.3} order={} name={}",
            marker.time, marker.order, marker.name
        );
    }
    let mut events = scenario.events.clone();
    events.sort_by(|a, b| cmp_time_order(a.time, a.order, b.time, b.order));
    for event in &events {
        let action_descs: Vec<String> = event.actions.iter().map(ToString::to_string).collect();
        eprintln!(
            "event t={:.3} order={} {}",
            event.time,
            event.order,
            action_descs.join(" | ")
        );
    }
    let event_count = scenario.events.len();
    let action_count: usize = scenario
        .events
        .iter()
        .map(|event| event.actions.len())
        .sum();
    let marker_count = scenario.scene_markers.len();
    eprintln!(
        "OK compile-only: {} (events={}, actions={}, markers={})",
        path.display(),
        event_count,
        action_count,
        marker_count
    );
}

fn resolve_limiter_mode(config: &AppConfig) -> LimiterMode {
    let guard_mode = match config.audio.limiter {
        crate::config::LimiterSetting::None => LimiterMode::None,
        crate::config::LimiterSetting::SoftClip => {
            LimiterMode::SoftClip(crate::audio::limiter::SoftClipParams::default())
        }
        crate::config::LimiterSetting::PeakLimiter => {
            LimiterMode::PeakLimiter(crate::audio::limiter::PeakLimiterParams::default())
        }
    };
    Limiter::from_env_or(guard_mode)
}

fn load_scenario_or_exit(
    path: &Path,
    args: &crate::cli::Args,
    config: &AppConfig,
) -> (String, Scenario) {
    let scenario_label = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scenario")
        .to_string();
    let scenario = compile_scenario_from_script(path, args, config).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    info!(
        "scenario seed: {} (replay with --seed {})",
        scenario.seed, scenario.seed
    );
    (scenario_label, scenario)
}

/// Path-specific wiring knobs. Every field is an actual difference between the
/// GUI/headless path (`init_runtime`) and the offline render path (`run_render`).
struct WiringOptions {
    #[cfg(test)]
    offline_body_probe: Option<OfflineBodyProbe>,
    ui_channel_capacity: usize,
    /// Enables listener analysis even without DCC coupling (GUI display / report).
    listener_forced: bool,
    wait_user_exit: bool,
    start_playing: bool,
    audio_prod: Option<ringbuf::HeapProd<f32>>,
    wav_tx: Option<Sender<Arc<[f32]>>>,
    reporter: Option<JsonlReporter>,
    deterministic_analysis: bool,
    /// Offline render only: footprints return at the hop that asked, whatever the worker's
    /// pace. The instrument keeps wall-clock delivery, even with `--play=false`.
    deterministic_footprints: bool,
    guard_meter: Option<Arc<LimiterMeter>>,
    underrun_frames: Option<Arc<AtomicU64>>,
    reserve_runtime_ids_through: u64,
    profile: Option<RunProfile>,
    audio_counters: Option<Arc<crate::audio::output::AudioCallbackCounters>>,
}

struct RuntimeWiring {
    ui_frame_rx: Receiver<UiFrame>,
    report_error_rx: Option<Receiver<String>>,
    start_flag: Arc<AtomicBool>,
    worker_handle: thread::JoinHandle<()>,
    analysis_handle: thread::JoinHandle<()>,
    listener_analysis_handle: Option<thread::JoinHandle<()>>,
}

/// Wiring shared by both runtime paths: analysis core, channels, analysis
/// worker threads, Community/Conductor setup, and the worker thread spawn.
fn wire_runtime(
    config: &AppConfig,
    runtime_sample_rate: u32,
    scenario_label: String,
    scenario: Scenario,
    stop_flag: Arc<AtomicBool>,
    opts: WiringOptions,
) -> Result<RuntimeWiring, String> {
    let WiringOptions {
        #[cfg(test)]
        offline_body_probe,
        ui_channel_capacity,
        listener_forced,
        wait_user_exit,
        start_playing,
        audio_prod,
        wav_tx,
        reporter,
        deterministic_analysis,
        deterministic_footprints,
        guard_meter,
        underrun_frames,
        reserve_runtime_ids_through,
        profile,
        audio_counters,
    } = opts;

    let action_profiles = if scenario.temporal_mode == crate::scenario::TemporalMode::Observe {
        config
            .temporal_action_profiles
            .as_ref()
            .map(|spec| {
                crate::temporal_cognition::action_profiles::Profiles::load(config, spec)
                    .map(Arc::new)
            })
            .transpose()
            .map_err(|error| format!("{error:#}"))?
    } else {
        None
    };
    if action_profiles.is_some() && runtime_sample_rate != config.audio.sample_rate {
        return Err("action profile sample rate differs from the runtime device".into());
    }
    let core = build_analysis_runtime_core(config, runtime_sample_rate);
    let fs = core.fs;
    let hop = core.hop;
    let hop_duration = core.hop_duration;
    let lparams_runtime = core.lparams.clone();

    let (ui_frame_tx, ui_frame_rx) = bounded::<UiFrame>(ui_channel_capacity);
    let (report_error_tx, report_error_rx) = if reporter.is_some() || profile.is_some() {
        let (tx, rx) = bounded(1);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let (analysis_update_tx, analysis_update_rx) = bounded::<LandscapeUpdate>(8);
    let (audio_to_analysis_tx, audio_to_analysis_rx) = bounded::<(u64, Arc<[f32]>)>(64);
    let (analysis_result_tx, analysis_result_rx) = bounded::<analysis_worker::AnalysisResult>(4);

    // NSGT-RT based audio analysis thread.
    let observing = scenario.temporal_mode == crate::scenario::TemporalMode::Observe;
    let mut temporal_snapshots = [None, None];
    let mut candidate_tables = [None, None];
    let mut reference_context = None;
    let mut taps = std::array::from_fn::<_, 2, _>(|bus| {
        observing.then(|| {
            let tap = crate::temporal_cognition::observation::Tap::spawn(
                bus as u8,
                runtime_sample_rate,
                hop,
                core.nsgt.nfft(),
                core.landscape.space.clone(),
                crate::temporal_cognition::observation::Options {
                    action_profiles: action_profiles.clone(),
                    body_prototypes: config.temporal_body_prototypes.as_ref().map(|model| {
                        let scales = config
                            .temporal_body
                            .expect("prototype model requires scales");
                        (
                            scales,
                            crate::temporal_cognition::body_model::Prototypes::new(model, scales),
                        )
                    }),
                    deterministic: deterministic_analysis,
                    ridge: config.temporal_ridge,
                    acoustic: config.temporal_acoustic,
                    memory: config.temporal_memory,
                    gesture: config.temporal_gesture,
                    period: config.temporal_period,
                },
            );
            temporal_snapshots[bus] = Some(Arc::clone(&tap.snapshot));
            candidate_tables[bus] = Some(Arc::clone(&tap.candidate_table));
            if bus == 0 && config.temporal_private_trace.is_some() {
                reference_context = Some(Arc::clone(&tap.reference_context));
            }
            tap
        })
    });
    let body_capture = if observing {
        config.temporal_body.map(|body| {
            crate::temporal_cognition::body::Capture::spawn(
                core.nsgt.clone(),
                body,
                deterministic_analysis,
                config.temporal_body_prototypes.as_ref().map(|model| {
                    crate::temporal_cognition::body_model::Prototypes::new(model, body)
                }),
            )
        })
    } else {
        None
    };
    let analysis_stream = AnalysisStream::new(core.lparams.clone(), core.nsgt.clone());
    let analysis_handle = spawn_analysis_worker(
        "field-analysis",
        analysis_stream,
        audio_to_analysis_rx,
        analysis_result_tx,
        analysis_update_rx,
        analysis_worker::AnalysisDelivery::Latest,
        taps[0].take(),
    );

    let dcc_coupler = DccCoupler::new(config.dcc);
    let listener_analysis_enabled =
        observing || listener_forced || dcc_coupler.coupling_strength() > 0.0;
    let (
        presentation_to_listener_tx,
        listener_result_rx,
        listener_analysis_update_tx,
        listener_analysis_handle,
    ) = if listener_analysis_enabled {
        let listener_stream = AnalysisStream::new(core.lparams.clone(), core.nsgt.clone());
        let (presentation_tx, presentation_rx) = bounded::<(u64, Arc<[f32]>)>(64);
        let (result_tx, result_rx) = bounded::<analysis_worker::AnalysisResult>(4);
        let (update_tx, update_rx) = bounded::<LandscapeUpdate>(8);
        let handle = spawn_analysis_worker(
            "listener-analysis",
            listener_stream,
            presentation_rx,
            result_tx,
            update_rx,
            analysis_worker::AnalysisDelivery::Ordered,
            taps[1].take(),
        );
        (
            Some(presentation_tx),
            Some(result_rx),
            Some(update_tx),
            Some(handle),
        )
    } else {
        (None, None, None, None)
    };

    let landscape = core.landscape;
    let dorsal = core.dorsal;

    let mut pop = Community::new(crate::core::timebase::Timebase { fs, hop });
    let scenario_max_id = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .filter_map(|action| match action {
            Action::Spawn { ids, .. } => ids.iter().copied().max(),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    pop.reserve_runtime_ids_through(scenario_max_id.max(reserve_runtime_ids_through));
    pop.set_seed(scenario.seed);
    pop.set_control_update_mode(scenario.control_update_mode);
    if reporter.is_some() {
        pop.enable_auto_observe();
    }
    let scaffold = scenario.scaffold;
    let meter_shaping = scenario.meter_shaping;
    let conductor = Conductor::from_scenario(scenario);

    let start_flag = Arc::new(AtomicBool::new(start_playing));

    let cfg = WorkerConfig {
        private_trace: config.temporal_private_trace,
        onset_comparison: config.temporal_onset_comparison,
        scenario_name: scenario_label,
        wait_user_exit,
        start_flag: start_flag.clone(),
        exiting: stop_flag,
        scaffold,
        meter_shaping,
        guard_meter,
        underrun_frames,
        audio_counters,
        dcc_coupler,
        hop,
        hop_duration,
        fs,
        deterministic_analysis,
        deterministic_footprints,
    };
    let channels = WorkerChannels {
        candidate_tables,
        reference_context,
        temporal_snapshots,
        ui_tx: ui_frame_tx,
        report_error_tx,
        audio_prod,
        wav_tx,
        audio_to_analysis_tx: Some(audio_to_analysis_tx),
        analysis_result_rx,
        analysis_update_tx,
        presentation_to_listener_tx,
        listener_result_rx,
        listener_analysis_update_tx,
    };

    let worker_handle = thread::Builder::new()
        .name("worker".into())
        .spawn(move || {
            let mut state = WorkerState::new(
                pop,
                conductor,
                landscape,
                lparams_runtime,
                dorsal,
                reporter,
                &cfg,
            );
            state.body_snapshot = body_capture.as_ref().map(|c| Box::new(c.snapshot()));
            state.schedule_renderer.body_capture = body_capture;
            state.profile = profile;
            #[cfg(test)]
            {
                state.offline_body_probe = offline_body_probe;
            }
            worker_loop(cfg, channels, state)
        })
        .expect("spawn worker");

    Ok(RuntimeWiring {
        ui_frame_rx,
        report_error_rx,
        start_flag,
        worker_handle,
        analysis_handle,
        listener_analysis_handle,
    })
}

pub(crate) fn init_runtime(
    args: crate::cli::Args,
    config: AppConfig,
    stop_flag: Arc<AtomicBool>,
) -> RuntimeInit {
    if let Err(err) = config.validate() {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
    crate::life::modal::register_modal();
    let latency_ms = config.audio.latency_ms;
    let guard_mode = resolve_limiter_mode(&config);
    let guard_meter = if args.play {
        Some(Arc::new(LimiterMeter::default()))
    } else {
        None
    };

    // Audio
    let (audio_out, audio_prod, audio_init_error) = if args.play {
        match AudioOutput::new(
            latency_ms,
            config.analysis.hop_size,
            guard_mode,
            guard_meter.clone(),
        ) {
            Ok((out, prod)) => (Some(out), Some(prod), None),
            Err(e) => {
                let msg = e.to_string();
                eprintln!("Audio init failed: {msg}");
                if args.nogui {
                    std::process::exit(1);
                }
                (None, None, Some(msg))
            }
        }
    } else {
        (None, None, None)
    };

    // Decide runtime sample rate: use actual stream rate when playing, otherwise config value.
    let runtime_sample_rate: u32 = if args.play {
        let device_rate = audio_out
            .as_ref()
            .map(|out| out.config.sample_rate)
            .unwrap_or(config.audio.sample_rate);
        if device_rate != config.audio.sample_rate {
            debug!(
                "Runtime sample rate overridden by device: {} (config {})",
                device_rate, config.audio.sample_rate
            );
        }
        device_rate
    } else {
        config.audio.sample_rate
    };
    if runtime_sample_rate != config.audio.sample_rate {
        let mut device_config = config.clone();
        device_config.audio.sample_rate = runtime_sample_rate;
        if let Err(err) = device_config.validate() {
            eprintln!("Audio device configuration invalid: {err:#}");
            std::process::exit(1);
        }
    }

    let hop_ms = config.analysis.hop_size as f32 / runtime_sample_rate as f32 * 1000.0;
    let visual_delay_frames = 0;
    debug!(
        "Visual delay frames: {} (latency_ms={:.1}, hop_ms={:.2})",
        visual_delay_frames, latency_ms, hop_ms
    );
    let ui_channel_capacity = (visual_delay_frames + 4).max(16);

    let path = Path::new(&args.scenario_path);
    let (scenario_label, scenario) = load_scenario_or_exit(path, &args, &config);
    let mut reporter = args
        .report
        .as_deref()
        .map(JsonlReporter::create)
        .transpose()
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        });
    report_try(&mut reporter, "meta", |writer| {
        writer.write_meta(scenario.seed)
    });
    report_try(&mut reporter, "scene markers", |writer| {
        writer.write_scene_markers(&scenario.scene_markers)
    });
    let observing = scenario.temporal_mode == crate::scenario::TemporalMode::Observe;
    let deterministic_analysis = (reporter.is_some() || observing) && !args.play;
    let profile = args.profile.as_deref().map(|path| {
        if let Some(report_path) = args.report.as_deref()
            && (Path::new(path) == Path::new(report_path)
                || Path::new(path)
                    .canonicalize()
                    .ok()
                    .is_some_and(|profile_path| {
                        Path::new(report_path).canonicalize().ok().as_ref() == Some(&profile_path)
                    }))
        {
            eprintln!("profile and report paths must differ");
            std::process::exit(1);
        }
        let coupling = DccCoupler::new(config.dcc).coupling_strength();
        RunProfile::create(
            path,
            scenario.seed,
            reporter.is_some(),
            coupling,
            observing || !args.nogui || args.report.is_some() || coupling > 0.0,
            runtime_sample_rate,
            config.analysis.hop_size,
            audio_out
                .as_ref()
                .map(|out| (out.device_info(), out.counters())),
        )
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        })
    });

    let wiring = wire_runtime(
        &config,
        runtime_sample_rate,
        scenario_label,
        scenario,
        stop_flag,
        WiringOptions {
            #[cfg(test)]
            offline_body_probe: None,
            ui_channel_capacity,
            listener_forced: !args.nogui || args.report.is_some(),
            wait_user_exit: config.playback.wait_user_exit,
            start_playing: !config.playback.wait_user_start,
            audio_prod,
            wav_tx: None,
            reporter,
            deterministic_analysis,
            deterministic_footprints: false,
            guard_meter,
            underrun_frames: audio_out.as_ref().map(AudioOutput::underrun_frames),
            reserve_runtime_ids_through: 0,
            profile,
            audio_counters: audio_out.as_ref().map(AudioOutput::counters),
        },
    )
    .unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });

    RuntimeInit {
        ui_frame_rx: wiring.ui_frame_rx,
        report_error_rx: wiring.report_error_rx,
        worker_handle: Some(wiring.worker_handle),
        analysis_handle: Some(wiring.analysis_handle),
        listener_analysis_handle: wiring.listener_analysis_handle,
        start_flag: wiring.start_flag,
        audio_out,
        audio_init_error,
        visual_delay_frames,
    }
}

pub fn run_headless(args: crate::cli::Args, config: AppConfig, stop_flag: Arc<AtomicBool>) {
    let rt = init_runtime(args, config, stop_flag);
    rt.start_flag.store(true, Ordering::SeqCst);
    let _audio = rt.audio_out;
    // Discarding the join result would let a panicked worker exit 0, which makes
    // headless script tests pass on a dead run. Surface it as a failure instead.
    let mut failures = Vec::new();
    for (name, handle) in [
        ("worker", rt.worker_handle),
        ("analysis", rt.analysis_handle),
        ("listener-analysis", rt.listener_analysis_handle),
    ] {
        if let Some(handle) = handle
            && let Err(err) = join_thread(name, handle)
        {
            failures.push(err);
        }
    }
    if let Some(rx) = rt.report_error_rx
        && let Ok(err) = rx.try_recv()
    {
        failures.push(err);
    }
    if let Some(audio) = _audio.as_ref() {
        let errors = audio
            .counters()
            .callback_errors_total
            .load(Ordering::Relaxed);
        if errors > 0 {
            failures.push(format!("audio callback failed: {errors} stream errors"));
        }
    }
    if !failures.is_empty() {
        for err in &failures {
            eprintln!("{err}");
        }
        std::process::exit(1);
    }
}

fn render_compile_args(scenario_path: &str, seed: Option<u64>) -> crate::cli::Args {
    crate::cli::Args {
        play: false,
        scenario_path: scenario_path.to_string(),
        config: "config.toml".to_string(),
        wait_user_exit: None,
        wait_user_start: None,
        nogui: true,
        compile_only: false,
        report: None,
        profile: None,
        seed,
    }
}

pub fn run_render(
    scenario_path: &str,
    wav_path: String,
    config: AppConfig,
    stop_flag: Arc<AtomicBool>,
    seed: Option<u64>,
    report_path: Option<&str>,
    reserve_runtime_ids_through: u64,
) -> Result<(), String> {
    if reserve_runtime_ids_through == u64::MAX {
        return Err("reserve-runtime-ids-through must be less than u64::MAX".to_string());
    }
    config.validate().map_err(|err| format!("{err:#}"))?;
    crate::life::modal::register_modal();

    let guard_mode = resolve_limiter_mode(&config);
    let guard_meter = Some(Arc::new(LimiterMeter::default()));

    let path = Path::new(scenario_path);
    if let Err(e) = validate_scenario_script_extension(path) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let compile_args = render_compile_args(scenario_path, seed);
    let (scenario_label, scenario) = load_scenario_or_exit(path, &compile_args, &config);
    if let Err(e) = validate_scenario(&scenario) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let mut reporter = report_path.map(JsonlReporter::create).transpose()?;
    if let Some(writer) = reporter.as_mut() {
        writer.write_meta(scenario.seed)?;
        writer.write_scene_markers(&scenario.scene_markers)?;
    }

    // Render mode never opens an audio device: use configured sample-rate directly.
    let runtime_sample_rate = config.audio.sample_rate;
    let (wav_tx, wav_rx) = bounded::<Arc<[f32]>>(16);
    let wav_handle = crate::audio::writer::WavOutput::run(
        wav_rx,
        wav_path,
        runtime_sample_rate,
        guard_mode,
        guard_meter.clone(),
    );

    let wiring = wire_runtime(
        &config,
        runtime_sample_rate,
        scenario_label,
        scenario,
        stop_flag,
        WiringOptions {
            #[cfg(test)]
            offline_body_probe: None,
            ui_channel_capacity: 1,
            listener_forced: reporter.is_some(),
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: Some(wav_tx),
            reporter,
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter,
            underrun_frames: None,
            reserve_runtime_ids_through,
            profile: None,
            audio_counters: None,
        },
    )?;

    join_thread("worker", wiring.worker_handle)?;
    join_thread("wav", wav_handle)?;
    join_thread("analysis", wiring.analysis_handle)?;
    if let Some(handle) = wiring.listener_analysis_handle {
        join_thread("listener-analysis", handle)?;
    }
    if let Some(rx) = wiring.report_error_rx
        && let Ok(err) = rx.try_recv()
    {
        return Err(err);
    }
    Ok(())
}

fn join_thread(name: &str, handle: thread::JoinHandle<()>) -> Result<(), String> {
    handle.join().map_err(|payload| {
        let detail = if let Some(msg) = payload.downcast_ref::<&str>() {
            (*msg).to_string()
        } else if let Some(msg) = payload.downcast_ref::<String>() {
            msg.clone()
        } else {
            "unknown panic payload".to_string()
        };
        format!("{name} thread failed: {detail}")
    })
}

/// Immutable per-run settings and shared control flags for the worker thread.
struct WorkerConfig {
    private_trace: Option<crate::config::TemporalPrivateTraceConfig>,
    onset_comparison: Option<crate::config::TemporalOnsetComparisonConfig>,
    scenario_name: String,
    wait_user_exit: bool,
    start_flag: Arc<AtomicBool>,
    exiting: Arc<AtomicBool>,
    scaffold: ScaffoldConfig,
    meter_shaping: crate::core::meter::MeterShaping,
    guard_meter: Option<Arc<LimiterMeter>>,
    underrun_frames: Option<Arc<AtomicU64>>,
    audio_counters: Option<Arc<crate::audio::output::AudioCallbackCounters>>,
    dcc_coupler: DccCoupler,
    hop: usize,
    hop_duration: Duration,
    fs: f32,
    /// Offline/report paths consume analysis with a fixed lag so results do not
    /// depend on worker scheduling. Real-time leaves the listener best-effort.
    deterministic_analysis: bool,
    deterministic_footprints: bool,
}

/// Channel endpoints and the audio ring-buffer producer owned by the worker.
struct WorkerChannels {
    candidate_tables: [Option<
        Arc<
            std::sync::Mutex<
                Option<Arc<crate::temporal_cognition::action_profiles::consumer::Publication>>,
            >,
        >,
    >; 2],
    reference_context:
        Option<Arc<std::sync::Mutex<crate::temporal_cognition::reference_inventory::Context>>>,
    temporal_snapshots:
        [Option<Arc<std::sync::Mutex<crate::temporal_cognition::observation::Snapshot>>>; 2],
    ui_tx: Sender<UiFrame>,
    report_error_tx: Option<Sender<String>>,
    audio_prod: Option<ringbuf::HeapProd<f32>>,
    wav_tx: Option<Sender<Arc<[f32]>>>,
    audio_to_analysis_tx: Option<Sender<(u64, Arc<[f32]>)>>,
    analysis_result_rx: Receiver<analysis_worker::AnalysisResult>,
    analysis_update_tx: Sender<LandscapeUpdate>,
    presentation_to_listener_tx: Option<Sender<(u64, Arc<[f32]>)>>,
    listener_result_rx: Option<Receiver<analysis_worker::AnalysisResult>>,
    listener_analysis_update_tx: Option<Sender<LandscapeUpdate>>,
}

/// Mutable worker state. Built once before the loop; hop phases take it `&mut`.
struct WorkerState {
    #[cfg(test)]
    offline_body_probe: Option<OfflineBodyProbe>,
    body_snapshot: Option<Box<crate::temporal_cognition::body::Snapshot>>,
    temporal_frames: Box<[crate::temporal_cognition::observation::Snapshot; 2]>,
    pop: Community,
    conductor: Conductor,
    current_landscape: LandscapeFrame,
    lparams: LandscapeParams,
    dorsal: DorsalStream,
    reporter: Option<JsonlReporter>,
    profile: Option<RunProfile>,
    timebase: crate::core::timebase::Timebase,
    playback_state: PlaybackState,
    finish_logged: bool,
    finished: bool,
    log_space: Log2Space,
    frame_idx: u64,
    monitor: AudioMonitor,
    last_guard_log: Instant,
    last_ui_update: Instant,
    last_tick_log: Instant,
    last_analysis_frame: Option<u64>,
    last_listener_analysis_frame: Option<u64>,
    listener_min_valid_frame: u64,
    listener_twin: ListenerTwin,
    latest_listener_fast_state: Option<ListenerFastState>,
    latest_listener_state: Option<ListenerState>,
    // Production-side meter core (auditory-motor coupling), separate from the
    // perception meter inside ListenerTwin. See "Perception vs Production".
    // Composer-set terrain priors (meter_stability / temporal_basin) seed it.
    prod_meter: MeterNetwork,
    temporal_expectation: Option<crate::core::temporal_expectation::AcousticTemporalExpectation>,
    participation_predictions: Vec<(u64, ParticipationPrediction)>,
    /// Traced decisions wait here so their encoding is charged to the report phase.
    participation_decisions: Vec<(
        u64,
        crate::life::temporal_participation::ParticipationDecision,
    )>,
    generator_model: crate::life::generator_model::GeneratorModel,
    hab_ecology: crate::core::habituation::HabituationField,
    hab_listener: crate::core::habituation::HabituationField,
    schedule_renderer: ScheduleRenderer,
    scenario_end_tick: Option<Tick>,
    phonation_batches_buf: Vec<PhonationBatch>,
}

impl WorkerState {
    fn current_time(&self) -> f32 {
        self.timebase
            .tick_to_sec(self.timebase.frame_start_tick(self.frame_idx))
    }

    fn new(
        pop: Community,
        conductor: Conductor,
        landscape: Landscape,
        lparams: LandscapeParams,
        dorsal: DorsalStream,
        reporter: Option<JsonlReporter>,
        cfg: &WorkerConfig,
    ) -> Self {
        let current_landscape: LandscapeFrame = landscape;
        let log_space = current_landscape.space.clone();
        let timebase = crate::core::timebase::Timebase {
            fs: cfg.fs,
            hop: cfg.hop,
        };
        let mut prod_meter = MeterNetwork::new();
        prod_meter.set_shaping(cfg.meter_shaping);
        let mut generator_model =
            crate::life::generator_model::GeneratorModel::new(timebase, log_space.clone());
        generator_model.advance_to(timebase.frame_start_tick(0));
        let hab_bins = current_landscape.space.n_bins();
        let hab_ecology = crate::core::habituation::HabituationField::new(
            &lparams.habituation,
            lparams.consonance_representation.theta,
            hab_bins,
        );
        let hab_listener = crate::core::habituation::HabituationField::new(
            &lparams.habituation,
            lparams.consonance_representation.theta,
            hab_bins,
        );
        let mut listener_twin = ListenerTwin::with_sample_rate(
            cfg.fs,
            crate::listener_twin::ListenerTwinConfig::default(),
        );
        if reporter.is_some() && !listener_twin.enable_contour_reporting(cfg.fs) {
            warn!("listener contour reporting unavailable below 2400 Hz sample rate");
        }
        let now = Instant::now();
        Self {
            #[cfg(test)]
            offline_body_probe: None,
            pop,
            temporal_frames: Default::default(),
            conductor,
            current_landscape,
            lparams,
            dorsal,
            reporter,
            profile: None,
            timebase,
            playback_state: PlaybackState::NotStarted,
            finish_logged: false,
            finished: false,
            log_space,
            frame_idx: 0,
            monitor: AudioMonitor::new(),
            last_guard_log: now - Duration::from_millis(200),
            last_ui_update: now,
            last_tick_log: now,
            last_analysis_frame: None,
            last_listener_analysis_frame: None,
            listener_min_valid_frame: 0,
            listener_twin,
            latest_listener_fast_state: None,
            latest_listener_state: None,
            prod_meter,
            temporal_expectation:
                crate::core::temporal_expectation::AcousticTemporalExpectation::new(cfg.fs as u32),
            participation_predictions: Vec::new(),
            participation_decisions: Vec::new(),
            generator_model,
            hab_ecology,
            hab_listener,
            schedule_renderer: ScheduleRenderer::new(timebase),
            scenario_end_tick: None,
            phonation_batches_buf: Vec::new(),
            body_snapshot: None,
        }
    }
}

fn worker_loop(cfg: WorkerConfig, mut channels: WorkerChannels, mut state: WorkerState) {
    state.schedule_renderer.profile = state.profile.as_ref().map(|_| Default::default());
    if channels.temporal_snapshots.iter().any(Option::is_some) {
        state.schedule_renderer.action_observer = Some(Box::new(
            crate::life::action_observation::Observer::new(cfg.fs as u32),
        ));
        if state.schedule_renderer.body_capture.is_some() {
            state
                .schedule_renderer
                .action_observer
                .as_mut()
                .unwrap()
                .enable_predictions();
            if cfg.deterministic_footprints
                && let Some(worker) = state
                    .schedule_renderer
                    .action_observer
                    .as_mut()
                    .and_then(|observer| observer.footprint_worker())
            {
                worker.deliver_footprints_deterministically();
            }
            if std::env::var_os("CONCHORDAL_DISABLE_CANDIDATE_ENERGY").is_some() {
                state
                    .schedule_renderer
                    .action_observer
                    .as_mut()
                    .unwrap()
                    .disable_candidates();
            }
        }
        if let Some(trace) = cfg.private_trace {
            state
                .schedule_renderer
                .action_observer
                .as_mut()
                .unwrap()
                .enable_trace(trace);
        }
    }
    if cfg.start_flag.load(Ordering::SeqCst) {
        state.playback_state = PlaybackState::Playing;
        if let Some(count) = cfg.underrun_frames.as_ref() {
            count.store(0, Ordering::Relaxed);
        }
    }
    let idle_silence = vec![0.0f32; cfg.hop];

    for (target, snapshot) in state
        .temporal_frames
        .iter_mut()
        .zip(&channels.temporal_snapshots)
    {
        if let Some(snapshot) = snapshot {
            *target = *snapshot.lock().expect("initial observation snapshot");
        }
    }
    let mut initial_frame = build_initial_ui_frame(
        &cfg.scenario_name,
        &state.conductor,
        &state.pop,
        &state.current_landscape,
        state.current_time(),
        state.playback_state,
        cfg.fs,
    );
    initial_frame.temporal = Arc::new(*state.temporal_frames);
    initial_frame.self_sound = state
        .schedule_renderer
        .action_observer
        .as_ref()
        .map(|o| o.snapshot);
    let _ = channels.ui_tx.try_send(initial_frame);

    loop {
        if cfg
            .audio_counters
            .as_ref()
            .is_some_and(|counters| counters.callback_errors_total.load(Ordering::Relaxed) > 0)
        {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
        if cfg.exiting.load(Ordering::SeqCst) || state.finished {
            if let Some(capture) = state.schedule_renderer.body_capture.as_mut() {
                capture.finish();
                let snapshot = capture.snapshot();
                if let Some(observer) = state.schedule_renderer.action_observer.as_mut() {
                    observer.observe_body(&snapshot);
                }
                if let Some(current) = state.body_snapshot.as_mut() {
                    **current = snapshot;
                }
                report_try(&mut state.reporter, "final body observation", |writer| {
                    writer.write_body_observation(&snapshot)
                });
            }
            if let Some(observer) = state.schedule_renderer.action_observer.as_mut() {
                observer.finish();
                for record in observer.drain_candidate_energy() {
                    report_try(
                        &mut state.reporter,
                        "final body candidate energy",
                        |writer| writer.write_candidate_energy(&record),
                    );
                }
                for record in observer.drain_traces() {
                    report_try(&mut state.reporter, "final private trace", |writer| {
                        writer.write_private_trace(&record)
                    });
                }
                for prediction in observer.drain_descriptor_predictions() {
                    report_try(
                        &mut state.reporter,
                        "final self sound descriptor prediction",
                        |writer| writer.write_descriptor_prediction(&prediction),
                    );
                }
                for outcome in observer.drain() {
                    report_try(&mut state.reporter, "final self sound outcome", |writer| {
                        writer.write_self_sound_outcome(&outcome)
                    });
                }
            }
            if channels.temporal_snapshots.iter().any(Option::is_some)
                && channels.audio_to_analysis_tx.is_some()
            {
                // Close input and drain publishers before reading their final observer state.
                // This runs after generation has stopped, never in a live hop or callback.
                channels.audio_to_analysis_tx.take();
                channels.presentation_to_listener_tx.take();
                for _ in &channels.analysis_result_rx {}
                if let Some(rx) = &channels.listener_result_rx {
                    for _ in rx {}
                }
                let now_sec = state.current_time();
                if let Some(observer) = state.schedule_renderer.action_observer.as_ref() {
                    report_try(
                        &mut state.reporter,
                        "final self sound observation",
                        |writer| writer.write_self_sound_observation(&observer.snapshot),
                    );
                }
                for (target, snapshot) in state
                    .temporal_frames
                    .iter_mut()
                    .zip(&channels.temporal_snapshots)
                {
                    if let Some(snapshot) = snapshot {
                        *target = *snapshot.lock().expect("final observation snapshot");
                        report_try(
                            &mut state.reporter,
                            "final temporal observation",
                            |writer| writer.write_temporal_observation(now_sec, target),
                        );
                    }
                }
                state.last_ui_update = Instant::now() - UI_MIN_INTERVAL;
                let now_tick = state.timebase.frame_start_tick(state.frame_idx);
                send_runtime_ui_frame(
                    &mut state,
                    &channels,
                    &cfg,
                    Arc::from([]),
                    0.0,
                    [0.0; 2],
                    now_tick,
                );
            }
            if cfg.exiting.load(Ordering::SeqCst) || !cfg.wait_user_exit {
                break;
            }
        }

        if state.playback_state == PlaybackState::NotStarted
            && !cfg.start_flag.load(Ordering::SeqCst)
        {
            thread::sleep(Duration::from_millis(10));
            continue;
        } else if state.playback_state == PlaybackState::NotStarted {
            state.playback_state = PlaybackState::Playing;
            if let Some(count) = cfg.underrun_frames.as_ref() {
                count.store(0, Ordering::Relaxed);
            }
        }

        if state.finished && cfg.wait_user_exit {
            if channels.temporal_snapshots.iter().any(Option::is_some)
                && state.last_ui_update.elapsed() >= UI_MIN_INTERVAL
            {
                let now_tick = state.timebase.frame_start_tick(state.frame_idx);
                send_runtime_ui_frame(
                    &mut state,
                    &channels,
                    &cfg,
                    Arc::from([]),
                    0.0,
                    [0.0; 2],
                    now_tick,
                );
                state.last_ui_update = Instant::now();
            }
            if let Some(prod) = channels.audio_prod.as_mut() {
                while prod.vacant_len() >= cfg.hop {
                    AudioOutput::push_samples(prod, &idle_silence);
                }
            }
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let mut produced_any = false;
        if channels.audio_prod.is_some() {
            while channels
                .audio_prod
                .as_ref()
                .is_some_and(|prod| prod.vacant_len() >= cfg.hop)
                && !cfg.exiting.load(Ordering::SeqCst)
                && !state.finished
            {
                process_hop(&cfg, &mut channels, &mut state);
                produced_any = true;
            }
        } else {
            // Offline/render-only mode: progress even without audio output.
            process_hop(&cfg, &mut channels, &mut state);
            produced_any = true;
        }

        if !produced_any {
            thread::sleep(Duration::from_millis(1));
        }
    }
    for (voice_id, prediction) in state.participation_predictions.drain(..) {
        let sample = participation_outcome(voice_id, cfg.fs as u32, prediction, None)
            .expect("end of input closes every pending observation");
        report_try(&mut state.reporter, "participation outcome", |writer| {
            writer.write_participation_outcome(&sample)
        });
    }
    report_try(&mut state.reporter, "rhythm summary", |writer| {
        writer.write_rhythm_summary()
    });
    report_try(
        &mut state.reporter,
        "listener confidence summary",
        |writer| writer.write_listener_confidence_summary(),
    );
    report_try(&mut state.reporter, "flush", |writer| writer.flush());
    let mut failure = state
        .reporter
        .as_mut()
        .and_then(|writer| writer.failure.take());
    if let Some(profile) = state.profile.as_mut() {
        use crate::runtime_profile::{BackgroundProfile, BodyWorkerProfile, SharedWorkerProfile};
        profile.background = Some(BackgroundProfile {
            scope: "terminal worker counters after drain/join; null means disabled, not zero cost; resource distributions cover all received frames and sample-clock 100ms windows, not the selected hop measurement interval; wall processing includes publication lock/copy, delivery includes queue wait, neither is thread CPU time; private startup is worker lane initialization; table/rejected are unused for private body; excludes final profile serialization",
            shared: std::array::from_fn(|bus| {
                channels.temporal_snapshots[bus].as_ref().map(|_| {
                    let snapshot = &state.temporal_frames[bus];
                    SharedWorkerProfile {
                        bus: snapshot.bus,
                        state: snapshot.state,
                        received_frames: snapshot.received_frames,
                        source_missing_samples: snapshot.source_missing_samples,
                        delivery_dropped_frames: snapshot.delivery_dropped_frames,
                        rejected_frames: snapshot.rejected_frames,
                        worker_resources: snapshot.worker_resources,
                        action_profile_resources: snapshot.action_profile_resources,
                    }
                })
            }),
            body: state
                .body_snapshot
                .as_ref()
                .map(|snapshot| BodyWorkerProfile {
                    finished: snapshot.finished,
                    input_end: snapshot.input_end,
                    processed_frames: snapshot.processed_frames,
                    invalid_hops: snapshot.invalid_hops,
                    capture_drops: snapshot.capture_drops,
                    outside_voice_hops: snapshot.outside_voice_hops,
                    worker_resources: snapshot.worker_resources,
                }),
            candidate_energy: state
                .schedule_renderer
                .action_observer
                .as_ref()
                .and_then(|observer| observer.snapshot.body_defaults)
                .map(|defaults| defaults.candidate_energy),
        });
    }
    if let Some(profile) = state.profile.take()
        && let Err(err) = profile.write()
    {
        failure = Some(match failure {
            Some(previous) => format!("{previous}; {err}"),
            None => err,
        });
    }
    if let Some(tx) = channels.report_error_tx
        && let Some(err) = failure
    {
        let _ = tx.send(err);
    }
}

/// One generated hop: analysis merge, ecology step, render, routing, telemetry.
fn process_hop(cfg: &WorkerConfig, channels: &mut WorkerChannels, state: &mut WorkerState) {
    let t_start = Instant::now();
    if state.profile.is_some() {
        begin_allocations();
    }
    let frame_idx = state.frame_idx;
    let buffer_capacity = channels
        .audio_prod
        .as_ref()
        .map(|prod| prod.capacity().get())
        .unwrap_or(cfg.hop);
    let occupancy = channels
        .audio_prod
        .as_ref()
        .map(|prod| buffer_capacity.saturating_sub(prod.vacant_len()))
        .unwrap_or(0);

    let now_tick = state.timebase.frame_start_tick(state.frame_idx);
    let now_sec = state.timebase.tick_to_sec(now_tick);
    state.generator_model.advance_to(now_tick);
    if state.frame_idx == 0 || state.last_tick_log.elapsed() >= Duration::from_secs(1) {
        debug!(
            "[tick] frame_idx={} now_tick={} now_sec={:.6}",
            state.frame_idx, now_tick, now_sec
        );
        state.last_tick_log = Instant::now();
    }
    state.pop.set_current_frame(state.frame_idx);

    let analysis_wait_start = Instant::now();
    let analysis_updated = wait_for_analysis(state, &channels.analysis_result_rx, &cfg.exiting);
    let analysis_wait = analysis_wait_start.elapsed();
    let landscape_start = state.profile.as_ref().map(|_| Instant::now());
    apply_landscape_updates(state, channels, cfg, analysis_updated, now_tick, now_sec);
    let landscape_update = landscape_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let listener_wait_start = Instant::now();
    let listener_state_update = wait_for_listener(state, channels, cfg);
    let listener_wait = listener_wait_start.elapsed();
    if let Some(listener_state) = listener_state_update {
        state.latest_listener_state = listener_state;
    }
    let listener_pressure = cfg.dcc_coupler.pressure(state.latest_listener_state);
    let listener_pressure_update =
        listener_state_update.map(|listener_state| cfg.dcc_coupler.pressure(listener_state));

    for (target, snapshot) in state
        .temporal_frames
        .iter_mut()
        .zip(&channels.temporal_snapshots)
    {
        if let Some(snapshot) = snapshot
            && let Ok(current) = snapshot.try_lock()
        {
            *target = *current;
        }
    }

    if let Some(observer) = state.schedule_renderer.action_observer.as_mut() {
        let tables = std::array::from_fn(|bus| {
            let expected = state.temporal_frames[bus]
                .action_profile_features
                .as_ref()
                .map(crate::temporal_cognition::action_profiles::consumer::Key::from)?;
            let published = channels.candidate_tables[bus].as_ref()?.try_lock().ok()?;
            published
                .as_ref()
                .filter(|p| p.key == expected && p.key.issued_at <= now_tick)
                .cloned()
        });
        observer.shared_candidates(tables);
    }

    if let Some(context) = &channels.reference_context
        && let Ok(context) = context.try_lock()
        && let Some(observer) = state.schedule_renderer.action_observer.as_mut()
    {
        observer.trace_context(context.clone(), now_tick);
    }
    let population_start = state.profile.as_ref().map(|_| Instant::now());
    let phonation_count = advance_population(state, cfg, now_tick, listener_pressure);
    let population_elapsed = population_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    let reports_start = state.profile.as_ref().map(|_| Instant::now());
    emit_hop_reports(
        state,
        cfg,
        now_sec,
        phonation_count,
        listener_state_update.flatten(),
        listener_pressure_update,
    );
    let reports_elapsed = reports_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let render_start = state.profile.as_ref().map(|_| Instant::now());
    let (presentation_chunk, habitat_chunk, max_abs, synthesis_elapsed) =
        render_and_route_audio(state, channels, cfg, now_tick, phonation_count);
    let render_route = render_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    let post_render_start = state.profile.as_ref().map(|_| Instant::now());
    let channel_peak = [max_abs, max_abs];

    drive_production_meter(state, cfg, habitat_chunk.as_ref());
    state.latest_listener_fast_state = Some(state.listener_twin.observe_presentation_audio(
        now_sec,
        state.frame_idx,
        presentation_chunk.as_ref(),
    ));
    state.listener_twin.observe_contour_audio(
        state.timebase.frame_start_tick(state.frame_idx),
        presentation_chunk.as_ref(),
        |event| {
            let sample = ListenerContourSample {
                generated_frame_id: state.frame_idx,
                time_sec: event.available_sec,
                event: event.kind,
                onset_sec: event.onset_sec,
                periodic_frequency_hz: event.periodic_frequency_hz,
                periodicity: event.periodicity,
                delta_log2: event.score.and_then(|score| score.delta_log2),
                gain_bits: event.score.and_then(|score| score.gain_bits),
                loss_bits: event.score.and_then(|score| score.loss_bits),
                context_support: event.score.map(|score| score.context_support),
                error_threshold_bits: event.score.and_then(|score| score.error_threshold_bits),
                calibration_events: event.score.map(|score| score.calibration_events),
                error_candidate: event.score.is_some_and(|score| score.error_candidate),
                gap_threshold_sec: event.gap_threshold_sec,
                missing_start_sec: event.missing_start_sec,
            };
            report_try(&mut state.reporter, "listener contour", |writer| {
                writer.write_listener_contour(&sample)
            });
        },
    );

    // Feed every habitat hop to NSGT-RT. Dropping hops breaks time continuity.
    if let Err(err) = channels
        .audio_to_analysis_tx
        .as_ref()
        .expect("analysis input open during generation")
        .send((state.frame_idx, Arc::clone(&habitat_chunk)))
    {
        warn!("analysis worker disconnected: {err}");
        cfg.exiting.store(true, Ordering::SeqCst);
    }

    // Lag is measured against generated frames, because population dynamics depend on
    // the landscape evolution in the generated timebase (not wall-clock playback).
    // landscape_age in frames (spec: <= MAX_LANDSCAPE_LAG_FRAMES after warmup).
    let analysis_lag = state
        .last_analysis_frame
        .map(|id| state.frame_idx.saturating_sub(id));

    let finished_now = if state.pop.abort_requested {
        true
    } else {
        state.scenario_end_tick.is_some() && state.schedule_renderer.is_idle()
    };
    if finished_now {
        state.playback_state = PlaybackState::Finished;
    }
    if finished_now && channels.wav_tx.is_some() {
        channels.wav_tx.take();
        info!("[t={:.6}] WAV closed.", state.current_time());
    }

    let must_send_ui =
        state.conductor.is_done() || state.pop.abort_requested || state.scenario_end_tick.is_some();
    let should_send_ui = must_send_ui || state.last_ui_update.elapsed() >= UI_MIN_INTERVAL;

    let peak_level = state.monitor.max_peak.max(max_abs);

    log_guard_meter(cfg, &mut state.last_guard_log, now_sec);

    if should_send_ui {
        send_runtime_ui_frame(
            state,
            channels,
            cfg,
            presentation_chunk,
            peak_level,
            channel_peak,
            now_tick,
        );
    }

    if finished_now {
        state.finished = true;
        if !state.finish_logged {
            let note = if cfg.wait_user_exit {
                "Waiting for user exit."
            } else {
                "Exiting."
            };
            info!("[t={:.6}] Scenario finished. {note}", state.current_time());
            state.finish_logged = true;
        }
        if !cfg.wait_user_exit {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
    }

    state.monitor.update(
        state.current_time(),
        state.frame_idx,
        cfg.hop,
        cfg.hop_duration,
        buffer_capacity,
        occupancy,
        max_abs,
        t_start.elapsed(),
        analysis_wait,
        listener_wait,
        cfg.underrun_frames.as_deref(),
        analysis_lag,
        state.conductor.is_done(),
    );

    if !state.finished {
        state.frame_idx += 1;
    }
    // Include preceding telemetry in the hop cost, but exclude this record's own write.
    report_try(&mut state.reporter, "hop timing", |writer| {
        writer.write_hop_timing(&HopTimingSample {
            frame_idx,
            time_sec: now_sec,
            elapsed_us: t_start.elapsed().as_secs_f64() * 1_000_000.0,
            analysis_wait_us: analysis_wait.as_secs_f64() * 1_000_000.0,
            listener_wait_us: listener_wait.as_secs_f64() * 1_000_000.0,
            hop_budget_us: cfg.hop_duration.as_secs_f64() * 1_000_000.0,
            audio_output: if cfg.underrun_frames.is_some() {
                "device"
            } else {
                "no_device"
            },
            underrun_frames_total: cfg
                .underrun_frames
                .as_ref()
                .map(|count| count.load(Ordering::Relaxed)),
        })
    });
    if let Some(profile) = state.profile.as_mut() {
        let alive_voice_count = state
            .pop
            .voices
            .iter()
            .filter(|voice| voice.is_alive())
            .count();
        let post_render_us = post_render_start
            .map(|start| start.elapsed().as_secs_f64() * 1_000_000.0)
            .unwrap_or_default();
        let elapsed_us = t_start.elapsed().as_secs_f64() * 1_000_000.0;
        let worker_allocations = finish_allocations();
        profile.record(HopProfile {
            frame_idx,
            time_sec: now_sec,
            alive_voice_count,
            elapsed_us,
            analysis_wait_us: analysis_wait.as_secs_f64() * 1_000_000.0,
            listener_wait_us: listener_wait.as_secs_f64() * 1_000_000.0,
            landscape_update_us: landscape_update.as_secs_f64() * 1_000_000.0,
            population_us: population_elapsed.as_secs_f64() * 1_000_000.0,
            reports_us: reports_elapsed.as_secs_f64() * 1_000_000.0,
            render_route_us: render_route.as_secs_f64() * 1_000_000.0,
            synthesis_us: synthesis_elapsed.as_secs_f64() * 1_000_000.0,
            rendering: state.schedule_renderer.profile,
            rendered_tone_count: state.schedule_renderer.active_tone_count(),
            post_render_us,
            worker_allocations,
            underrun_frames_total: cfg
                .underrun_frames
                .as_ref()
                .map(|count| count.load(Ordering::Relaxed)),
        });
    }
}

/// Drain analysis results into the landscape.
/// Spec: landscape may lag analysis by <= MAX_LANDSCAPE_LAG_FRAMES; block until
/// the landscape is fresh enough or the run is ending.
fn wait_for_analysis(
    state: &mut WorkerState,
    analysis_result_rx: &Receiver<analysis_worker::AnalysisResult>,
    exiting: &AtomicBool,
) -> bool {
    let mut analysis_updated = false;
    loop {
        let (merged, analysis_unavailable) = merge_latest_analysis_results(
            analysis_result_rx,
            &mut state.current_landscape,
            &mut state.log_space,
            &mut state.generator_model,
            &state.lparams,
            &mut state.last_analysis_frame,
            state.frame_idx,
        );
        if merged {
            analysis_updated = true;
        }

        if analysis_ok(
            state.frame_idx,
            state.last_analysis_frame,
            MAX_LANDSCAPE_LAG_FRAMES,
        ) {
            break;
        }
        if analysis_unavailable {
            eprintln!("Field analysis disconnected or invalidated; ending run.");
            exiting.store(true, Ordering::SeqCst);
            break;
        }
        if exiting.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_micros(200));
    }
    analysis_updated
}

/// Apply pending kernel-parameter updates, advance ecology habituation, and
/// refresh the generator model's observed field and rhythm gate.
fn apply_landscape_updates(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
    analysis_updated: bool,
    now_tick: Tick,
    now_sec: f32,
) {
    let landscape_params_updated = apply_pending_landscape_update(
        &mut state.pop,
        &mut state.lparams,
        &mut state.current_landscape,
        &channels.analysis_update_tx,
        channels.listener_analysis_update_tx.as_ref(),
    );
    drive_and_apply_habituation(
        &mut state.current_landscape,
        &mut state.hab_ecology,
        &state.lparams,
        cfg.hop_duration.as_secs_f32(),
    );
    if (analysis_updated || landscape_params_updated)
        && let Some(analysis_id) = state.last_analysis_frame
    {
        // NSGT is right-aligned; the observed scan is stamped at the analysis frame end.
        let obs_tick = state.timebase.frame_end_tick(analysis_id);
        state.generator_model.observe_consonance_field_level(
            obs_tick,
            Arc::from(&state.current_landscape.consonance_field_level_eff[..]),
        );
    }

    apply_scaffold(
        &mut state.current_landscape.rhythm,
        cfg.scaffold,
        now_sec,
        state.frame_idx,
    );
    state
        .generator_model
        .update_gate_from_rhythm(now_tick, &state.current_landscape.rhythm);
}

/// Drain listener analysis results. Deterministic render mirrors the
/// main-analysis fixed-lag wait so the listener frame consumed at render-frame
/// N does not depend on worker scheduling. Real-time keeps the single
/// best-effort batch to avoid stalling the generator on listener-analysis
/// latency. Memory processes every received observation before publishing the
/// latest state; input backpressure still invalidates dropped presentation audio.
fn wait_for_listener(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
) -> Option<Option<ListenerState>> {
    let mut listener_state_update: Option<Option<ListenerState>> = None;
    loop {
        let (listener_state, listener_disconnected) = observe_listener_analysis_results(
            channels.listener_result_rx.as_ref(),
            &mut state.listener_twin,
            &state.lparams,
            &mut state.hab_listener,
            state.timebase,
            state.frame_idx,
            state.listener_min_valid_frame,
            &mut state.last_listener_analysis_frame,
        );
        if let Some(listener_state) = listener_state {
            listener_state_update = Some(listener_state);
        }
        if !cfg.deterministic_analysis || channels.listener_result_rx.is_none() {
            break;
        }
        if matches!(listener_state, Some(None)) {
            eprintln!("Deterministic listener analysis invalidated; ending run.");
            cfg.exiting.store(true, Ordering::SeqCst);
            break;
        }
        if analysis_ok(
            state.frame_idx,
            state.last_listener_analysis_frame,
            MAX_LANDSCAPE_LAG_FRAMES,
        ) {
            break;
        }
        if listener_disconnected {
            eprintln!("Listener analysis thread stopped delivering results; ending run.");
            cfg.exiting.store(true, Ordering::SeqCst);
            break;
        }
        if cfg.exiting.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_micros(200));
    }
    listener_state_update
}

/// Dispatch scenario events, collect phonation batches, and step the ecology.
/// Returns the number of valid batches in `state.phonation_batches_buf`.
fn advance_population(
    state: &mut WorkerState,
    cfg: &WorkerConfig,
    now_tick: Tick,
    listener_pressure: ListenerPressure,
) -> usize {
    state.conductor.dispatch_until(
        state.current_time(),
        state.frame_idx,
        &state.current_landscape,
        None::<&mut crate::core::stream::analysis::AnalysisStream>,
        &mut state.pop,
    );
    if let Some(capture) = state.schedule_renderer.body_capture.as_mut() {
        capture.prepare(
            state
                .pop
                .voices
                .iter()
                .map(|v| (v.id(), v.metadata.generation, v.body_snapshot())),
            now_tick,
        );
    }
    if cfg.onset_comparison.is_some() {
        for voice in &mut state.pop.voices {
            voice
                .phonation_engine
                .set_onset_comparison(cfg.onset_comparison);
            voice
                .phonation_engine
                .set_decision_trace(state.reporter.is_some());
        }
        request_body_footprints(state, cfg.fs, now_tick);
    }
    if let Some(observer) = state.temporal_expectation.as_mut() {
        // Track from birth, including before a later switch into participation.
        state.schedule_renderer.prepare_self_sound(
            state.pop.voices.iter().map(|voice| voice.id()),
            now_tick,
            observer,
        );
        for voice in &mut state.pop.voices {
            let id = voice.id();
            if let Some(own) = state.schedule_renderer.self_sound_history(id) {
                voice
                    .phonation_engine
                    .observe_participation_context(own, |update| {
                        report_try(&mut state.reporter, "participation context", |writer| {
                            writer.write_participation_context(id, cfg.fs as u32, &update)
                        });
                    });
            }
        }
        let frame_end = now_tick.saturating_add(cfg.hop as Tick);
        let needs_forecast = state.pop.voices.iter().any(|voice| {
            voice
                .phonation_engine
                .needs_temporal_context(now_tick, frame_end)
        });
        let shared = needs_forecast.then(|| observer.forecast()).flatten();
        for voice in &mut state.pop.voices {
            if voice
                .phonation_engine
                .needs_temporal_context(now_tick, frame_end)
            {
                let mut local = shared;
                // Audit the unmodified habitat forecast; leave-self-out is a decision view.
                voice
                    .phonation_engine
                    .set_outcome_forecast(state.reporter.as_ref().and(shared.as_ref()));
                let mut own = state.schedule_renderer.self_sound_history(voice.id());
                if let (Some(local), Some(own)) = (local.as_mut(), own.as_deref_mut()) {
                    observer.use_external_energy(local, &mut own.external);
                }
                voice
                    .phonation_engine
                    .set_temporal_context(local.as_ref(), own.map_or([0.0; 3], |own| own.profile));
            }
        }
    }
    let phonation_count = if state.scenario_end_tick.is_none() {
        state.pop.collect_phonation_batches_into(
            &mut state.generator_model,
            &state.current_landscape,
            now_tick,
            &mut state.phonation_batches_buf,
        )
    } else {
        0
    };

    if let Some(acoustic) = state.temporal_expectation.as_ref() {
        state.schedule_renderer.prepare_energy_contexts(
            acoustic,
            &state.phonation_batches_buf[..phonation_count],
            now_tick,
        );
    }

    if state.reporter.is_some() {
        for voice in &mut state.pop.voices {
            let id = voice.id();
            state.participation_predictions.extend(
                voice
                    .phonation_engine
                    .drain_participation_predictions()
                    .map(|prediction| (id, prediction)),
            );
            state.participation_decisions.extend(
                voice
                    .phonation_engine
                    .drain_participation_decisions()
                    .map(|decision| (id, decision)),
            );
        }
    }

    if state.scenario_end_tick.is_none() && state.conductor.is_done() {
        state.scenario_end_tick = Some(now_tick);
        state.pop.voices.clear();
        state.schedule_renderer.shutdown_at(now_tick);
    }

    state.pop.advance_with_listener_pressure(
        cfg.hop,
        state.frame_idx,
        cfg.hop_duration.as_secs_f32(),
        &state.current_landscape,
        listener_pressure,
    );
    state.pop.cleanup_dead(
        state.frame_idx,
        cfg.hop_duration.as_secs_f32(),
        state.conductor.is_done(),
        &state.current_landscape,
    );
    phonation_count
}

/// I11-1 §4.2: at most one representative-onset footprint request per Voice per hop.
/// The hop path carries the identity, the request and the reply; the projection
/// itself runs on the candidate worker.
fn request_body_footprints(state: &mut WorkerState, fs: f32, now: Tick) {
    let voices = &mut state.pop.voices;
    let renderer = &mut state.schedule_renderer;
    let (Some(capture), Some(observer)) = (
        renderer.body_capture.as_ref(),
        renderer.action_observer.as_mut(),
    ) else {
        return;
    };
    let Some(worker) = observer.footprint_worker() else {
        return;
    };
    for voice in voices {
        if !voice.is_alive() {
            continue;
        }
        let Some(recipe) = voice.footprint_recipe(fs) else {
            continue;
        };
        let Some((_, body_generation)) = capture.token(voice.id(), voice.metadata.generation)
        else {
            continue;
        };
        let identity = crate::life::action_candidates::footprint::Identity::new(
            voice.id(),
            body_generation,
            &recipe,
        );
        let resent = voice
            .phonation_engine
            .footprint_hop(identity, now, &recipe, |request| {
                worker.request_footprint(request)
            });
        if resent {
            worker.stats.footprint_resent += 1;
        }
    }
}

/// Route returned footprints to the Voice that asked for them (I11-1 §4.2).
/// A reply the full queue dropped releases the request, and the Voice resends it next hop.
fn route_body_footprints(
    voices: &mut [crate::life::voice::Voice],
    observer: &mut crate::life::action_observation::Observer,
    reporter: &mut Option<JsonlReporter>,
    now: Tick,
) {
    let Some(worker) = observer.footprint_worker() else {
        return;
    };
    let mut superseded = 0;
    for record in worker.drain_footprints(now) {
        let discarded = voices
            .iter_mut()
            .find(|voice| voice.id() == record.identity.source_id)
            .is_some_and(|voice| voice.phonation_engine.footprint_receive(&record));
        superseded += u64::from(discarded);
        report_try(reporter, "body footprint", |writer| {
            writer.write_body_footprint(&record, discarded)
        });
    }
    worker.stats.footprint_superseded += superseded;
    let mut released = 0;
    for identity in worker.drain_released_footprints() {
        if let Some(voice) = voices
            .iter_mut()
            .find(|voice| voice.id() == identity.source_id)
            && voice.phonation_engine.footprint_release(identity)
        {
            released += 1;
        }
    }
    worker.stats.footprint_released += released;
}

/// Emit per-hop JSONL report records. No-op without a reporter.
fn emit_hop_reports(
    state: &mut WorkerState,
    cfg: &WorkerConfig,
    now_sec: f32,
    phonation_count: usize,
    listener_state_update: Option<ListenerState>,
    listener_pressure_update: Option<ListenerPressure>,
) {
    if state.reporter.is_none() {
        return;
    }
    for (voice_id, decision) in state.participation_decisions.drain(..) {
        report_try(&mut state.reporter, "participation decision", |writer| {
            writer.write_participation_decision(voice_id, cfg.fs as u32, &decision)
        });
    }
    // Only reporting is decimated; each analysis hop has already advanced history.
    let report_hops = (0.1 * cfg.fs / cfg.hop as f32).ceil().max(1.0) as u64;
    if state.frame_idx.is_multiple_of(report_hops) {
        for observation in state.temporal_frames.iter() {
            if observation.state != crate::temporal_cognition::observation::ObservationState::Off {
                report_try(&mut state.reporter, "temporal observation", |writer| {
                    writer.write_temporal_observation(now_sec, observation)
                });
            }
        }
        for (bus, history) in [
            (
                "habitat",
                state.current_landscape.spectral_history.as_deref(),
            ),
            (
                "presentation",
                state.listener_twin.spectral_history.as_deref(),
            ),
        ] {
            if let Some(snapshot) = history {
                report_try(&mut state.reporter, "spectral history", |writer| {
                    writer.write_spectral_history(bus, state.frame_idx * cfg.hop as u64, snapshot)
                });
            }
        }
    }
    let onset_samples = onset_samples_from_batches(
        &state.pop.voices,
        &state.phonation_batches_buf[..phonation_count],
        cfg.fs,
        cfg.scaffold,
        state.frame_idx,
    );
    let runtime_events = state.pop.drain_runtime_events();
    let phonation_gate_open_events = state.pop.drain_phonation_gate_open_events();
    let death_records = state.pop.take_death_records();
    let active_population_ids = state.pop.active_population_ids();
    let population_steps = summarize_populations(
        &state.pop.voices,
        &active_population_ids,
        &state.current_landscape,
        now_sec,
    );
    let kuramoto = state.pop.kuramoto_order_parameter();
    let rhythm_observation = RhythmObservation {
        time_sec: now_sec,
        kuramoto_order_r: kuramoto.map(|(r, _)| r),
        kuramoto_active_count: kuramoto.map(|(_, n)| n).unwrap_or(0),
        onsets_in_hop: onset_samples.len(),
        theta_hz: positive_hz(state.current_landscape.rhythm.theta.freq_hz),
        delta_hz: positive_hz(state.current_landscape.rhythm.delta.freq_hz),
        env_open: state.current_landscape.rhythm.env_open.clamp(0.0, 1.0),
        env_level: state.current_landscape.rhythm.env_level.clamp(0.0, 1.0),
        measure_hz: positive_hz(state.current_landscape.rhythm.measure.freq_hz),
        measure_phase: state.current_landscape.rhythm.measure.phase,
        measure_confidence: state.current_landscape.rhythm.measure.alpha,
        measure_ratio: state.current_landscape.rhythm.measure_ratio,
    };
    report_try(&mut state.reporter, "runtime events", |writer| {
        writer.write_runtime_events(&runtime_events)
    });
    report_try(&mut state.reporter, "phonation gate open", |writer| {
        writer.write_phonation_gate_opens(&phonation_gate_open_events)
    });
    report_try(&mut state.reporter, "deaths", |writer| {
        writer.write_deaths(&death_records, cfg.hop, cfg.fs)
    });
    report_try(&mut state.reporter, "onsets", |writer| {
        writer.write_onsets(&onset_samples)
    });
    report_try(&mut state.reporter, "population steps", |writer| {
        writer.write_population_steps(&population_steps)
    });
    report_try(&mut state.reporter, "rhythm observation", |writer| {
        writer.write_rhythm_observation(rhythm_observation)
    });
    let hstate = &state.current_landscape.perc_habituation_state_scan;
    let n = hstate.len().max(1);
    let mean_h = hstate.iter().sum::<f32>() / n as f32;
    let max_h = hstate.iter().copied().fold(0.0f32, f32::max);
    let mut mean_erosion = 0.0f32;
    for i in 0..state.current_landscape.consonance_field_score.len() {
        mean_erosion += state.current_landscape.consonance_field_score[i]
            - state.current_landscape.consonance_field_score_eff[i];
    }
    mean_erosion /= n as f32;
    let tracked_bin = state
        .current_landscape
        .consonance_field_score
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            },
        )
        .0;
    let tracked_state = state.current_landscape.perc_habituation_state_scan[tracked_bin];
    let tracked_score = state.current_landscape.consonance_field_score[tracked_bin];
    let tracked_score_eff = state.current_landscape.consonance_field_score_eff[tracked_bin];
    report_try(&mut state.reporter, "habituation", |w| {
        w.write_habituation(
            now_sec,
            mean_h,
            max_h,
            mean_erosion,
            tracked_bin,
            tracked_state,
            tracked_score,
            tracked_score_eff,
        )
    });
    let now_tick = state.timebase.frame_start_tick(state.frame_idx);
    let second_ticks = state.timebase.sec_to_tick(1.0).max(1);
    if state.frame_idx == 0
        || now_tick / second_ticks != now_tick.saturating_sub(cfg.hop as Tick) / second_ticks
    {
        report_try(&mut state.reporter, "habituation scan", |writer| {
            writer.write_habituation_scan(now_sec, &state.current_landscape)
        });
    }
    if let Some(listener_state) = listener_state_update {
        let sample = ListenerStateSample {
            time_sec: listener_state.time_sec,
            generated_frame_id: listener_state.generated_frame_id,
            analysis_frame_id: listener_state.analysis_frame_id,
            analysis_lag_frames: listener_state.analysis_lag_frames,
            stability_level: listener_state.stability_level,
            resolvability_level: listener_state.resolvability_level,
            tension_level: listener_state.tension_level,
            attention_level: listener_state.attention_level,
            beat_hz: listener_state.beat_hz,
            beat_phase: listener_state.beat_phase,
            beat_confidence: listener_state.beat_confidence,
            subdivision_ratio: listener_state.subdivision_ratio,
            subdivision_confidence: listener_state.subdivision_confidence,
            measure_hz: listener_state.measure_hz,
            measure_ratio: listener_state.measure_ratio,
            measure_confidence: listener_state.measure_confidence,
        };
        report_try(&mut state.reporter, "listener state", |writer| {
            writer.write_listener_state(&sample)
        });
    }
    if let Some(pressure) = listener_pressure_update {
        report_try(&mut state.reporter, "dcc pressure", |writer| {
            writer.write_dcc_pressure(now_sec, pressure)
        });
    }
}

/// Render one hop and route it to the two mono buses:
///   presentation_chunk -> cpal output + wav + UI metering
///   habitat_chunk -> generator rhythm + NSGT analysis
/// Returns both buses, their peak, and renderer time (including own observation and match reports).
fn render_and_route_audio(
    state: &mut WorkerState,
    channels: &mut WorkerChannels,
    cfg: &WorkerConfig,
    now_tick: Tick,
    phonation_count: usize,
) -> (Arc<[f32]>, Arc<[f32]>, f32, Duration) {
    let synthesis_start = state.profile.as_ref().map(|_| Instant::now());
    let frame = state.schedule_renderer.render_with_prediction_matches(
        &state.phonation_batches_buf[..phonation_count],
        now_tick,
        &state.current_landscape.rhythm,
        |id, start, window, matched| {
            report_try(&mut state.reporter, "local prediction match", |writer| {
                writer.write_local_prediction_match(id, cfg.fs as u32, start, window, matched)
            });
        },
    );
    let synthesis_elapsed = synthesis_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let mut max_p = 0.0f32;
    for &s in frame.presentation {
        let abs_s = s.abs();
        if abs_s > max_p {
            max_p = abs_s;
        }
    }

    if let Some(prod) = channels.audio_prod.as_mut() {
        AudioOutput::push_samples(prod, frame.presentation);
    }

    let presentation_chunk: Arc<[f32]> = Arc::from(frame.presentation);
    let habitat_chunk: Arc<[f32]> = Arc::from(frame.habitat);
    #[cfg(test)]
    if let Some(mut probe) = state.offline_body_probe.take() {
        probe(
            state,
            now_tick,
            phonation_count,
            [&habitat_chunk, &presentation_chunk],
        );
        state.offline_body_probe = Some(probe);
    }
    if let Some(capture) = state.schedule_renderer.body_capture.as_ref() {
        let snapshot = capture.snapshot();
        if let Some(observer) = state.schedule_renderer.action_observer.as_mut() {
            observer.observe_body(&snapshot);
        }
        if let Some(current) = state.body_snapshot.as_mut()
            && (current.version != snapshot.version
                || current.capture_drops != snapshot.capture_drops
                || current.outside_voice_hops != snapshot.outside_voice_hops)
        {
            **current = snapshot;
            report_try(&mut state.reporter, "body observation", |writer| {
                writer.write_body_observation(&snapshot)
            });
        }
    }
    if let Some(observer) = state.schedule_renderer.action_observer.as_mut() {
        // Drain regardless of reporting; snapshots and capacity remain observer-owned.
        for record in observer.drain_candidate_energy() {
            report_try(&mut state.reporter, "body candidate energy", |writer| {
                writer.write_candidate_energy(&record)
            });
        }
        for record in observer.drain_body_defaults() {
            report_try(&mut state.reporter, "body default", |writer| {
                writer.write_body_default(&record)
            });
        }
        route_body_footprints(
            &mut state.pop.voices,
            observer,
            &mut state.reporter,
            now_tick,
        );
        for record in observer.drain_traces() {
            report_try(&mut state.reporter, "private trace", |writer| {
                writer.write_private_trace(&record)
            });
        }
        for prediction in observer.drain_descriptor_predictions() {
            report_try(
                &mut state.reporter,
                "self sound descriptor prediction",
                |writer| writer.write_descriptor_prediction(&prediction),
            );
        }
        for outcome in observer.drain() {
            report_try(&mut state.reporter, "self sound outcome", |writer| {
                writer.write_self_sound_outcome(&outcome)
            });
        }
    }
    if state.reporter.is_some() {
        // Drain after actual rendering, before a later hop can retire a Voice.
        state
            .schedule_renderer
            .drain_prediction_errors(|id, from, through, window, errors| {
                report_try(&mut state.reporter, "local prediction errors", |writer| {
                    writer.write_local_prediction_errors(
                        id,
                        cfg.fs as u32,
                        from,
                        through,
                        window,
                        errors,
                    )
                });
            });
    }
    if let Some(tx) = channels.wav_tx.as_ref()
        && let Err(err) = tx.send(Arc::clone(&presentation_chunk))
    {
        warn!("WAV render output disconnected: {err}");
        cfg.exiting.store(true, Ordering::SeqCst);
    }
    if let Some(tx) = channels.presentation_to_listener_tx.as_ref() {
        if cfg.deterministic_analysis {
            // Never drop a presentation hop: the listener worker must
            // observe the same hop sequence every run. Blocking applies
            // backpressure but cannot deadlock, because the fixed-lag
            // consume above keeps the worker at most one frame behind.
            if let Err(err) = tx.send((state.frame_idx, Arc::clone(&presentation_chunk))) {
                warn!("listener analysis worker disconnected: {err}");
                channels.presentation_to_listener_tx = None;
            }
        } else {
            match tx.try_send((state.frame_idx, Arc::clone(&presentation_chunk))) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    state.latest_listener_state = None;
                    state.listener_twin.spectral_history = None;
                    state.last_listener_analysis_frame = None;
                    state.listener_min_valid_frame = state.frame_idx.saturating_add(1);
                    debug!("listener analysis backlog full; dropped presentation hop");
                }
                Err(TrySendError::Disconnected(_)) => {
                    warn!("listener analysis worker disconnected");
                    channels.presentation_to_listener_tx = None;
                    state.latest_listener_state = None;
                    state.listener_twin.spectral_history = None;
                }
            }
        }
    }

    (presentation_chunk, habitat_chunk, max_p, synthesis_elapsed)
}

fn participation_outcome(
    voice_id: u64,
    sample_rate: u32,
    prediction: ParticipationPrediction,
    observation: Option<TemporalObservation>,
) -> Option<ParticipationOutcomeSample> {
    if observation.is_some_and(|o| o.end_frame <= prediction.target_start_frame) {
        return None;
    }
    let matched = observation.is_some_and(|o| {
        o.start_frame == prediction.target_start_frame && o.end_frame == prediction.target_end_frame
    });
    Some(ParticipationOutcomeSample {
        voice_id,
        sample_rate,
        prediction,
        status: if matched {
            "observed"
        } else if observation.is_some() {
            "input_gap"
        } else {
            "end_of_input"
        },
        observed_start_frame: observation.map(|o| o.start_frame),
        observed_end_frame: observation.map(|o| o.end_frame),
        observed_habitat_band_energy: observation.filter(|_| matched).map(|o| o.band_energy),
    })
}

/// Drive the production meter from habitat flux and weak onset reinforcement.
fn drive_production_meter(state: &mut WorkerState, cfg: &WorkerConfig, habitat_chunk: &[f32]) {
    if state.finished {
        return;
    }
    if let Some(observer) = state.temporal_expectation.as_mut() {
        let now = state.timebase.frame_start_tick(state.frame_idx);
        observer.process(now, habitat_chunk, |observation| {
            state
                .participation_predictions
                .retain(|(voice_id, prediction)| {
                    let Some(sample) = participation_outcome(
                        *voice_id,
                        cfg.fs as u32,
                        *prediction,
                        Some(observation),
                    ) else {
                        return true;
                    };
                    report_try(&mut state.reporter, "participation outcome", |writer| {
                        writer.write_participation_outcome(&sample)
                    });
                    false
                });
        });
    }
    state.dorsal.process(habitat_chunk);
    let flux_drive = (state.dorsal.last_metrics().flux.max(0.0) * 500.0)
        .tanh()
        .clamp(0.0, 1.0);
    const ONSET_DRIVE_GAIN: f32 = 0.25;
    let onset_drive = (state
        .pop
        .last_phonation_onset_strength_in_hop()
        .unwrap_or(0.0)
        * ONSET_DRIVE_GAIN)
        .clamp(0.0, 1.0);
    let drive = flux_drive.max(onset_drive);
    let meter_state = state
        .prod_meter
        .process(cfg.hop_duration.as_secs_f32(), drive);
    state.current_landscape.rhythm = NeuralRhythms::from_meter_state(&meter_state);
}

fn log_guard_meter(cfg: &WorkerConfig, last_guard_log: &mut Instant, current_time: f32) {
    if let Some(meter) = cfg.guard_meter.as_ref()
        && last_guard_log.elapsed() >= Duration::from_millis(200)
        && let Some(stats) = meter.take_snapshot()
    {
        warn!(
            "[t={:.6}] Limiter: over={} max_red_db={:.2} in={:.3} out={:.3}",
            current_time,
            stats.num_over,
            stats.max_reduction_db,
            stats.max_abs_in,
            stats.max_abs_out
        );
        *last_guard_log = Instant::now();
    }
}

fn send_runtime_ui_frame(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
    presentation_chunk: Arc<[f32]>,
    peak_level: f32,
    channel_peak: [f32; 2],
    now_tick: Tick,
) {
    let _ = channels
        .ui_tx
        .try_send(build_runtime_ui_frame(RuntimeUiFrameInput {
            body: state.body_snapshot.as_deref(),
            self_sound: state
                .schedule_renderer
                .action_observer
                .as_ref()
                .map(|o| o.snapshot),
            temporal: &state.temporal_frames,
            scenario_name: &cfg.scenario_name,
            conductor: &state.conductor,
            pop: &state.pop,
            generator_model: &state.generator_model,
            current_landscape: &state.current_landscape,
            log_space: &state.log_space,
            presentation_chunk,
            listener_fast_state: state.latest_listener_fast_state,
            listener_state: state.latest_listener_state,
            current_time: state.current_time(),
            playback_state: state.playback_state,
            peak_level,
            channel_peak,
            now_tick,
            hop: cfg.hop,
            fs: cfg.fs,
        }));
    state.last_ui_update = Instant::now();
}

#[derive(Clone, Copy, Debug, Default)]
struct ParamsUpdateEffect {
    harmonicity_changed: bool,
    roughness_changed: bool,
}

fn apply_pending_landscape_update(
    pop: &mut Community,
    params: &mut LandscapeParams,
    current_landscape: &mut LandscapeFrame,
    analysis_update_tx: &Sender<LandscapeUpdate>,
    listener_analysis_update_tx: Option<&Sender<LandscapeUpdate>>,
) -> bool {
    if let Some(update) = pop.take_pending_update() {
        let effect = apply_params_update(params, &update);
        if let Some(mode) = update.pitch_objective_mode {
            current_landscape.pitch_objective_mode = mode;
        }
        if effect.harmonicity_changed {
            let _ = recompute_harmonicity_from_nsgt_power(current_landscape, params);
        }
        if effect.harmonicity_changed || effect.roughness_changed {
            current_landscape.recompute_consonance(params);
        }
        if let Err(err) = analysis_update_tx.send(update) {
            warn!("analysis update disconnected: {err}");
        }
        if let Some(tx) = listener_analysis_update_tx
            && let Err(err) = tx.send(update)
        {
            warn!("listener analysis update disconnected: {err}");
        }
        return effect.harmonicity_changed || effect.roughness_changed;
    }
    false
}

fn recompute_harmonicity_from_nsgt_power(
    current_landscape: &mut LandscapeFrame,
    params: &LandscapeParams,
) -> bool {
    if current_landscape.nsgt_power.len() != current_landscape.space.n_bins()
        || current_landscape.harmonicity.len() != current_landscape.space.n_bins()
    {
        return false;
    }
    let (harmonicity, _) = params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(&current_landscape.nsgt_power, &current_landscape.space);
    if harmonicity.len() != current_landscape.harmonicity.len() {
        return false;
    }
    current_landscape.harmonicity = harmonicity;
    true
}

fn apply_params_update(params: &mut LandscapeParams, upd: &LandscapeUpdate) -> ParamsUpdateEffect {
    let mut effect = ParamsUpdateEffect::default();
    if let Some(k) = upd.roughness_k {
        let roughness_k = if k.is_finite() { k.max(1e-6) } else { 1e-6 };
        let prev = params.roughness_k;
        params.roughness_k = roughness_k;
        effect.roughness_changed = (prev - roughness_k).abs() > f32::EPSILON;
    }
    if let Some(mode) = upd.pitch_objective_mode {
        effect.harmonicity_changed |= false;
        let _ = mode;
    }
    effect
}

fn compose_consonance_field_score_level_with_params(
    h_state01: f32,
    r_state01: f32,
    params: &LandscapeParams,
) -> (f32, f32) {
    let c_score = params.consonance_kernel.score(h_state01, r_state01);
    let c_level = params.consonance_representation.level(c_score);
    (c_score, c_level)
}

#[cfg(test)]
mod perceptual_action_assay;

#[cfg(test)]
mod tests {
    #[test]
    fn participation_outcome_waits_for_actual_audio_and_keeps_gaps_unknown() {
        let prediction = super::ParticipationPrediction {
            issued_frame: 5,
            onset_frame: 11,
            forecast_observed_frame: 0,
            target_start_frame: 20,
            target_end_frame: 30,
            pred_continuation_habitat_band_energy: [0.4; 3],
        };
        let mut observer =
            crate::core::temporal_expectation::AcousticTemporalExpectation::new(1000).unwrap();
        let mut observations = Vec::new();
        observer.process(0, &[0.0; 30], |o| observations.push(o));
        assert!(super::participation_outcome(1, 1000, prediction, Some(observations[1])).is_none());
        let observed =
            super::participation_outcome(1, 1000, prediction, Some(observations[2])).unwrap();
        assert_eq!(observed.status, "observed");
        assert_eq!(observed.observed_habitat_band_energy, Some([0.0; 3]));
        let mut shifted = observations[2];
        shifted.start_frame += 1;
        shifted.end_frame += 1;
        let gap = super::participation_outcome(1, 1000, prediction, Some(shifted)).unwrap();
        assert_eq!(gap.status, "input_gap");
        assert_eq!(gap.observed_habitat_band_energy, None);
        let unfinished = super::participation_outcome(1, 1000, prediction, None).unwrap();
        assert_eq!(unfinished.status, "end_of_input");
        assert_eq!(unfinished.observed_habitat_band_energy, None);
    }
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
    use crate::core::timebase::Timebase;

    #[test]
    #[ignore = "offline spectral evidence export; set CONCHORDAL_SPECTRAL_ASSAY_PLAN"]
    fn export_listener_spectral_evidence() -> anyhow::Result<()> {
        use std::io::{BufWriter, Write};

        let plan_path = std::env::var_os("CONCHORDAL_SPECTRAL_ASSAY_PLAN")
            .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_SPECTRAL_ASSAY_PLAN is required"))?;
        let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(plan_path)?)?;
        let config_path = plan["config"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("config path is required"))?;
        anyhow::ensure!(Path::new(config_path).is_file(), "config must exist");
        let config = AppConfig::load_or_default(config_path)?;
        let cases = plan["cases"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("cases array is required"))?;
        anyhow::ensure!(!cases.is_empty(), "no cases");
        for case in cases {
            let input = case["input"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("input path is required"))?;
            let output = case["output"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("output path is required"))?;
            let mut reader = hound::WavReader::open(input)?;
            let spec = reader.spec();
            anyhow::ensure!(
                spec.channels == 1
                    && spec.bits_per_sample == 16
                    && spec.sample_format == hound::SampleFormat::Int,
                "expected mono PCM16"
            );
            let audio = reader
                .samples::<i16>()
                .map(|sample| sample.map(|value| value as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?;
            let mut input_config = config.clone();
            input_config.audio.sample_rate = spec.sample_rate;
            input_config.validate()?;
            let core = build_analysis_runtime_core(&input_config, spec.sample_rate);
            let space = core.nsgt.space().clone();
            let mut stream = AnalysisStream::new(core.lparams, core.nsgt);
            let mut spectral_history = crate::core::spectral_history::SpectralHistory::new(
                space.clone(),
                spec.sample_rate as f64,
                stream.window_samples(),
            );
            let mut writer = BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(output)?,
            );
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "type": "analysis_contract", "sample_rate": spec.sample_rate,
                    "hop_samples": core.hop, "window_samples": stream.window_samples(),
                    "input_samples": audio.len(), "trailing_samples": audio.len() % core.hop,
                    "frequency_hz": space.centers_hz,
                    "scope": "Native analysis features, including startup. Only complete hops are processed; the trailing input remains pending, without synthesized silence. full_window describes the FFT container, not per-band latency, confidence, or runtime eligibility. No stream identity or cognitive memory duration."
                }),
            )?;
            writeln!(writer)?;
            let mut end_sample = 0;
            for chunk in audio.chunks_exact(core.hop) {
                end_sample += chunk.len();
                let frame = stream.process(chunk);
                spectral_history.observe(
                    (end_sample - chunk.len()) as u64,
                    end_sample as u64,
                    &frame.nsgt_power,
                );
                let history = (end_sample / core.hop % 10 == 0
                    || end_sample + core.hop > audio.len())
                .then(|| spectral_history.snapshot())
                .flatten();
                space.assert_scan_len_named(&frame.nsgt_power, "nsgt_power");
                space.assert_scan_len_named(&frame.subjective_intensity, "subjective_intensity");
                anyhow::ensure!(
                    frame
                        .nsgt_power
                        .iter()
                        .chain(&frame.subjective_intensity)
                        .all(|value| value.is_finite() && *value >= 0.0),
                    "invalid spectral evidence"
                );
                serde_json::to_writer(
                    &mut writer,
                    &serde_json::json!({
                        "type": "spectral_evidence", "available_sec": end_sample as f64 / spec.sample_rate as f64,
                        "full_window": end_sample >= stream.window_samples(),
                        "nsgt_power_scan": frame.nsgt_power,
                        "subjective_intensity_scan": frame.subjective_intensity,
                        "history_ages_sec": crate::core::temporal_history::HISTORY_AGES_SEC,
                        "history_post_order": history.as_ref().map(|h| h.post_order),
                        "history_observed_through_sample": history.as_ref().map(|h| h.observed_through_sample),
                        "known_rms_by_age_scan": history.as_ref().map(|h| &h.known_rms_by_age_scan),
                        "known_coverage_by_age": history.as_ref().map(|h| h.known_coverage_by_age),
                    }),
                )?;
                writeln!(writer)?;
            }
            writer.flush()?;
            println!("observed {input}");
        }
        Ok(())
    }

    #[test]
    fn analysis_ok_truth_table() {
        assert!(analysis_ok(0, None, MAX_LANDSCAPE_LAG_FRAMES));
        assert!(analysis_ok(1, Some(0), MAX_LANDSCAPE_LAG_FRAMES));
        assert!(analysis_ok(10, Some(9), MAX_LANDSCAPE_LAG_FRAMES));
        assert!(!analysis_ok(10, Some(8), MAX_LANDSCAPE_LAG_FRAMES));
        assert!(!analysis_ok(10, None, MAX_LANDSCAPE_LAG_FRAMES));
    }

    #[test]
    fn analysis_runtime_core_uses_config_values() {
        let mut config = AppConfig::default();
        config.analysis.nfft = 2048;
        config.analysis.hop_size = 256;
        config.analysis.tau_ms = 25.0;
        config.psychoacoustics.loudness_exp = 0.31;
        config.psychoacoustics.roughness_k = 0.2;
        config.psychoacoustics.consonance.field.kernel.a = 0.9;
        config.psychoacoustics.consonance.field.level.beta = 3.0;
        config.psychoacoustics.consonance.density.roughness_gain = 1.5;

        let core = build_analysis_runtime_core(&config, 44_100);

        assert_eq!(core.fs, 44_100.0);
        assert_eq!(core.hop, 256);
        assert!((core.hop_duration.as_secs_f32() - 256.0 / 44_100.0).abs() < 1e-9);
        assert_eq!(core.lparams.tau_ms, 25.0);
        assert_eq!(core.lparams.loudness_exp, 0.31);
        assert_eq!(core.lparams.roughness_k, 0.2);
        assert_eq!(core.lparams.consonance_kernel.a, 0.9);
        assert_eq!(core.lparams.consonance_representation.beta, 3.0);
        assert_eq!(core.lparams.consonance_density_roughness_gain, 1.5);
        assert_eq!(core.landscape.space.n_bins(), core.nsgt.space().n_bins());
    }

    fn build_test_params(space: &Log2Space) -> LandscapeParams {
        LandscapeParams {
            fs: 48_000.0,
            max_hist_cols: 1,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            consonance_kernel: ConsonanceKernel::default(),
            consonance_representation: ConsonanceRepresentationParams::default(),
            consonance_density_roughness_gain: 1.0,
            habituation: crate::core::habituation::HabituationParams::default(),
            loudness_exp: 1.0,
            ref_power: 1.0,
            tau_ms: 1.0,
            roughness_k: 1.0,
            roughness_ref_f0_hz: 1000.0,
            roughness_ref_sep_erb: 0.25,
            roughness_ref_mass_split: 0.5,
            roughness_ref_eps: 1e-12,
        }
    }

    #[test]
    fn scenario_dispatch_after_ten_minutes_uses_audio_frame_time() {
        let fs = 48_000.0;
        let hop = 512;
        let timebase = Timebase { fs, hop };
        let cfg = WorkerConfig {
            private_trace: None,
            onset_comparison: None,
            scenario_name: "clock regression".into(),
            wait_user_exit: false,
            start_flag: Arc::new(AtomicBool::new(true)),
            exiting: Arc::new(AtomicBool::new(false)),
            scaffold: Default::default(),
            meter_shaping: Default::default(),
            guard_meter: None,
            underrun_frames: None,
            audio_counters: None,
            dcc_coupler: DccCoupler::new(Default::default()),
            hop,
            hop_duration: Duration::from_secs_f64(hop as f64 / fs as f64),
            fs,
            deterministic_analysis: true,
            deterministic_footprints: true,
        };
        let scenario = crate::scenario::Scenario {
            temporal_mode: crate::scenario::TemporalMode::Off,
            seed: 1,
            control_update_mode: Default::default(),
            scaffold: Default::default(),
            meter_shaping: Default::default(),
            scene_markers: Vec::new(),
            events: vec![crate::scenario::TimedEvent {
                time: 605.0,
                order: 0,
                actions: vec![Action::Finish],
            }],
            duration_sec: 605.0,
        };
        let space = Log2Space::new(200.0, 4_000.0, 12);
        let mut state = WorkerState::new(
            Community::new(timebase),
            Conductor::from_scenario(scenario),
            Landscape::new(space.clone()),
            build_test_params(&space),
            DorsalStream::new(fs),
            None,
            &cfg,
        );
        for frame in [56_680, 56_718, 56_719] {
            state.frame_idx = frame;
            let now_tick = timebase.frame_start_tick(frame);
            state.generator_model.advance_to(now_tick);
            advance_population(&mut state, &cfg, now_tick, ListenerPressure::default());
            assert_eq!(state.pop.abort_requested, frame == 56_719);
        }
    }

    #[test]
    fn listener_memory_is_independent_of_receive_batching() {
        let space = Log2Space::new(200.0, 4_000.0, 12);
        let mut params = build_test_params(&space);
        params.habituation.enabled = true;
        params.habituation.satiation_sec = 0.5;
        params.habituation.recovery_sec = 1.0;
        let timebase = Timebase {
            fs: params.fs,
            hop: 4_800,
        };
        let mut outcomes = Vec::new();
        for batch_size in [1, 3, 24] {
            let mut listener = ListenerTwin::with_sample_rate(params.fs, Default::default());
            let mut hab = crate::core::habituation::HabituationField::new(
                &params.habituation,
                params.consonance_representation.theta,
                space.n_bins(),
            );
            let (tx, rx) = bounded(batch_size);
            let mut last_frame = None;
            let mut last_state = None;
            for start in (0..24).step_by(batch_size) {
                let end = (start + batch_size).min(24);
                for frame_id in start..end {
                    let mut frame = Landscape::new(space.clone());
                    if frame_id < 18 {
                        frame.subjective_intensity[10] = 1.0;
                        frame.harmonicity[10] = 10.0;
                    }
                    tx.send((frame_id as u64, Some(frame))).unwrap();
                }
                let (update, disconnected) = observe_listener_analysis_results(
                    Some(&rx),
                    &mut listener,
                    &params,
                    &mut hab,
                    timebase,
                    end as u64,
                    0,
                    &mut last_frame,
                );
                assert!(!disconnected);
                let state = update.unwrap().unwrap();
                assert_eq!(state.analysis_frame_id, end as u64 - 1);
                last_state = Some(state);
            }
            let retained = hab.state().to_vec();
            assert!(retained.iter().any(|&value| value > 0.0));
            let (update, _) = observe_listener_analysis_results(
                Some(&rx),
                &mut listener,
                &params,
                &mut hab,
                timebase,
                25,
                0,
                &mut last_frame,
            );
            assert!(update.is_none());
            assert_eq!(hab.state(), retained);
            let state = last_state.unwrap();
            outcomes.push((retained, state.stability_level, state.tension_level));
        }
        assert_eq!(outcomes[0], outcomes[1]);
        assert_eq!(outcomes[0], outcomes[2]);
    }

    #[test]
    fn listener_analysis_gap_clears_state_instead_of_reusing_prior_pressure() {
        let space = Log2Space::new(200.0, 4_000.0, 12);
        let params = build_test_params(&space);
        let mut listener = ListenerTwin::with_sample_rate(params.fs, Default::default());
        let mut hab = crate::core::habituation::HabituationField::new(
            &params.habituation,
            params.consonance_representation.theta,
            space.n_bins(),
        );
        let timebase = Timebase {
            fs: params.fs,
            hop: 128,
        };
        let mut last_frame = Some(0);
        let (tx, rx) = bounded(2);
        let coupler = DccCoupler::new(crate::config::DccConfig {
            coupling_strength: 0.25,
            max_temperature_bonus: 0.1,
        });
        let mut evidence = Landscape::new(space.clone());
        evidence.subjective_intensity[2] = 1.0;
        evidence.harmonicity[3] = 10.0;
        tx.send((1, Some(evidence.clone()))).unwrap();
        let (update, _) = observe_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            timebase,
            2,
            0,
            &mut last_frame,
        );
        let mut current = update.expect("initial evidence update");
        let initial_pressure = coupler.pressure(current);
        assert!(initial_pressure.temperature_bonus > 0.0);
        tx.send((3, None)).unwrap();
        let (update, disconnected) = observe_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            timebase,
            4,
            0,
            &mut last_frame,
        );
        assert!(
            matches!(update, Some(None)),
            "a gap must invalidate the previous observation"
        );
        assert!(!disconnected);
        assert!(last_frame.is_none());
        current = update.unwrap();
        assert_eq!(coupler.pressure(current), ListenerPressure::default());
        let (update, _) = observe_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            timebase,
            5,
            0,
            &mut last_frame,
        );
        assert!(
            update.is_none(),
            "an empty queue is distinct from invalidation"
        );
        assert_eq!(coupler.pressure(current), ListenerPressure::default());
        tx.send((4, Some(Landscape::new(space)))).unwrap();
        let (update, _) = observe_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            timebase,
            6,
            5,
            &mut last_frame,
        );
        assert!(
            update.is_none(),
            "late pre-gap results must not restore pressure"
        );
        assert!(last_frame.is_none());
        assert_eq!(coupler.pressure(current), ListenerPressure::default());
        tx.send((5, Some(evidence))).unwrap();
        let (update, _) = observe_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            timebase,
            6,
            5,
            &mut last_frame,
        );
        current = update.expect("post-gap evidence update");
        assert_eq!(last_frame, Some(5));
        assert_eq!(coupler.pressure(current), initial_pressure);
    }

    #[test]
    #[ignore = "27-case research campaign; writes reports under target/habituation-isolation"]
    fn habituation_isolation_campaign() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let output = Path::new("target/habituation-isolation").join(stamp.to_string());
        std::fs::create_dir_all(&output).unwrap();
        std::fs::copy(file!(), output.join("runtime.rs")).unwrap();
        std::fs::copy(
            "samples/research/habituation_field_assay.rhai",
            output.join("scenario.rhai"),
        )
        .unwrap();
        for (control, adaptation_off, exploration_off) in [
            ("baseline", false, false),
            ("adaptation_off", true, false),
            ("joint_off", true, true),
        ] {
            for seed in [1, 21, 42] {
                for (condition, enabled, recovery_sec) in [
                    ("off", false, 8.0),
                    ("normal", true, 8.0),
                    ("slow", true, 80.0),
                ] {
                    let mut config = AppConfig::default();
                    config.psychoacoustics.habituation.enabled = enabled;
                    config.psychoacoustics.habituation.recovery_sec = recovery_sec;
                    let script = "samples/research/habituation_field_assay.rhai";
                    let mut scenario = compile_scenario_from_script(
                        Path::new(script),
                        &render_compile_args(script, Some(seed)),
                        &config,
                    )
                    .unwrap();
                    scenario.seed = seed;
                    let mut mover_count = 0;
                    for action in scenario.events.iter_mut().flat_map(|e| &mut e.actions) {
                        if let Action::Spawn {
                            population_id: 3,
                            ids,
                            spec,
                            ..
                        } = action
                        {
                            mover_count += ids.len();
                            assert!(spec.control.adaptation.enabled);
                            if adaptation_off {
                                spec.control.adaptation.enabled = false;
                            }
                            if exploration_off {
                                spec.control.pitch.temperature = Some(0.0);
                                spec.control.pitch.crowding_strength = 0.0;
                            }
                        }
                    }
                    assert_eq!(
                        mover_count, 5,
                        "review the assay if its mover population changes"
                    );
                    let case = output.join(format!("{control}-{condition}-seed-{seed}"));
                    std::fs::create_dir(&case).unwrap();
                    std::fs::write(case.join("scenario-ir.txt"), format!("{scenario:#?}")).unwrap();
                    std::fs::write(
                        case.join("config.toml"),
                        toml::to_string_pretty(&config).unwrap(),
                    )
                    .unwrap();
                    let report = case.join("report.jsonl");
                    let mut reporter = JsonlReporter::create(report.to_str().unwrap()).unwrap();
                    reporter.write_meta(seed).unwrap();
                    reporter
                        .write_scene_markers(&scenario.scene_markers)
                        .unwrap();
                    let wiring = wire_runtime(
                        &config,
                        config.audio.sample_rate,
                        "habituation isolation".into(),
                        scenario,
                        Arc::new(AtomicBool::new(false)),
                        WiringOptions {
                            #[cfg(test)]
                            offline_body_probe: None,
                            ui_channel_capacity: 1,
                            listener_forced: true,
                            wait_user_exit: false,
                            start_playing: true,
                            audio_prod: None,
                            wav_tx: None,
                            reporter: Some(reporter),
                            deterministic_analysis: true,
                            deterministic_footprints: true,
                            guard_meter: None,
                            underrun_frames: None,
                            reserve_runtime_ids_through: 0,
                            profile: None,
                            audio_counters: None,
                        },
                    )
                    .unwrap();
                    join_thread("worker", wiring.worker_handle).unwrap();
                    join_thread("analysis", wiring.analysis_handle).unwrap();
                    join_thread("listener", wiring.listener_analysis_handle.unwrap()).unwrap();
                    assert!(wiring.report_error_rx.unwrap().try_recv().is_err());
                    println!("isolation report: {}", report.display());
                }
            }
        }
    }

    #[test]
    #[ignore = "fixed-input DCC research campaign; writes reports and offline WAVs under target/dcc-isolation"]
    fn dcc_fixed_input_campaign() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let output = Path::new("target/dcc-isolation").join(stamp.to_string());
        std::fs::create_dir_all(&output).unwrap();
        std::fs::copy(file!(), output.join("runtime.rs")).unwrap();
        std::fs::copy(
            "tests/scripts/dcc_pitch_feedback.rhai",
            output.join("scenario.rhai"),
        )
        .unwrap();
        let seeds = std::env::var("CONCHORDAL_RESEARCH_SEEDS").unwrap_or_else(|_| "1,21,42".into());
        for seed in seeds.split(',').map(|value| value.parse::<u64>().unwrap()) {
            let mut baseline_audio = None;
            let mut baseline_listener = None;
            for strength in [0.0, 0.1, 0.25, 0.5, 1.0] {
                let mut config = AppConfig::default();
                config.dcc.coupling_strength = strength;
                let script = "tests/scripts/dcc_pitch_feedback.rhai";
                let mut scenario = compile_scenario_from_script(
                    Path::new(script),
                    &render_compile_args(script, Some(seed)),
                    &config,
                )
                .unwrap();
                scenario.seed = seed;
                for action in scenario.events.iter_mut().flat_map(|e| &mut e.actions) {
                    if let Action::Spawn {
                        population_id,
                        spec,
                        ..
                    } = action
                        && *population_id <= 4
                    {
                        // Drone sway reads the shared rhythm; SeqGate makes the
                        // presentation reference independent of habitat feedback.
                        spec.articulation = crate::scenario::ArticulationCoreConfig::Seq {
                            duration: 20.0,
                            breath_gain_init: None,
                        };
                    }
                }
                let name = format!("strength-{strength}-seed-{seed}");
                let report = output.join(format!("{name}.jsonl"));
                let wav = output.join(format!("{name}.wav"));
                let mut reporter = JsonlReporter::create(report.to_str().unwrap()).unwrap();
                reporter.write_meta(seed).unwrap();
                reporter
                    .write_scene_markers(&scenario.scene_markers)
                    .unwrap();
                let (wav_tx, wav_rx) = bounded(16);
                let writer = crate::audio::writer::WavOutput::run(
                    wav_rx,
                    wav.to_str().unwrap().to_string(),
                    config.audio.sample_rate,
                    resolve_limiter_mode(&config),
                    None,
                );
                let wiring = wire_runtime(
                    &config,
                    config.audio.sample_rate,
                    "DCC fixed input".into(),
                    scenario,
                    Arc::new(AtomicBool::new(false)),
                    WiringOptions {
                        #[cfg(test)]
                        offline_body_probe: None,
                        ui_channel_capacity: 1,
                        listener_forced: true,
                        wait_user_exit: false,
                        start_playing: true,
                        audio_prod: None,
                        wav_tx: Some(wav_tx),
                        reporter: Some(reporter),
                        deterministic_analysis: true,
                        deterministic_footprints: true,
                        guard_meter: None,
                        underrun_frames: None,
                        reserve_runtime_ids_through: 0,
                        profile: None,
                        audio_counters: None,
                    },
                )
                .unwrap();
                join_thread("worker", wiring.worker_handle).unwrap();
                join_thread("writer", writer).unwrap();
                join_thread("analysis", wiring.analysis_handle).unwrap();
                join_thread("listener", wiring.listener_analysis_handle.unwrap()).unwrap();
                assert!(wiring.report_error_rx.unwrap().try_recv().is_err());
                let audio = std::fs::read(&wav).unwrap();
                let listener: Vec<serde_json::Value> = std::fs::read_to_string(&report)
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                    .filter(|row| row["type"] == "listener_state")
                    .collect();
                assert!(listener.len() > 100);
                if let Some(baseline) = baseline_audio.as_ref() {
                    assert_eq!(&audio, baseline, "presentation audio changed at {name}");
                    assert_eq!(Some(&listener), baseline_listener.as_ref());
                } else {
                    baseline_audio = Some(audio);
                    baseline_listener = Some(listener);
                }
                println!("fixed input report: {}", report.display());
            }
        }
    }

    #[test]
    fn pending_update_does_not_panic_when_analysis_channel_is_disconnected() {
        let space = Log2Space::new(80.0, 2_000.0, 96);
        let mut params = build_test_params(&space);
        let mut landscape = Landscape::new(space);
        let mut pop = Community::new(Timebase {
            fs: 48_000.0,
            hop: 256,
        });

        pop.apply_action(
            Action::SetHarmonicityParams {
                update: LandscapeUpdate {
                    roughness_k: Some(2.0),
                    pitch_objective_mode: None,
                },
            },
            &landscape,
            None::<&mut crate::core::stream::analysis::AnalysisStream>,
        );

        let (analysis_update_tx, analysis_update_rx) = bounded::<LandscapeUpdate>(1);
        drop(analysis_update_rx);
        let changed = apply_pending_landscape_update(
            &mut pop,
            &mut params,
            &mut landscape,
            &analysis_update_tx,
            None,
        );
        assert!(changed);
    }
}

#[cfg(test)]
type OfflineBodyProbe = Box<dyn FnMut(&WorkerState, Tick, usize, [&[f32]; 2]) + Send>;

#[cfg(test)]
mod body_profiles;
