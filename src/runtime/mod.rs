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
use crate::core::timebase::Tick;
use crate::dcc_coupler::{DccCoupler, ListenerPressure};
use crate::life::community::Community;
use crate::life::conductor::Conductor;
use crate::life::report::{
    HopTimingSample, JsonlReporter, ListenerStateSample, RhythmObservation,
    onset_samples_from_batches, scaffold_phase_0_1, summarize_populations,
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
        wave: WaveFrame {
            fs,
            samples: Arc::from(Vec::<f32>::new()),
        },
        spec: SpecFrame {
            spec_hz: current_landscape.space.centers_hz.clone(),
            amps: vec![0.0; current_landscape.space.n_bins()],
        },
        listener: ListenerFrame::default(),
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
fn merge_latest_listener_analysis_results(
    listener_result_rx: Option<&Receiver<analysis_worker::AnalysisResult>>,
    listener_twin: &mut ListenerTwin,
    lparams: &LandscapeParams,
    hab_listener: &mut crate::core::habituation::HabituationField,
    dt_sec: f32,
    timebase: crate::core::timebase::Timebase,
    generated_frame_id: u64,
    min_valid_frame: u64,
    last_listener_analysis_frame: &mut Option<u64>,
) -> (Option<Option<ListenerState>>, bool) {
    let Some(rx) = listener_result_rx else {
        return (None, false);
    };
    let mut latest_audio: Option<analysis_worker::AnalysisResult> = None;
    let mut disconnected = false;
    loop {
        match rx.try_recv() {
            Ok((analyzed_id, Some(_))) if analyzed_id < min_valid_frame => continue,
            Ok((analyzed_id, frame)) => {
                *last_listener_analysis_frame = frame.as_ref().map(|_| analyzed_id);
                latest_audio = Some((analyzed_id, frame));
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
    let Some((analysis_frame_id, frame)) = latest_audio else {
        return (None, disconnected);
    };
    let Some(mut frame) = frame else {
        return (Some(None), disconnected);
    };

    frame.recompute_consonance(lparams);
    // Listener habituation advances per processed presentation frame; under the
    // deterministic-render catch-up loop this can advance >1x per hop. Accepted:
    // with the default dcc coupling_strength = 0.0 the listener field feeds only
    // ListenerTwin telemetry, not pre-synth ALIFE. The ecology field advances
    // exactly once per hop (deterministic).
    drive_and_apply_habituation(&mut frame, hab_listener, lparams, dt_sec);
    let analysis_time_sec = timebase.tick_to_sec(timebase.frame_end_tick(analysis_frame_id));
    let state = listener_twin.observe_presentation_landscape(
        analysis_time_sec,
        generated_frame_id,
        analysis_frame_id,
        &frame,
    );
    (Some(Some(state)), disconnected)
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
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            analysis_worker::run(
                analysis_stream,
                audio_to_analysis_rx,
                analysis_result_tx,
                analysis_update_rx,
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
    ui_channel_capacity: usize,
    /// Enables listener analysis even without DCC coupling (GUI display / report).
    listener_forced: bool,
    wait_user_exit: bool,
    start_playing: bool,
    audio_prod: Option<ringbuf::HeapProd<f32>>,
    wav_tx: Option<Sender<Arc<[f32]>>>,
    reporter: Option<JsonlReporter>,
    deterministic_analysis: bool,
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
) -> RuntimeWiring {
    let WiringOptions {
        ui_channel_capacity,
        listener_forced,
        wait_user_exit,
        start_playing,
        audio_prod,
        wav_tx,
        reporter,
        deterministic_analysis,
        guard_meter,
        underrun_frames,
        reserve_runtime_ids_through,
        profile,
        audio_counters,
    } = opts;

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
    let analysis_stream = AnalysisStream::new(core.lparams.clone(), core.nsgt.clone());
    let analysis_handle = spawn_analysis_worker(
        "field-analysis",
        analysis_stream,
        audio_to_analysis_rx,
        analysis_result_tx,
        analysis_update_rx,
    );

    let dcc_coupler = DccCoupler::new(config.dcc);
    let listener_analysis_enabled = listener_forced || dcc_coupler.coupling_strength() > 0.0;
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
    };
    let channels = WorkerChannels {
        ui_tx: ui_frame_tx,
        report_error_tx,
        audio_prod,
        wav_tx,
        audio_to_analysis_tx,
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
            state.profile = profile;
            worker_loop(cfg, channels, state)
        })
        .expect("spawn worker");

    RuntimeWiring {
        ui_frame_rx,
        report_error_rx,
        start_flag,
        worker_handle,
        analysis_handle,
        listener_analysis_handle,
    }
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
    let deterministic_analysis = reporter.is_some() && !args.play;
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
            !args.nogui || args.report.is_some() || coupling > 0.0,
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
            ui_channel_capacity,
            listener_forced: !args.nogui || args.report.is_some(),
            wait_user_exit: config.playback.wait_user_exit,
            start_playing: !config.playback.wait_user_start,
            audio_prod,
            wav_tx: None,
            reporter,
            deterministic_analysis,
            guard_meter,
            underrun_frames: audio_out.as_ref().map(AudioOutput::underrun_frames),
            reserve_runtime_ids_through: 0,
            profile,
            audio_counters: audio_out.as_ref().map(AudioOutput::counters),
        },
    );

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
            ui_channel_capacity: 1,
            listener_forced: reporter.is_some(),
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: Some(wav_tx),
            reporter,
            deterministic_analysis: true,
            guard_meter,
            underrun_frames: None,
            reserve_runtime_ids_through,
            profile: None,
            audio_counters: None,
        },
    );

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
}

/// Channel endpoints and the audio ring-buffer producer owned by the worker.
struct WorkerChannels {
    ui_tx: Sender<UiFrame>,
    report_error_tx: Option<Sender<String>>,
    audio_prod: Option<ringbuf::HeapProd<f32>>,
    wav_tx: Option<Sender<Arc<[f32]>>>,
    audio_to_analysis_tx: Sender<(u64, Arc<[f32]>)>,
    analysis_result_rx: Receiver<analysis_worker::AnalysisResult>,
    analysis_update_tx: Sender<LandscapeUpdate>,
    presentation_to_listener_tx: Option<Sender<(u64, Arc<[f32]>)>>,
    listener_result_rx: Option<Receiver<analysis_worker::AnalysisResult>>,
    listener_analysis_update_tx: Option<Sender<LandscapeUpdate>>,
}

/// Mutable worker state. Built once before the loop; hop phases take it `&mut`.
struct WorkerState {
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
    current_time: f32,
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
    generator_model: crate::life::generator_model::GeneratorModel,
    hab_ecology: crate::core::habituation::HabituationField,
    hab_listener: crate::core::habituation::HabituationField,
    schedule_renderer: ScheduleRenderer,
    scenario_end_tick: Option<Tick>,
    phonation_batches_buf: Vec<PhonationBatch>,
}

impl WorkerState {
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
        let now = Instant::now();
        Self {
            pop,
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
            current_time: 0.0,
            frame_idx: 0,
            monitor: AudioMonitor::new(),
            last_guard_log: now - Duration::from_millis(200),
            last_ui_update: now,
            last_tick_log: now,
            last_analysis_frame: None,
            last_listener_analysis_frame: None,
            listener_min_valid_frame: 0,
            listener_twin: ListenerTwin::with_sample_rate(
                cfg.fs,
                crate::listener_twin::ListenerTwinConfig::default(),
            ),
            latest_listener_fast_state: None,
            latest_listener_state: None,
            prod_meter,
            generator_model,
            hab_ecology,
            hab_listener,
            schedule_renderer: ScheduleRenderer::new(timebase),
            scenario_end_tick: None,
            phonation_batches_buf: Vec::new(),
        }
    }
}

fn worker_loop(cfg: WorkerConfig, mut channels: WorkerChannels, mut state: WorkerState) {
    if cfg.start_flag.load(Ordering::SeqCst) {
        state.playback_state = PlaybackState::Playing;
        if let Some(count) = cfg.underrun_frames.as_ref() {
            count.store(0, Ordering::Relaxed);
        }
    }
    let idle_silence = vec![0.0f32; cfg.hop];

    let _ = channels.ui_tx.try_send(build_initial_ui_frame(
        &cfg.scenario_name,
        &state.conductor,
        &state.pop,
        &state.current_landscape,
        state.current_time,
        state.playback_state,
        cfg.fs,
    ));

    loop {
        if cfg
            .audio_counters
            .as_ref()
            .is_some_and(|counters| counters.callback_errors_total.load(Ordering::Relaxed) > 0)
        {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
        if cfg.exiting.load(Ordering::SeqCst) {
            eprintln!("Stopping worker thread.");
            break;
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

        if cfg.exiting.load(Ordering::SeqCst) || (state.finished && !cfg.wait_user_exit) {
            break;
        }

        if !produced_any {
            thread::sleep(Duration::from_millis(1));
        }
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
    apply_landscape_updates(state, channels, cfg, analysis_updated, now_tick, now_sec);

    let listener_wait_start = Instant::now();
    let listener_state_update = wait_for_listener(state, channels, cfg);
    let listener_wait = listener_wait_start.elapsed();
    if let Some(listener_state) = listener_state_update {
        state.latest_listener_state = listener_state;
    }
    let listener_pressure = cfg.dcc_coupler.pressure(state.latest_listener_state);
    let listener_pressure_update =
        listener_state_update.map(|listener_state| cfg.dcc_coupler.pressure(listener_state));

    let phonation_count = advance_population(state, cfg, now_tick, listener_pressure);
    emit_hop_reports(
        state,
        cfg,
        now_sec,
        phonation_count,
        listener_state_update.flatten(),
        listener_pressure_update,
    );

    let (presentation_chunk, habitat_chunk, max_abs) =
        render_and_route_audio(state, channels, cfg, now_tick, phonation_count);
    let channel_peak = [max_abs, max_abs];

    drive_production_meter(state, cfg, habitat_chunk.as_ref());
    state.latest_listener_fast_state = Some(state.listener_twin.observe_presentation_audio(
        now_sec,
        state.frame_idx,
        presentation_chunk.as_ref(),
    ));

    // Feed every habitat hop to NSGT-RT. Dropping hops breaks time continuity.
    if let Err(err) = channels
        .audio_to_analysis_tx
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
        info!("[t={:.6}] WAV closed.", state.current_time);
    }

    let must_send_ui =
        state.conductor.is_done() || state.pop.abort_requested || state.scenario_end_tick.is_some();
    let should_send_ui = must_send_ui || state.last_ui_update.elapsed() >= UI_MIN_INTERVAL;

    let peak_level = state.monitor.max_peak.max(max_abs);

    log_guard_meter(cfg, &mut state.last_guard_log, state.current_time);

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
            info!("[t={:.6}] Scenario finished. {note}", state.current_time);
            state.finish_logged = true;
        }
        if !cfg.wait_user_exit {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
    }

    state.monitor.update(
        state.current_time,
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
        state.current_time += cfg.hop_duration.as_secs_f32();
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
        let elapsed_us = t_start.elapsed().as_secs_f64() * 1_000_000.0;
        let worker_allocations = finish_allocations();
        profile.record(HopProfile {
            frame_idx,
            time_sec: now_sec,
            alive_voice_count,
            elapsed_us,
            analysis_wait_us: analysis_wait.as_secs_f64() * 1_000_000.0,
            listener_wait_us: listener_wait.as_secs_f64() * 1_000_000.0,
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
/// best-effort drain to avoid stalling the generator on listener-analysis
/// latency.
fn wait_for_listener(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
) -> Option<Option<ListenerState>> {
    let mut listener_state_update: Option<Option<ListenerState>> = None;
    loop {
        let (listener_state, listener_disconnected) = merge_latest_listener_analysis_results(
            channels.listener_result_rx.as_ref(),
            &mut state.listener_twin,
            &state.lparams,
            &mut state.hab_listener,
            cfg.hop_duration.as_secs_f32(),
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
        state.current_time,
        state.frame_idx,
        &state.current_landscape,
        None::<&mut crate::core::stream::analysis::AnalysisStream>,
        &mut state.pop,
    );
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
    let onset_samples = onset_samples_from_batches(
        &state.pop.voices,
        &state.phonation_batches_buf[..phonation_count],
        now_sec,
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
/// Returns `(presentation_chunk, habitat_chunk, max_abs)`.
fn render_and_route_audio(
    state: &mut WorkerState,
    channels: &mut WorkerChannels,
    cfg: &WorkerConfig,
    now_tick: Tick,
    phonation_count: usize,
) -> (Arc<[f32]>, Arc<[f32]>, f32) {
    let frame = state.schedule_renderer.render(
        &state.phonation_batches_buf[..phonation_count],
        now_tick,
        &state.current_landscape.rhythm,
    );

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
                    state.last_listener_analysis_frame = None;
                    state.listener_min_valid_frame = state.frame_idx.saturating_add(1);
                    debug!("listener analysis backlog full; dropped presentation hop");
                }
                Err(TrySendError::Disconnected(_)) => {
                    warn!("listener analysis worker disconnected");
                    channels.presentation_to_listener_tx = None;
                    state.latest_listener_state = None;
                }
            }
        }
    }

    (presentation_chunk, habitat_chunk, max_p)
}

/// Drive the production meter from both the habitat-bus flux and the
/// population's own onsets (low-latency auditory-motor reinforcement).
/// Combined via max() and kept weak to avoid closed-loop wobble; flux
/// alone is the fallback when no onset fires.
fn drive_production_meter(state: &mut WorkerState, cfg: &WorkerConfig, habitat_chunk: &[f32]) {
    if state.finished {
        return;
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
            scenario_name: &cfg.scenario_name,
            conductor: &state.conductor,
            pop: &state.pop,
            generator_model: &state.generator_model,
            current_landscape: &state.current_landscape,
            log_space: &state.log_space,
            presentation_chunk,
            listener_fast_state: state.latest_listener_fast_state,
            listener_state: state.latest_listener_state,
            current_time: state.current_time,
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
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
    use crate::core::timebase::Timebase;

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
        tx.send((1, Some(Landscape::new(space.clone())))).unwrap();
        tx.send((3, None)).unwrap();
        let (update, disconnected) = merge_latest_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            128.0 / params.fs,
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
        let (update, _) = merge_latest_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            128.0 / params.fs,
            timebase,
            5,
            0,
            &mut last_frame,
        );
        assert!(
            update.is_none(),
            "an empty queue is distinct from invalidation"
        );
        tx.send((4, Some(Landscape::new(space)))).unwrap();
        let (update, _) = merge_latest_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            128.0 / params.fs,
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
