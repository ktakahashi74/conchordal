use std::collections::VecDeque;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};
use tracing::*;

use crossbeam_channel::{Receiver, Sender, bounded};
use ringbuf::traits::Observer;

use crate::audio::limiter::{Limiter, LimiterMeter, LimiterMode};
#[cfg(debug_assertions)]
use crate::audio::writer::WavOutput;
use crate::core::analysis_worker;
use crate::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeParams, LandscapeUpdate};
use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
use crate::core::nsgt_rt::{RtConfig, RtNsgtKernelLog2};
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::core::stream::{analysis::AnalysisStream, dorsal::DorsalStream};
use crate::core::timebase::Tick;
use crate::life::conductor::Conductor;
use crate::life::individual::{PhonationBatch, SoundBody};
use crate::life::population::Population;
use crate::life::scenario::{Action, Scenario};
use crate::life::schedule_renderer::ScheduleRenderer;
use crate::life::scripting::ScriptHost;
use crate::life::sound::{AudioCommand, VoiceTarget};
use crate::ui::viewdata::{
    AgentStateInfo, DorsalFrame, PlaybackState, SimulationMeta, SpecFrame, UiFrame, WaveFrame,
};
use crate::{
    audio::output::AudioOutput, config::AppConfig, core::harmonicity_kernel::HarmonicityParams,
};

const MAX_LANDSCAPE_LAG_FRAMES: u64 = 1;

struct AudioMonitor {
    min_occupancy: Option<usize>,
    max_peak: f32,
    slow_chunks: u32,
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
        analysis_lag: Option<u64>,
        conductor_done: bool,
    ) -> f32 {
        self.min_occupancy = Some(
            self.min_occupancy
                .map_or(buffer_occupancy, |m| m.min(buffer_occupancy)),
        );
        self.max_peak = self.max_peak.max(chunk_peak);

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
                "[t={:.6}] Audio chunk compute slow: {:?} (hop {:?}) frame_idx={}",
                current_time, chunk_elapsed, hop_duration, frame_idx
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

        let peak_level = self.max_peak;

        if self.last_stats_log.elapsed() > Duration::from_secs(1) {
            if let Some(min_occ) = self.min_occupancy.take() {
                debug!(
                    "[t={:.6}] Audio stats: min_occ={}, cap={}, hop={}, max_peak={:.3}, slow_chunks={}",
                    current_time, min_occ, buffer_capacity, hop, peak_level, self.slow_chunks
                );
            }
            self.max_peak = 0.0;
            self.slow_chunks = 0;
            self.last_stats_log = Instant::now();
        }

        peak_level
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

pub struct App {
    ui_frame_rx: Receiver<UiFrame>,
    _ctrl_tx: Sender<()>, // placeholder
    last_frame: UiFrame,
    ui_queue: VecDeque<UiFrame>,
    visual_delay_frames: usize,
    _audio: Option<AudioOutput>,
    audio_init_error: Option<String>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    wav_handle: Option<std::thread::JoinHandle<()>>,
    exiting: Arc<AtomicBool>,
    rhythm_history: VecDeque<(f64, crate::core::modulation::NeuralRhythms)>,
    dorsal_history: VecDeque<(f64, DorsalFrame)>,
    start_flag: Arc<AtomicBool>,
    level_history: VecDeque<(std::time::Instant, [f32; 2])>,
    show_raw_nsgt_power: bool,
    has_ui_frame: bool,
}

struct RuntimeInit {
    ui_frame_rx: Receiver<UiFrame>,
    ctrl_tx: Sender<()>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    wav_handle: Option<std::thread::JoinHandle<()>>,
    start_flag: Arc<AtomicBool>,
    audio_out: Option<AudioOutput>,
    audio_init_error: Option<String>,
    visual_delay_frames: usize,
}

pub fn compile_scenario_from_script(
    script_path: &Path,
    _args: &crate::cli::Args,
    _config: &AppConfig,
) -> Result<Scenario, String> {
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
    let path_str = script_path.to_string_lossy();
    ScriptHost::load_script(&path_str).map_err(|e| {
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

pub fn validate_scenario(scenario: &Scenario) -> Result<(), String> {
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
                Action::Spawn { .. } | Action::Update { .. } | Action::Release { .. } => {}
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
    markers.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.order.cmp(&b.order))
    });
    for marker in &markers {
        eprintln!(
            "scene t={:.3} order={} name={}",
            marker.time, marker.order, marker.name
        );
    }
    let mut events = scenario.events.clone();
    events.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.order.cmp(&b.order))
    });
    for event in &events {
        let action_descs: Vec<String> = event.actions.iter().map(ToString::to_string).collect();
        eprintln!(
            "event t={:.3} order={} {}",
            event.time,
            event.order,
            action_descs.join(" | ")
        );
    }
    let mut event_count = 0usize;
    let mut action_count = 0usize;
    let marker_count = scenario.scene_markers.len();
    event_count += scenario.events.len();
    for event in &scenario.events {
        action_count += event.actions.len();
    }
    eprintln!(
        "OK compile-only: {} (events={}, actions={}, markers={})",
        path.display(),
        event_count,
        action_count,
        marker_count
    );
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        args: crate::cli::Args,
        config: AppConfig,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        let repaint_ctx = cc.egui_ctx.clone();
        let stop_flag_watch = stop_flag.clone();
        std::thread::spawn(move || {
            while !stop_flag_watch.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
            }
            repaint_ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            repaint_ctx.request_repaint();
            std::thread::sleep(Duration::from_millis(200));
            if stop_flag_watch.load(Ordering::SeqCst) {
                // Force exit in case the event loop is asleep (e.g., window not exposed).
                std::process::exit(0);
            }
        });

        let rt = init_runtime(args, config, stop_flag.clone());

        //cc.egui_ctx.set_pixels_per_point(1.0);
        cc.egui_ctx
            .send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::Vec2 {
                x: 1200.0,
                y: 900.0,
            }));
        cc.egui_ctx
            .send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::Vec2 {
                x: 1200.0,
                y: 900.0,
            }));

        Self {
            ui_frame_rx: rt.ui_frame_rx,
            _ctrl_tx: rt.ctrl_tx,
            last_frame: UiFrame::default(),
            ui_queue: VecDeque::new(),
            visual_delay_frames: rt.visual_delay_frames,
            _audio: rt.audio_out,
            audio_init_error: rt.audio_init_error,
            wav_handle: rt.wav_handle,
            worker_handle: rt.worker_handle,
            exiting: stop_flag,
            rhythm_history: VecDeque::with_capacity(4096),
            dorsal_history: VecDeque::with_capacity(4096),
            start_flag: rt.start_flag,
            level_history: VecDeque::with_capacity(256),
            show_raw_nsgt_power: false,
            has_ui_frame: false,
        }
    }

    fn apply_ui_frame(&mut self, frame: UiFrame) {
        self.last_frame = frame;

        let t = self.last_frame.time_sec as f64;
        if self.last_frame.meta.playback_state != PlaybackState::Finished {
            self.rhythm_history
                .push_back((t, self.last_frame.landscape.rhythm));
            self.dorsal_history.push_back((t, self.last_frame.dorsal));
        }

        let t_last = self
            .rhythm_history
            .back()
            .map(|(t, _)| *t)
            .or_else(|| self.dorsal_history.back().map(|(t, _)| *t));
        if let Some(t_last) = t_last {
            while self
                .rhythm_history
                .front()
                .is_some_and(|(time, _)| *time < t_last - 5.0)
            {
                self.rhythm_history.pop_front();
            }
            while self
                .dorsal_history
                .front()
                .is_some_and(|(time, _)| *time < t_last - 5.0)
            {
                self.dorsal_history.pop_front();
            }
        }

        // Track 1-second peak history for level meter.
        let now = std::time::Instant::now();
        self.level_history
            .push_back((now, self.last_frame.meta.channel_peak));
        while let Some((t, _)) = self.level_history.front() {
            if now.duration_since(*t).as_secs_f32() > 1.0 {
                self.level_history.pop_front();
            } else {
                break;
            }
        }
        let mut window_peak = [0.0f32; 2];
        for (_, peaks) in &self.level_history {
            window_peak[0] = window_peak[0].max(peaks[0]);
            window_peak[1] = window_peak[1].max(peaks[1]);
        }
        self.last_frame.meta.window_peak = window_peak;
    }
}

fn init_runtime(
    args: crate::cli::Args,
    config: AppConfig,
    stop_flag: Arc<AtomicBool>,
) -> RuntimeInit {
    let latency_ms = config.audio.latency_ms;
    #[cfg(debug_assertions)]
    let wav_enabled = args.wav.is_some();
    #[cfg(not(debug_assertions))]
    let wav_enabled = false;
    let guard_mode = match config.audio.limiter {
        crate::config::LimiterSetting::None => LimiterMode::None,
        crate::config::LimiterSetting::SoftClip => {
            LimiterMode::SoftClip(crate::audio::limiter::SoftClipParams::default())
        }
        crate::config::LimiterSetting::PeakLimiter => {
            LimiterMode::PeakLimiter(crate::audio::limiter::PeakLimiterParams::default())
        }
    };
    let guard_mode = Limiter::from_env_or(guard_mode);
    let guard_meter = if args.play || wav_enabled {
        Some(Arc::new(LimiterMeter::default()))
    } else {
        None
    };

    // Audio
    let (audio_out, audio_prod, audio_init_error) = if args.play {
        match AudioOutput::new(latency_ms, guard_mode, guard_meter.clone()) {
            Ok((out, prod)) => (Some(out), Some(prod), None),
            Err(e) => {
                let msg = e.to_string();
                eprintln!("Audio init failed: {msg}");
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

    // WAV (debug-only)
    #[cfg(debug_assertions)]
    let (wav_tx, wav_handle) = if let Some(path) = args.wav.clone() {
        let (wav_tx, wav_rx) = bounded::<Arc<[f32]>>(16);
        let wav_handle = WavOutput::run(
            wav_rx,
            path,
            runtime_sample_rate,
            guard_mode,
            guard_meter.clone(),
        );
        (Some(wav_tx), Some(wav_handle))
    } else {
        (None, None)
    };
    #[cfg(not(debug_assertions))]
    let (wav_tx, wav_handle) = (None, None);

    // Analysis/NSGT setup
    let fs: f32 = runtime_sample_rate as f32;
    let space = Log2Space::new(55.0, 8000.0, 96);
    let lparams = LandscapeParams {
        fs,
        max_hist_cols: 256,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005), // ΔERB LUT step
        harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
        consonance_kernel: ConsonanceKernel {
            a: config.psychoacoustics.consonance_kernel.a,
            b: config.psychoacoustics.consonance_kernel.b,
            c: config.psychoacoustics.consonance_kernel.c,
            d: config.psychoacoustics.consonance_kernel.d,
        },
        consonance_representation: ConsonanceRepresentationParams {
            beta: config.psychoacoustics.consonance_level.beta,
            theta: config.psychoacoustics.consonance_level.theta,
            temperature: config.psychoacoustics.consonance_weight.temperature,
            epsilon: config.psychoacoustics.consonance_weight.epsilon,
        },
        roughness_scalar_mode: crate::core::landscape::RoughnessScalarMode::Total,
        roughness_half: 0.1,
        loudness_exp: config.psychoacoustics.loudness_exp, // Zwicker
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
    let hop_ms = (hop as f32 / fs) * 1000.0;
    let visual_delay_frames = 0;
    debug!(
        "Visual delay frames: {} (latency_ms={:.1}, hop_ms={:.2})",
        visual_delay_frames, latency_ms, hop_ms
    );
    let ui_channel_capacity = (visual_delay_frames + 4).max(16);

    // Channels
    let (ui_frame_tx, ui_frame_rx) = bounded::<UiFrame>(ui_channel_capacity);
    let (ctrl_tx, _ctrl_rx) = bounded::<()>(1);
    let (analysis_update_tx, analysis_update_rx) = bounded::<LandscapeUpdate>(8);

    let base_space = nsgt.space().clone();
    let analysis_stream = AnalysisStream::new(lparams.clone(), nsgt.clone());
    let landscape = Landscape::new(base_space.clone());
    let dorsal = DorsalStream::new(fs);
    let lparams_runtime = lparams.clone();

    // Analysis pipeline channels
    let (audio_to_analysis_tx, audio_to_analysis_rx) = bounded::<(u64, Arc<[f32]>)>(64);
    let (analysis_result_tx, analysis_result_rx) = bounded::<(u64, Landscape)>(4);

    // Spawn analysis thread (NSGT-RT based audio analysis).
    {
        std::thread::Builder::new()
            .name("analysis".into())
            .spawn(move || {
                analysis_worker::run(
                    analysis_stream,
                    audio_to_analysis_rx,
                    analysis_result_tx,
                    analysis_update_rx,
                )
            })
            .expect("spawn analysis worker");
    }

    let path = Path::new(&args.scenario_path);
    let scenario_label = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scenario")
        .to_string();
    let scenario = compile_scenario_from_script(path, &args, &config).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let mut pop = Population::new(crate::core::timebase::Timebase {
        fs: runtime_sample_rate as f32,
        hop,
    });
    pop.set_seed(scenario.seed);
    let conductor = Conductor::from_scenario(scenario);

    // Give the worker its own handle if WAV output is enabled.
    let wav_tx_for_worker = wav_tx;

    // Spawn worker thread
    let stop_flag_worker = stop_flag.clone();
    let start_flag = Arc::new(AtomicBool::new(!config.playback.wait_user_start));
    let start_flag_for_worker = start_flag.clone();

    let worker_handle = Some(
        thread::Builder::new()
            .name("worker".into())
            .spawn(move || {
                worker_loop(
                    scenario_label,
                    config.playback.wait_user_exit,
                    start_flag_for_worker,
                    ui_frame_tx,
                    pop,
                    conductor,
                    landscape,
                    lparams_runtime,
                    dorsal,
                    audio_prod,
                    wav_tx_for_worker,
                    guard_meter.clone(),
                    stop_flag_worker,
                    audio_to_analysis_tx,
                    analysis_result_rx,
                    analysis_update_tx,
                    hop,
                    hop_duration,
                    fs,
                )
            })
            .expect("spawn worker"),
    );

    RuntimeInit {
        ui_frame_rx,
        ctrl_tx,
        worker_handle,
        wav_handle,
        start_flag,
        audio_out,
        audio_init_error,
        visual_delay_frames,
    }
}

pub fn run_headless(args: crate::cli::Args, config: AppConfig, stop_flag: Arc<AtomicBool>) {
    let rt = init_runtime(args, config, stop_flag);
    rt.start_flag.store(true, Ordering::SeqCst);
    let _audio = rt.audio_out;
    if let Some(handle) = rt.worker_handle {
        let _ = handle.join();
    }
    if let Some(handle) = rt.wav_handle {
        let _ = handle.join();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.viewport().close_requested()) {
            // Honor OS window close requests by stopping the worker thread.
            self.exiting.store(true, Ordering::SeqCst);
        }

        if self.exiting.load(Ordering::SeqCst) {
            debug!("SIGINT received: closing window.");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        // Drain all frames into the delay queue.
        while let Ok(frame) = self.ui_frame_rx.try_recv() {
            self.ui_queue.push_back(frame);
        }
        if !self.has_ui_frame
            && let Some(frame) = self.ui_queue.pop_front()
        {
            self.apply_ui_frame(frame);
            self.has_ui_frame = true;
        }
        while self.ui_queue.len() > self.visual_delay_frames {
            if let Some(frame) = self.ui_queue.pop_front() {
                self.apply_ui_frame(frame);
            }
        }

        crate::ui::windows::main_window(
            ctx,
            &self.last_frame,
            &self.rhythm_history,
            &self.dorsal_history,
            self.audio_init_error.as_deref(),
            &self.exiting,
            &self.start_flag,
            &mut self.show_raw_nsgt_power,
        );
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        debug!("App drop. Finalizing..");

        // Request shutdown so worker threads can exit before we join them.
        self.exiting.store(true, Ordering::SeqCst);

        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.wav_handle.take() {
            let _ = handle.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_loop(
    scenario_name: String,
    wait_user_exit: bool,
    start_flag: Arc<AtomicBool>,
    ui_tx: Sender<UiFrame>,
    mut pop: Population,
    mut conductor: Conductor,
    current_landscape: Landscape,
    mut lparams: LandscapeParams,
    mut dorsal: DorsalStream,
    mut audio_prod: Option<ringbuf::HeapProd<f32>>,
    mut wav_tx: Option<Sender<Arc<[f32]>>>,
    guard_meter: Option<Arc<LimiterMeter>>,
    exiting: Arc<AtomicBool>,
    audio_to_analysis_tx: Sender<(u64, Arc<[f32]>)>,
    analysis_result_rx: Receiver<(u64, Landscape)>,
    analysis_update_tx: Sender<LandscapeUpdate>,
    hop: usize,
    hop_duration: Duration,
    fs: f32,
) {
    let mut current_landscape: LandscapeFrame = current_landscape;
    let mut playback_state = if start_flag.load(Ordering::SeqCst) {
        PlaybackState::Playing
    } else {
        PlaybackState::NotStarted
    };
    let mut finish_logged = false;
    let mut finished = false;
    let mut log_space = current_landscape.space.clone();

    let mut current_time: f32 = 0.0;
    let mut frame_idx: u64 = 0;
    let mut monitor = AudioMonitor::new();
    let mut last_guard_log = Instant::now() - Duration::from_millis(200);
    let mut last_ui_update = Instant::now();
    let ui_min_interval = Duration::from_millis(33);
    let mut last_analysis_frame: Option<u64> = None;
    let timebase = crate::core::timebase::Timebase { fs, hop };
    let mut world = crate::life::world_model::WorldModel::new(timebase, log_space.clone());
    let mut schedule_renderer = ScheduleRenderer::new(timebase);
    let init_now_tick = timebase.frame_start_tick(frame_idx);
    world.advance_to(init_now_tick);
    let mut last_tick_log = Instant::now();
    let idle_silence = vec![0.0f32; hop];
    let mut scenario_end_tick: Option<Tick> = None;
    let mut phonation_batches_buf: Vec<PhonationBatch> = Vec::new();
    let mut audio_cmds: Vec<AudioCommand> = Vec::new();
    let mut voice_targets: Vec<VoiceTarget> = Vec::new();

    // Initial UI frame so metadata is visible before playback starts.
    let init_meta = SimulationMeta {
        time_sec: current_time,
        duration_sec: conductor.total_duration(),
        agent_count: pop.individuals.len(),
        event_queue_len: conductor.remaining_events(),
        peak_level: 0.0,
        scenario_name: scenario_name.clone(),
        scene_name: conductor.current_scene_name(current_time),
        playback_state: playback_state.clone(),
        channel_peak: [0.0; 2],
        window_peak: [0.0; 2],
    };
    let init_frame = UiFrame {
        wave: WaveFrame {
            fs,
            samples: Arc::from(Vec::<f32>::new()),
        },
        spec: SpecFrame {
            spec_hz: current_landscape.space.centers_hz.clone(),
            amps: vec![0.0; current_landscape.space.n_bins()],
        },
        dorsal: DorsalFrame::default(),
        landscape: current_landscape.clone(),
        time_sec: current_time,
        meta: init_meta,
        next_gate_tick_est: None,
        theta_hz: None,
        delta_hz: None,
        pred_n_theta_per_delta: None,
        pred_tau_tick: None,
        pred_horizon_tick: None,
        pred_c_level01_next_gate: None,
        pred_gain_raw_mean: None,
        pred_gain_raw_min: None,
        pred_gain_raw_max: None,
        pred_gain_mixed_mean: None,
        pred_gain_mixed_min: None,
        pred_gain_mixed_max: None,
        pred_sync_mean: None,
        gate_boundary_in_hop: None,
        pred_available_in_hop: None,
        phonation_onsets_in_hop: None,
        agents: Vec::new(),
    };
    let _ = ui_tx.try_send(init_frame);

    loop {
        if exiting.load(Ordering::SeqCst) {
            eprintln!("Stopping worker thread.");
            break;
        }

        if playback_state == PlaybackState::NotStarted && !start_flag.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(10));
            continue;
        } else if playback_state == PlaybackState::NotStarted {
            playback_state = PlaybackState::Playing;
        }

        if finished && wait_user_exit {
            if let Some(prod) = audio_prod.as_mut() {
                while prod.vacant_len() >= hop {
                    AudioOutput::push_samples(prod, &idle_silence);
                }
            }
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let buffer_capacity = audio_prod
            .as_ref()
            .map(|p| p.capacity().get())
            .unwrap_or(hop);

        let mut produced_any = false;
        let mut process_frame = |mut prod_opt: Option<&mut ringbuf::HeapProd<f32>>| {
            produced_any = true;
            let (_free, occupancy) = if let Some(prod) = prod_opt.as_ref() {
                let free = prod.vacant_len();
                (free, buffer_capacity.saturating_sub(free))
            } else {
                (hop, 0)
            };
            let now_tick = timebase.frame_start_tick(frame_idx);
            let now_sec = timebase.tick_to_sec(now_tick);
            world.advance_to(now_tick);
            world.update_gate_from_rhythm(now_tick, &current_landscape.rhythm);
            if frame_idx == 0 || last_tick_log.elapsed() >= Duration::from_secs(1) {
                info!(
                    "[tick] frame_idx={} now_tick={} now_sec={:.6}",
                    frame_idx, now_tick, now_sec
                );
                last_tick_log = Instant::now();
            }
            pop.set_current_frame(frame_idx);

            // Spec: landscape may lag analysis by <= MAX_LANDSCAPE_LAG_FRAMES.
            loop {
                // Merge analysis results (latest-only) into the landscape.
                let mut latest_audio: Option<(u64, Landscape)> = None;
                while let Ok((analyzed_id, frame)) = analysis_result_rx.try_recv() {
                    last_analysis_frame = Some(analyzed_id);
                    latest_audio = Some((analyzed_id, frame));
                }
                let mut analysis_updated = false;
                if let Some((analysis_id, frame)) = latest_audio {
                    let space_changed = current_landscape.space.n_bins() != frame.space.n_bins()
                        || current_landscape.space.fmin != frame.space.fmin
                        || current_landscape.space.fmax != frame.space.fmax
                        || current_landscape.space.bins_per_oct != frame.space.bins_per_oct;
                    if space_changed {
                        current_landscape.resize_to_space(frame.space.clone());
                        log_space = current_landscape.space.clone();
                        world.set_space(log_space.clone());
                    }
                    current_landscape.roughness = frame.roughness;
                    current_landscape.roughness_shape_raw = frame.roughness_shape_raw;
                    current_landscape.roughness01 = frame.roughness01;
                    current_landscape.harmonicity = frame.harmonicity;
                    current_landscape.harmonicity_path_a = frame.harmonicity_path_a;
                    current_landscape.harmonicity_path_b = frame.harmonicity_path_b;
                    current_landscape.roughness_total = frame.roughness_total;
                    current_landscape.roughness_max = frame.roughness_max;
                    current_landscape.roughness_p95 = frame.roughness_p95;
                    current_landscape.roughness_scalar_raw = frame.roughness_scalar_raw;
                    current_landscape.roughness_norm = frame.roughness_norm;
                    current_landscape.roughness01_scalar = frame.roughness01_scalar;
                    current_landscape.loudness_mass = frame.loudness_mass;
                    current_landscape.root_affinity = frame.root_affinity;
                    current_landscape.overtone_affinity = frame.overtone_affinity;
                    current_landscape.binding_strength = frame.binding_strength;
                    current_landscape.harmonic_tilt = frame.harmonic_tilt;
                    current_landscape.harmonicity_mirror_weight = frame.harmonicity_mirror_weight;
                    current_landscape.subjective_intensity = frame.subjective_intensity;
                    current_landscape.nsgt_power = frame.nsgt_power;
                    current_landscape.recompute_consonance(&lparams);
                    // analysis_id is the analysis frame index from analysis_result_rx.
                    // NSGT is right-aligned; analysis represents sound up to the frame end.
                    let obs_tick = timebase.frame_end_tick(analysis_id);
                    world.observe_consonance_level01(
                        obs_tick,
                        Arc::from(current_landscape.consonance_level01.clone()),
                    );
                    analysis_updated = true;
                }
                if analysis_updated && cfg!(debug_assertions) && frame_idx.is_multiple_of(30) {
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
                        .consonance_score
                        .get(max_i)
                        .copied()
                        .unwrap_or(0.0);
                    let c_level = current_landscape
                        .consonance_level01
                        .get(max_i)
                        .copied()
                        .unwrap_or(0.0);
                    let (c_score_pred, c_level_pred) =
                        compose_consonance_score_level_with_params(h, r, &lparams);
                    debug!(
                        "c_score_check bin={} h={:.4} r={:.4} c_score={:.4} c_score_pred={:.4} c_level={:.4} c_level_pred={:.4}",
                        max_i, h, r, c_score, c_score_pred, c_level, c_level_pred
                    );
                }

                if analysis_ok(frame_idx, last_analysis_frame, MAX_LANDSCAPE_LAG_FRAMES) {
                    break;
                }
                if exiting.load(Ordering::SeqCst) {
                    break;
                }
                thread::sleep(Duration::from_micros(200));
            }

            apply_pending_landscape_update(
                &mut pop,
                &mut lparams,
                &mut current_landscape,
                &analysis_update_tx,
            );

            let t_start = Instant::now();
            conductor.dispatch_until(
                current_time,
                frame_idx,
                &current_landscape,
                None::<&mut crate::core::stream::analysis::AnalysisStream>,
                &mut pop,
            );
            pop.drain_audio_cmds(&mut audio_cmds);

            let phonation_count = if scenario_end_tick.is_none() {
                pop.collect_phonation_batches_into(
                    &mut world,
                    &current_landscape,
                    now_tick,
                    &mut phonation_batches_buf,
                )
            } else {
                0
            };
            let phonation_batches = &phonation_batches_buf[..phonation_count];
            let vitality = if pop.individuals.is_empty() {
                0.0
            } else {
                let sum: f32 = pop
                    .individuals
                    .iter()
                    .map(|agent| agent.last_signal.amplitude)
                    .sum();
                sum / pop.individuals.len() as f32
            };
            dorsal.set_vitality(vitality);

            if scenario_end_tick.is_none() && conductor.is_done() {
                scenario_end_tick = Some(now_tick);
                pop.individuals.clear();
                schedule_renderer.shutdown_at(now_tick);
            }

            pop.advance(
                hop,
                fs,
                frame_idx,
                hop_duration.as_secs_f32(),
                &current_landscape,
            );
            pop.cleanup_dead(frame_idx, hop_duration.as_secs_f32(), conductor.is_done());
            pop.fill_voice_targets(&mut voice_targets);

            // [FIX] Audio is MONO. Treat it as such.
            // Previously incorrectly treated as stereo, leading to bad metering and destructive downsampling.
            let (mono_chunk, max_abs, channel_peak) = {
                let time_chunk = schedule_renderer.render(
                    phonation_batches,
                    now_tick,
                    &current_landscape.rhythm,
                    &voice_targets,
                    &audio_cmds,
                );

                // Calculate Peak (Mono)
                let mut max_p = 0.0f32;
                for &s in time_chunk {
                    let abs_s = s.abs();
                    if abs_s > max_p {
                        max_p = abs_s;
                    }
                }

                // Output to Audio Backend
                // Note: If the audio backend expects Stereo, we might need to duplicate samples here.
                // But typically ringbuf just takes the slice. Assuming backend handles mono or we rely on OS mixing.
                if let Some(prod) = prod_opt.as_deref_mut() {
                    AudioOutput::push_samples(prod, time_chunk);
                }

                let mono_chunk: Arc<[f32]> = Arc::from(time_chunk);
                if let Some(tx) = wav_tx.as_ref() {
                    let _ = tx.try_send(Arc::clone(&mono_chunk));
                }

                // Channel peak for UI (Duplicate Mono to L/R)
                (mono_chunk, max_p, [max_p, max_p])
            };

            // [FIX] No Downmix needed. The signal is already Mono.
            if !finished {
                current_landscape.rhythm = dorsal.process(mono_chunk.as_ref());
            }
            let dorsal_metrics = dorsal.last_metrics();

            // Feed audio analysis (NSGT + peak extraction + R/H).
            let _ = audio_to_analysis_tx.try_send((frame_idx, Arc::clone(&mono_chunk)));

            // Lag is measured against generated frames, because population dynamics depend on
            // the landscape evolution in the generated timebase (not wall-clock playback).
            // landscape_age in frames (spec: <= MAX_LANDSCAPE_LAG_FRAMES after warmup).
            let analysis_lag = last_analysis_frame.map(|id| frame_idx.saturating_sub(id));

            let finished_now = if pop.abort_requested {
                true
            } else {
                scenario_end_tick.is_some() && schedule_renderer.is_idle()
            };
            if finished_now {
                playback_state = PlaybackState::Finished;
            }
            if finished_now && wav_tx.is_some() {
                wav_tx.take();
                info!("[t={:.6}] WAV closed.", current_time);
            }

            let must_send_ui = conductor.is_done() || pop.abort_requested;
            let should_send_ui = must_send_ui || last_ui_update.elapsed() >= ui_min_interval;

            let mut wave_frame: Option<WaveFrame> = None;
            let mut spec_frame: Option<SpecFrame> = None;
            let mut ui_landscape: Option<LandscapeFrame> = None;
            if should_send_ui {
                world.dorsal_metrics = Some(dorsal_metrics);
                wave_frame = Some(WaveFrame {
                    fs,
                    samples: Arc::clone(&mono_chunk),
                });
                spec_frame = Some(SpecFrame {
                    spec_hz: log_space.centers_hz.clone(),
                    amps: current_landscape
                        .nsgt_power
                        .iter()
                        .map(|&x| x.sqrt())
                        .collect(),
                });
                ui_landscape = Some(current_landscape.clone());
            }

            let elapsed = t_start.elapsed();
            let peak_level = monitor.update(
                current_time,
                frame_idx,
                hop,
                hop_duration,
                buffer_capacity,
                occupancy,
                max_abs,
                elapsed,
                analysis_lag,
                conductor.is_done(),
            );

            if let Some(meter) = guard_meter.as_ref()
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
                last_guard_log = Instant::now();
            }

            if let (Some(wave_frame), Some(spec_frame), Some(ui_landscape)) =
                (wave_frame, spec_frame, ui_landscape)
            {
                let agent_states: Vec<AgentStateInfo> = pop
                    .individuals
                    .iter()
                    .map(|agent| {
                        let f = agent.body.base_freq_hz();
                        AgentStateInfo {
                            id: agent.id,
                            freq_hz: f,
                            target_freq: 2.0f32.powf(agent.target_pitch_log2()),
                            integration_window: agent.integration_window(),
                            breath_gain: agent.articulation.gate(),
                            consonance: current_landscape.evaluate_pitch_level01(f),
                        }
                    })
                    .collect();
                let pred_stats = pop.last_pred_gate_stats();
                let theta_hz = {
                    let hz = current_landscape.rhythm.theta.freq_hz;
                    if hz.is_finite() && hz > 0.0 {
                        Some(hz)
                    } else {
                        None
                    }
                };
                let delta_hz = {
                    let hz = current_landscape.rhythm.delta.freq_hz;
                    if hz.is_finite() && hz > 0.0 {
                        Some(hz)
                    } else {
                        None
                    }
                };
                let (pred_tau_tick, pred_horizon_tick) =
                    world.predictor_tau_horizon_ticks(&current_landscape.rhythm);
                let pred_c_level01_next_gate = world.last_pred_next_gate().map(|(_, scan)| scan);
                let pred_available_in_hop = pred_c_level01_next_gate.is_some();
                let ui_frame = UiFrame {
                    wave: wave_frame,
                    spec: spec_frame,
                    dorsal: DorsalFrame {
                        e_low: dorsal_metrics.e_low,
                        e_mid: dorsal_metrics.e_mid,
                        e_high: dorsal_metrics.e_high,
                        flux: dorsal_metrics.flux,
                    },
                    landscape: ui_landscape,
                    time_sec: current_time,
                    meta: SimulationMeta {
                        time_sec: current_time,
                        duration_sec: conductor.total_duration(),
                        agent_count: pop.individuals.len(),
                        event_queue_len: conductor.remaining_events(),
                        peak_level,
                        scenario_name: scenario_name.clone(),
                        scene_name: conductor.current_scene_name(current_time),
                        playback_state: playback_state.clone(),
                        channel_peak,
                        window_peak: channel_peak,
                    },
                    next_gate_tick_est: world.next_gate_tick_est,
                    theta_hz,
                    delta_hz,
                    pred_n_theta_per_delta: Some(
                        world.predictor_n_theta_per_delta(&current_landscape.rhythm),
                    ),
                    pred_tau_tick: Some(pred_tau_tick),
                    pred_horizon_tick: Some(pred_horizon_tick),
                    pred_c_level01_next_gate,
                    pred_gain_raw_mean: pred_stats.map(|stats| stats.raw_mean),
                    pred_gain_raw_min: pred_stats.map(|stats| stats.raw_min),
                    pred_gain_raw_max: pred_stats.map(|stats| stats.raw_max),
                    pred_gain_mixed_mean: pred_stats.map(|stats| stats.mixed_mean),
                    pred_gain_mixed_min: pred_stats.map(|stats| stats.mixed_min),
                    pred_gain_mixed_max: pred_stats.map(|stats| stats.mixed_max),
                    pred_sync_mean: pred_stats.map(|stats| stats.sync_mean),
                    gate_boundary_in_hop: pop.last_gate_boundary_in_hop(),
                    pred_available_in_hop: Some(pred_available_in_hop),
                    phonation_onsets_in_hop: pop.last_phonation_onsets_in_hop(),
                    agents: agent_states,
                };
                let _ = ui_tx.try_send(ui_frame);
                last_ui_update = Instant::now();
            }

            if finished_now {
                finished = true;
                if !finish_logged {
                    let note = if wait_user_exit {
                        "Waiting for user exit."
                    } else {
                        "Exiting."
                    };
                    info!("[t={:.6}] Scenario finished. {note}", current_time);
                    finish_logged = true;
                }
                if !wait_user_exit {
                    exiting.store(true, Ordering::SeqCst);
                }
            }

            if !finished {
                current_time += hop_duration.as_secs_f32();
                frame_idx += 1;
            }
        };

        if let Some(prod) = audio_prod.as_mut() {
            while prod.vacant_len() >= hop {
                process_frame(Some(prod));
            }
        } else {
            // Offline/render-only mode: progress even without audio output.
            process_frame(None);
        }

        if exiting.load(Ordering::SeqCst) || (finished && !wait_user_exit) {
            break;
        }

        if !produced_any {
            thread::sleep(Duration::from_millis(1));
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ParamsUpdateEffect {
    harmonicity_changed: bool,
    roughness_changed: bool,
}

fn apply_pending_landscape_update(
    pop: &mut Population,
    params: &mut LandscapeParams,
    current_landscape: &mut LandscapeFrame,
    analysis_update_tx: &Sender<LandscapeUpdate>,
) {
    if let Some(update) = pop.take_pending_update() {
        let effect = apply_params_update(params, &update);
        if effect.harmonicity_changed {
            let _ = recompute_harmonicity_from_nsgt_power(current_landscape, params);
        }
        if effect.harmonicity_changed || effect.roughness_changed {
            current_landscape.recompute_consonance(params);
        }
        let _ = analysis_update_tx.try_send(update);
    }
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
    let dual = params
        .harmonicity_kernel
        .potential_h_dual_from_log2_spectrum(
            &current_landscape.nsgt_power,
            &current_landscape.space,
        );
    if dual.blended.len() != current_landscape.harmonicity.len()
        || dual.path_a.len() != current_landscape.harmonicity.len()
        || dual.path_b.len() != current_landscape.harmonicity.len()
    {
        return false;
    }
    current_landscape.harmonicity = dual.blended;
    current_landscape.harmonicity_path_a = dual.path_a;
    current_landscape.harmonicity_path_b = dual.path_b;
    current_landscape.root_affinity = dual.metrics.root_affinity;
    current_landscape.overtone_affinity = dual.metrics.overtone_affinity;
    current_landscape.binding_strength = dual.metrics.binding_strength;
    current_landscape.harmonic_tilt = dual.metrics.harmonic_tilt;
    let mirror = params.harmonicity_kernel.params.mirror_weight;
    current_landscape.harmonicity_mirror_weight = if mirror.is_finite() {
        mirror.clamp(0.0, 1.0)
    } else {
        0.0
    };
    true
}

fn apply_params_update(params: &mut LandscapeParams, upd: &LandscapeUpdate) -> ParamsUpdateEffect {
    let mut effect = ParamsUpdateEffect::default();
    if let Some(m) = upd.mirror {
        let mirror = if m.is_finite() {
            m.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let prev = params.harmonicity_kernel.params.mirror_weight;
        params.harmonicity_kernel.params.mirror_weight = mirror;
        effect.harmonicity_changed = (prev - mirror).abs() > f32::EPSILON;
    }
    if let Some(k) = upd.roughness_k {
        let roughness_k = if k.is_finite() { k.max(1e-6) } else { 1e-6 };
        let prev = params.roughness_k;
        params.roughness_k = roughness_k;
        effect.roughness_changed = (prev - roughness_k).abs() > f32::EPSILON;
    }
    effect
}

fn compose_consonance_score_level_with_params(
    h_state01: f32,
    r_state01: f32,
    params: &LandscapeParams,
) -> (f32, f32) {
    let c_score = params.consonance_kernel.score(h_state01, r_state01);
    let c_level = params.consonance_representation.level01(c_score);
    (c_score, c_level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::landscape::RoughnessScalarMode;
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

    fn build_test_params(space: &Log2Space) -> LandscapeParams {
        LandscapeParams {
            fs: 48_000.0,
            max_hist_cols: 1,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            consonance_kernel: ConsonanceKernel::default(),
            consonance_representation: ConsonanceRepresentationParams::default(),
            roughness_scalar_mode: RoughnessScalarMode::Total,
            roughness_half: 0.1,
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

    fn synthetic_nsgt_power(space: &Log2Space) -> Vec<f32> {
        let mut spectrum = vec![0.0f32; space.n_bins()];
        for (hz, amp) in [
            (196.0f32, 1.0f32),
            (347.0, 0.8),
            (523.25, 0.6),
            (739.99, 0.45),
        ] {
            if let Some(idx) = space.index_of_freq(hz) {
                spectrum[idx] = amp;
            }
        }
        spectrum
    }

    fn l2_norm_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn pending_mirror_update_changes_current_landscape_before_dispatch() {
        let space = Log2Space::new(80.0, 2_000.0, 96);
        let mut params = build_test_params(&space);
        params.harmonicity_kernel.params.mirror_weight = 0.0;

        let mut landscape = Landscape::new(space.clone());
        landscape.nsgt_power = synthetic_nsgt_power(&space);
        landscape.roughness01.fill(0.2);
        let (h0, _) = params
            .harmonicity_kernel
            .potential_h_from_log2_spectrum(&landscape.nsgt_power, &landscape.space);
        landscape.harmonicity = h0;
        landscape.recompute_consonance(&params);
        let before = landscape.harmonicity.clone();

        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 256,
        });
        pop.apply_action(
            Action::SetHarmonicityParams {
                update: LandscapeUpdate {
                    mirror: Some(1.0),
                    roughness_k: None,
                },
            },
            &landscape,
            None::<&mut crate::core::stream::analysis::AnalysisStream>,
        );

        let (analysis_update_tx, analysis_update_rx) = bounded::<LandscapeUpdate>(1);
        apply_pending_landscape_update(&mut pop, &mut params, &mut landscape, &analysis_update_tx);

        let sent = analysis_update_rx
            .try_recv()
            .expect("pending update should be forwarded to analysis");
        assert_eq!(sent.mirror, Some(1.0));
        let delta = l2_norm_diff(&before, &landscape.harmonicity);
        assert!(
            delta > 1e-3,
            "mirror update should affect current harmonicity before dispatch, delta={delta}"
        );
    }

    #[test]
    fn mirror_extremes_produce_distinct_harmonicity_potentials() {
        let space = Log2Space::new(80.0, 2_000.0, 96);
        let spectrum = synthetic_nsgt_power(&space);

        let mut root_params = HarmonicityParams::default();
        root_params.mirror_weight = 0.0;
        let mut ceil_params = root_params;
        ceil_params.mirror_weight = 1.0;

        let root_kernel = HarmonicityKernel::new(&space, root_params);
        let ceil_kernel = HarmonicityKernel::new(&space, ceil_params);
        let (h_root, _) = root_kernel.potential_h_from_log2_spectrum(&spectrum, &space);
        let (h_ceil, _) = ceil_kernel.potential_h_from_log2_spectrum(&spectrum, &space);

        let delta = l2_norm_diff(&h_root, &h_ceil);
        assert!(
            delta > 1e-3,
            "mirror=0 and mirror=1 should differ for fixed spectrum, delta={delta}"
        );
    }
}
