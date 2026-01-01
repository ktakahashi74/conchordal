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

use crate::audio::writer::WavOutput;
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::harmonicity_worker;
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeParams, LandscapeUpdate};
use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
use crate::core::nsgt_rt::{RtConfig, RtNsgtKernelLog2};
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::core::roughness_worker;
use crate::core::stream::{
    dorsal::DorsalStream, harmonicity::HarmonicityStream, roughness::RoughnessStream,
};
use crate::life::conductor::Conductor;
use crate::life::individual::SoundBody;
use crate::life::population::Population;
use crate::life::scenario::{Action, Scenario, TargetRef};
use crate::life::scripting::ScriptHost;
use crate::ui::viewdata::{
    AgentStateInfo, DorsalFrame, PlaybackState, SimulationMeta, SpecFrame, UiFrame, WaveFrame,
};
use crate::{
    audio::output::AudioOutput, config::AppConfig, core::harmonicity_kernel::HarmonicityParams,
};

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
        harmonicity_lag: Option<u64>,
        roughness_lag: Option<u64>,
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

        let should_warn = harmonicity_lag.is_some_and(|lag| lag >= 2)
            || roughness_lag.is_some_and(|lag| lag >= 2);
        if should_warn && self.last_lag_warn.elapsed() > Duration::from_secs(1) {
            let h = harmonicity_lag
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into());
            let r = roughness_lag
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into());
            warn!(
                "[t={:.3}] Analysis lag (frames): H={} R={} (Audio(gen)={})",
                current_time, h, r, frame_idx
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
        format!(
            "Failed to run scenario script {}: {e:#}",
            script_path.display()
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
                Action::SpawnAgents { count, .. } => {
                    if *count == 0 {
                        return Err("SpawnAgents has count=0".to_string());
                    }
                }
                Action::AddAgent { id, .. } => {
                    if *id == 0 {
                        return Err("AddAgent has id=0".to_string());
                    }
                }
                Action::RemoveAgent { target }
                | Action::ReleaseAgent { target, .. }
                | Action::SetFreq { target, .. }
                | Action::SetAmp { target, .. }
                | Action::SetCommitment { target, .. }
                | Action::SetDrift { target, .. } => {
                    validate_target(target)?;
                }
                Action::SetRhythmVitality { .. }
                | Action::SetGlobalCoupling { .. }
                | Action::SetRoughnessTolerance { .. }
                | Action::SetHarmonicity { .. } => {}
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

fn validate_target(target: &TargetRef) -> Result<(), String> {
    match target {
        TargetRef::Range { count, .. } if *count == 0 => {
            Err("TargetRef::Range has count=0".to_string())
        }
        TargetRef::Tag { tag } if tag.trim().is_empty() => {
            Err("TargetRef::Tag has empty tag".to_string())
        }
        _ => Ok(()),
    }
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
                y: 850.0,
            }));
        cc.egui_ctx
            .send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::Vec2 {
                x: 1200.0,
                y: 850.0,
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

    // Audio
    let (audio_out, audio_prod, audio_init_error) = if args.play {
        match AudioOutput::new(latency_ms) {
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
            .map(|out| out.config.sample_rate.0)
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

    // WAV
    let (wav_tx, wav_handle) = if let Some(path) = args.wav.clone() {
        let (wav_tx, wav_rx) = bounded::<Arc<[f32]>>(16);
        let wav_handle = WavOutput::run(wav_rx, path, runtime_sample_rate);
        (Some(wav_tx), Some(wav_handle))
    } else {
        (None, None)
    };

    // Population (life)
    let pop = Population::new(runtime_sample_rate as f32);

    // Analysis/NSGT setup
    let fs: f32 = runtime_sample_rate as f32;
    let space = Log2Space::new(55.0, 8000.0, 96);
    let lparams = LandscapeParams {
        fs,
        max_hist_cols: 256,
        alpha: 0.0,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005), // ΔERB LUT step
        harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
        roughness_scalar_mode: crate::core::landscape::RoughnessScalarMode::Total,
        roughness_half: 0.1,
        consonance_roughness_weight: config.psychoacoustics.roughness_weight,
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
    let (harmonicity_tx, harmonicity_rx) = bounded::<LandscapeUpdate>(8);
    let (roughness_tx, roughness_rx) = bounded::<LandscapeUpdate>(8);

    let base_space = nsgt.space().clone();
    let roughness_stream = RoughnessStream::new(lparams.clone(), nsgt.clone());
    let harmonicity_stream =
        HarmonicityStream::new(base_space.clone(), lparams.harmonicity_kernel.clone());
    let landscape = Landscape::new(base_space.clone());
    let dorsal = DorsalStream::new(fs);
    let lparams_runtime = lparams.clone();

    // Analysis pipeline channels
    let (spectrum_to_harmonicity_tx, spectrum_to_harmonicity_rx) = bounded::<(u64, Arc<[f32]>)>(64);
    let (harmonicity_result_tx, harmonicity_result_rx) = bounded::<(u64, Vec<f32>, Vec<f32>)>(4);
    let (audio_to_roughness_tx, audio_to_roughness_rx) = bounded::<(u64, Arc<[f32]>)>(64);
    let (roughness_from_analysis_tx, roughness_from_analysis_rx) = bounded::<(u64, Landscape)>(4);

    // Spawn harmonicity thread
    {
        std::thread::Builder::new()
            .name("harmonicity".into())
            .spawn(move || {
                harmonicity_worker::run(
                    harmonicity_stream,
                    spectrum_to_harmonicity_rx,
                    harmonicity_result_tx,
                    harmonicity_rx,
                );
            })
            .expect("spawn analysis worker");
    }

    // Spawn roughness thread (NSGT-RT based audio analysis).
    {
        std::thread::Builder::new()
            .name("roughness".into())
            .spawn(move || {
                roughness_worker::run(
                    roughness_stream,
                    audio_to_roughness_rx,
                    roughness_from_analysis_tx,
                    roughness_rx,
                )
            })
            .expect("spawn roughness worker");
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
                    stop_flag_worker,
                    spectrum_to_harmonicity_tx,
                    harmonicity_result_rx,
                    harmonicity_tx,
                    audio_to_roughness_tx,
                    roughness_from_analysis_rx,
                    roughness_tx,
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
    exiting: Arc<AtomicBool>,
    spectrum_to_harmonicity_tx: Sender<(u64, Arc<[f32]>)>,
    harmonicity_result_rx: Receiver<(u64, Vec<f32>, Vec<f32>)>,
    harmonicity_tx: Sender<LandscapeUpdate>,
    audio_to_roughness_tx: Sender<(u64, Arc<[f32]>)>,
    roughness_from_analysis_rx: Receiver<(u64, Landscape)>,
    roughness_tx: Sender<LandscapeUpdate>,
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
    let mut latest_spec_amps: Vec<f32> = vec![0.0; current_landscape.space.n_bins()];
    let mut log_space = current_landscape.space.clone();

    let mut current_time: f32 = 0.0;
    let mut frame_idx: u64 = 0;
    let mut monitor = AudioMonitor::new();
    let mut last_ui_update = Instant::now();
    let ui_min_interval = Duration::from_millis(33);
    let mut last_h_analysis_frame: Option<u64> = None;
    let mut last_r_analysis_frame: Option<u64> = None;
    let mut latest_h_scan: Option<Vec<f32>> = None;

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
            pop.set_current_frame(frame_idx);

            // Keep analysis aligned to generated frames: allow 1-frame delay, but do not advance
            // more than that. This keeps `C/R/H` coherent for population dynamics.
            let required_prev = frame_idx.saturating_sub(1);
            loop {
                // Merge analysis results (latest-only) into the landscape.
                let mut latest_body: Option<(u64, Vec<f32>, Vec<f32>)> = None;
                while let Ok((analyzed_id, h_scan, body_log)) = harmonicity_result_rx.try_recv() {
                    last_h_analysis_frame = Some(analyzed_id);
                    latest_body = Some((analyzed_id, h_scan, body_log));
                }
                if let Some((_, h_scan, _body_log)) = latest_body {
                    latest_h_scan = Some(h_scan);
                }

                let mut latest_audio: Option<(u64, Landscape)> = None;
                while let Ok((analyzed_id, frame)) = roughness_from_analysis_rx.try_recv() {
                    last_r_analysis_frame = Some(analyzed_id);
                    latest_audio = Some((analyzed_id, frame));
                }
                let mut roughness_updated = false;
                let mut harmonicity_updated = false;
                if let Some((_, frame)) = latest_audio {
                    let space_changed = current_landscape.space.n_bins() != frame.space.n_bins()
                        || current_landscape.space.fmin != frame.space.fmin
                        || current_landscape.space.fmax != frame.space.fmax
                        || current_landscape.space.bins_per_oct != frame.space.bins_per_oct;
                    if space_changed {
                        current_landscape.resize_to_space(frame.space.clone());
                        log_space = current_landscape.space.clone();
                    }
                    current_landscape.roughness = frame.roughness;
                    current_landscape.roughness01 = frame.roughness01;
                    current_landscape.roughness_total = frame.roughness_total;
                    current_landscape.roughness_max = frame.roughness_max;
                    current_landscape.roughness_p95 = frame.roughness_p95;
                    current_landscape.roughness_scalar_raw = frame.roughness_scalar_raw;
                    current_landscape.roughness_norm = frame.roughness_norm;
                    current_landscape.roughness01_scalar = frame.roughness01_scalar;
                    current_landscape.loudness_mass = frame.loudness_mass;
                    current_landscape.subjective_intensity = frame.subjective_intensity;
                    current_landscape.nsgt_power = frame.nsgt_power;
                    roughness_updated = true;
                }

                if let Some(h_scan) = &latest_h_scan
                    && h_scan.len() == current_landscape.harmonicity.len()
                {
                    current_landscape.harmonicity.clone_from(h_scan);
                    harmonicity_updated = true;
                }

                if roughness_updated || harmonicity_updated {
                    current_landscape.recompute_consonance(&lparams);
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
                        let c = current_landscape
                            .consonance
                            .get(max_i)
                            .copied()
                            .unwrap_or(0.0);
                        let c_pred = (h - lparams.consonance_roughness_weight * r).clamp(-1.0, 1.0);
                        debug!(
                            "c_signed_check bin={} h={:.4} r={:.4} c={:.4} c_pred={:.4}",
                            max_i, h, r, c, c_pred
                        );
                    }
                }

                // For frame 0 there is no previous frame to wait for.
                if frame_idx == 0 {
                    break;
                }
                let h_ok = last_h_analysis_frame.is_some_and(|id| id >= required_prev);
                let r_ok = last_r_analysis_frame.is_some_and(|id| id >= required_prev);
                if h_ok && r_ok {
                    break;
                }
                if exiting.load(Ordering::SeqCst) {
                    break;
                }
                thread::sleep(Duration::from_micros(200));
            }

            let t_start = Instant::now();
            conductor.dispatch_until(
                current_time,
                frame_idx,
                &current_landscape,
                None::<&mut crate::core::stream::roughness::RoughnessStream>,
                &mut pop,
            );
            dorsal.set_vitality(pop.global_vitality);
            if let Some(update) = pop.take_pending_update() {
                apply_params_update(&mut lparams, &update);
                current_landscape.recompute_consonance(&lparams);
                let _ = harmonicity_tx.try_send(update);
                let _ = roughness_tx.try_send(update);
            }

            // [FIX] Audio is MONO. Treat it as such.
            // Previously incorrectly treated as stereo, leading to bad metering and destructive downsampling.
            let (mono_chunk, max_abs, channel_peak) = {
                let time_chunk = pop.process_audio(
                    hop,
                    fs,
                    frame_idx,
                    hop_duration.as_secs_f32(),
                    &current_landscape,
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

            // Build log2 spectrum for analysis and UI (aligned with landscape space).
            let spectrum_body = pop.process_frame(
                frame_idx,
                &current_landscape.space,
                hop_duration.as_secs_f32(),
                conductor.is_done(),
            );
            latest_spec_amps.clear();
            latest_spec_amps.extend_from_slice(spectrum_body);
            let spectrum_body = Arc::from(spectrum_body);
            let _ = spectrum_to_harmonicity_tx.try_send((frame_idx, spectrum_body));
            let _ = audio_to_roughness_tx.try_send((frame_idx, Arc::clone(&mono_chunk)));

            // Lag is measured against generated frames, because population dynamics depend on
            // the landscape evolution in the generated timebase (not wall-clock playback).
            let harmonicity_lag = last_h_analysis_frame.map(|id| frame_idx.saturating_sub(id));
            let roughness_lag = last_r_analysis_frame.map(|id| frame_idx.saturating_sub(id));

            let finished_now =
                pop.individuals.is_empty() && (conductor.is_done() || pop.abort_requested);
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
                wave_frame = Some(WaveFrame {
                    fs,
                    samples: Arc::clone(&mono_chunk),
                });
                spec_frame = Some(SpecFrame {
                    spec_hz: log_space.centers_hz.clone(),
                    amps: latest_spec_amps.iter().map(|&x| x.sqrt()).collect(),
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
                harmonicity_lag,
                roughness_lag,
                conductor.is_done(),
            );

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
                            target_freq: 2.0f32.powf(agent.target_pitch_log2),
                            integration_window: agent.integration_window,
                            breath_gain: agent.articulation.gate(),
                            consonance: current_landscape.evaluate_pitch01(f),
                        }
                    })
                    .collect();
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

fn apply_params_update(params: &mut LandscapeParams, upd: &LandscapeUpdate) {
    if let Some(m) = upd.mirror {
        params.harmonicity_kernel.params.mirror_weight = m;
    }
    if let Some(l) = upd.limit {
        params.harmonicity_kernel.params.param_limit = l;
    }
    if let Some(k) = upd.roughness_k {
        params.roughness_k = k.max(1e-6);
    }
}
