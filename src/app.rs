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
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::core::roughness_worker;
use crate::core::stream::{
    dorsal::DorsalStream, harmonicity::HarmonicityStream, roughness::RoughnessStream,
};
use crate::life::conductor::Conductor;
use crate::life::individual::SoundBody;
use crate::life::population::{Population, PopulationParams};
use crate::life::scenario::Scenario;
use crate::life::scripting::ScriptHost;
use crate::ui::viewdata::{
    AgentStateInfo, PlaybackState, SimulationMeta, SpecFrame, UiFrame, WaveFrame,
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
    wav_tx: Option<Sender<Vec<f32>>>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    wav_handle: Option<std::thread::JoinHandle<()>>,
    exiting: Arc<AtomicBool>,
    rhythm_history: VecDeque<(f64, crate::core::modulation::NeuralRhythms)>,
    start_flag: Arc<AtomicBool>,
    level_history: VecDeque<(std::time::Instant, [f32; 2])>,
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        args: crate::Args,
        config: AppConfig,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        let latency_ms = config.audio.latency_ms;

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

        // WAV
        let (wav_tx, wav_rx) = bounded::<Vec<f32>>(16);
        let wav_handle = if let Some(path) = args.wav.clone() {
            Some(WavOutput::run(wav_rx, path, config.audio.sample_rate))
        } else {
            None
        };

        // Population (life)
        let pop = Population::new(PopulationParams {
            initial_tones_hz: vec![440.0],
            amplitude: 0.0,
        });

        // Analysis/NSGT setup
        let fs: f32 = config.audio.sample_rate as f32;
        let space = Log2Space::new(55.0, 8000.0, 100);
        let lparams = LandscapeParams {
            fs,
            max_hist_cols: 256,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005), // ΔERB LUT step
            harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
            habituation_tau: 8.0,
            habituation_weight: 0.5,
            habituation_max_depth: 1.0,
            loudness_exp: config.psychoacoustics.loudness_exp, // Zwicker
            tau_ms: config.analysis.tau_ms,
            ref_power: 1e-6,
            roughness_k: config.psychoacoustics.roughness_k,
        };
        let nfft = config.analysis.nfft;
        let hop = config.analysis.hop_size;
        let overlap = 1.0 - (hop as f32 / nfft as f32);
        let nsgt_kernel = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap,
                nfft_override: Some(nfft),
            },
            space,
        );
        let nsgt = RtNsgtKernelLog2::new(nsgt_kernel.clone());
        let hop_duration = Duration::from_secs_f32(hop as f32 / fs);
        let n_bins = nfft / 2 + 1;
        let hop_ms = (hop as f32 / fs) * 1000.0;
        let visual_delay_frames = (latency_ms / hop_ms).ceil() as usize + 1; // small safety margin
        let ui_channel_capacity = (visual_delay_frames + 4).max(16);

        // Channels
        let (ui_frame_tx, ui_frame_rx) = bounded::<UiFrame>(ui_channel_capacity);
        let (ctrl_tx, _ctrl_rx) = bounded::<()>(1);
        let (harmonicity_tx, harmonicity_rx) = bounded::<LandscapeUpdate>(8);
        let (roughness_tx, roughness_rx) = bounded::<LandscapeUpdate>(8);

        let base_space = nsgt.space().clone();
        let roughness_stream = RoughnessStream::new(lparams.clone(), nsgt.clone());
        let harmonicity_stream = HarmonicityStream::new(
            lparams.fs,
            base_space.clone(),
            lparams.harmonicity_kernel.clone(),
        );
        let landscape = Landscape::new(base_space.clone());
        let dorsal = DorsalStream::new(fs);
        let lparams_runtime = lparams.clone();

        // Analysis pipeline channels
        let (spectrum_to_harmonicity_tx, spectrum_to_harmonicity_rx) =
            bounded::<(u64, Vec<f32>)>(64);
        let (harmonicity_result_tx, harmonicity_result_rx) =
            bounded::<(u64, Vec<f32>, Vec<f32>)>(4);
        let (audio_to_roughness_tx, audio_to_roughness_rx) = bounded::<(u64, Vec<f32>)>(64);
        let (roughness_from_analysis_tx, roughness_from_analysis_rx) =
            bounded::<(u64, Landscape)>(4);

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

        let path = args.scenario_path.clone();
        let scenario_label = Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("scenario")
            .to_string();
        let ext = Path::new(&path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let conductor = match ext.as_str() {
            "rhai" => {
                let scenario = ScriptHost::load_script(&path).unwrap_or_else(|e| {
                    eprintln!("Failed to run scenario script {path}: {e}");
                    std::process::exit(1);
                });
                Conductor::from_scenario(scenario)
            }
            "json" | "json5" => {
                let contents = std::fs::read_to_string(&path).unwrap_or_else(|err| {
                    eprintln!("Failed to read scenario file {path}: {err}");
                    std::process::exit(1);
                });
                let scenario = json5::from_str::<Scenario>(&contents).unwrap_or_else(|e| {
                    eprintln!("Failed to parse scenario file {path}: {e}");
                    std::process::exit(1);
                });
                Conductor::from_scenario(scenario)
            }
            _ => {
                eprintln!("Unsupported scenario extension for {path}");
                std::process::exit(1);
            }
        };

        // Give the worker its own handle if WAV output is enabled.
        let wav_tx_for_worker = if args.wav.is_some() {
            Some(wav_tx.clone())
        } else {
            None
        };

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
                        n_bins,
                        nfft,
                    )
                })
                .expect("spawn worker"),
        );

        //cc.egui_ctx.set_pixels_per_point(1.0);
        cc.egui_ctx
            .send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::Vec2 {
                x: 1200.0,
                y: 1600.0,
            }));
        cc.egui_ctx
            .send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::Vec2 {
                x: 1200.0,
                y: 1600.0,
            }));

        Self {
            ui_frame_rx,
            _ctrl_tx: ctrl_tx,
            last_frame: UiFrame::default(),
            ui_queue: VecDeque::new(),
            visual_delay_frames,
            _audio: audio_out,
            audio_init_error,
            wav_tx: Some(wav_tx),
            wav_handle,
            worker_handle,
            exiting: stop_flag,
            rhythm_history: VecDeque::with_capacity(4096),
            start_flag,
            level_history: VecDeque::with_capacity(256),
        }
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

        // Drain all frames for high-frequency rhythm updates.
        while let Ok(frame) = self.ui_frame_rx.try_recv() {
            let t = frame.time_sec as f64;
            if frame.meta.playback_state != PlaybackState::Finished {
                self.rhythm_history.push_back((t, frame.landscape.rhythm));
            }
            self.ui_queue.push_back(frame);
        }
        if let Some((t_last, _)) = self.rhythm_history.back().copied() {
            while self
                .rhythm_history
                .front()
                .is_some_and(|(time, _)| *time < t_last - 5.0)
            {
                self.rhythm_history.pop_front();
            }
        }
        while !self.ui_queue.is_empty() {
            if let Some(frame) = self.ui_queue.pop_front() {
                self.last_frame = frame;
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

        if self.last_frame.meta.playback_state == PlaybackState::Finished {
            self.wav_tx.take();
        }

        crate::ui::windows::main_window(
            ctx,
            &self.last_frame,
            &self.rhythm_history,
            self.audio_init_error.as_deref(),
            &self.exiting,
            &self.start_flag,
        );
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        debug!("App drop. Finalizing..");

        // Request shutdown so worker threads can exit before we join them.
        self.exiting.store(true, Ordering::SeqCst);

        self.wav_tx.take();

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
    wav_tx: Option<Sender<Vec<f32>>>,
    exiting: Arc<AtomicBool>,
    spectrum_to_harmonicity_tx: Sender<(u64, Vec<f32>)>,
    harmonicity_result_rx: Receiver<(u64, Vec<f32>, Vec<f32>)>,
    harmonicity_tx: Sender<LandscapeUpdate>,
    audio_to_roughness_tx: Sender<(u64, Vec<f32>)>,
    roughness_from_analysis_rx: Receiver<(u64, Landscape)>,
    roughness_tx: Sender<LandscapeUpdate>,
    hop: usize,
    hop_duration: Duration,
    fs: f32,
    n_bins: usize,
    nfft: usize,
) {
    let mut current_landscape: LandscapeFrame = current_landscape;
    let mut playback_state = if start_flag.load(Ordering::SeqCst) {
        PlaybackState::Playing
    } else {
        PlaybackState::NotStarted
    };
    let mut finish_logged = false;
    let mut finished = false;
    let mut latest_spec_amps: Vec<f32> = vec![0.0; n_bins];
    let log_space = current_landscape.space.clone();

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
            samples: Vec::new(),
        },
        spec: SpecFrame {
            spec_hz: current_landscape.space.centers_hz.clone(),
            amps: vec![0.0; current_landscape.space.n_bins()],
        },
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

        if audio_prod.is_none() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let prod = audio_prod.as_mut().expect("audio producer exists");
        let buffer_capacity = prod.capacity().get();

        let mut produced_any = false;
        while prod.vacant_len() >= hop {
            produced_any = true;
            let free = prod.vacant_len();
            let occupancy = buffer_capacity.saturating_sub(free);
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
                if let Some((_, frame)) = latest_audio {
                    if current_landscape.space.n_bins() != frame.space.n_bins() {
                        current_landscape.space = frame.space.clone();
                    }
                    current_landscape.roughness = frame.roughness;
                    current_landscape.roughness_total = frame.roughness_total;
                    current_landscape.habituation = frame.habituation;
                    current_landscape.subjective_intensity = frame.subjective_intensity;
                }

                if let Some(h_scan) = &latest_h_scan {
                    if h_scan.len() == current_landscape.harmonicity.len() {
                        current_landscape.harmonicity.clone_from(h_scan);
                        current_landscape.recompute_consonance(&lparams);
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

            let (time_chunk_vec, max_abs, channel_peak) = {
                let time_chunk = pop.process_audio(
                    hop,
                    fs,
                    frame_idx,
                    hop_duration.as_secs_f32(),
                    &current_landscape,
                );

                let max_abs = time_chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

                let mut channel_peak = [0.0f32; 2];
                for (idx, &sample) in time_chunk.iter().enumerate() {
                    let ch = idx % 2;
                    channel_peak[ch] = channel_peak[ch].max(sample.abs());
                }
                // If mono, mirror to right channel for display.
                if channel_peak[1] == 0.0 {
                    channel_peak[1] = channel_peak[0];
                }

                AudioOutput::push_samples(prod, time_chunk);

                let chunk_vec = time_chunk.to_vec();
                if let Some(tx) = &wav_tx {
                    let _ = tx.try_send(chunk_vec.clone());
                }
                (chunk_vec, max_abs, channel_peak)
            };

            // Downmix interleaved stereo (L, R, L, R...) to mono for rhythm/roughness paths.
            let mono_chunk: Vec<f32> = time_chunk_vec
                .chunks_exact(2)
                .map(|frame| (frame[0] + frame[1]) * 0.5)
                .collect();
            if !finished {
                current_landscape.rhythm = dorsal.process(&mono_chunk);
            }

            // Build high-resolution spectrum for analysis (linear nfft, mapped to log space in worker).
            let spectrum_body = pop.process_frame(
                frame_idx,
                n_bins,
                fs,
                nfft,
                hop_duration.as_secs_f32(),
                conductor.is_done(),
            );
            latest_spec_amps.clear();
            latest_spec_amps.extend_from_slice(spectrum_body);
            let _ = spectrum_to_harmonicity_tx.try_send((frame_idx, spectrum_body.to_vec()));
            let _ = audio_to_roughness_tx.try_send((frame_idx, mono_chunk.clone()));

            // Lag is measured against generated frames, because population dynamics depend on
            // the landscape evolution in the generated timebase (not wall-clock playback).
            let harmonicity_lag = last_h_analysis_frame.map(|id| frame_idx.saturating_sub(id));
            let roughness_lag = last_r_analysis_frame.map(|id| frame_idx.saturating_sub(id));

            let finished_now =
                pop.abort_requested || (conductor.is_done() && pop.individuals.is_empty());
            if finished_now {
                playback_state = PlaybackState::Finished;
            }

            let must_send_ui = conductor.is_done() || pop.abort_requested;
            let should_send_ui = must_send_ui || last_ui_update.elapsed() >= ui_min_interval;

            let mut wave_frame: Option<WaveFrame> = None;
            let mut spec_frame: Option<SpecFrame> = None;
            let mut ui_landscape: Option<LandscapeFrame> = None;
            if should_send_ui {
                // Map linear spectrum to log2 bins for UI only when we intend to send.
                let mut ui_log_amps = vec![0.0f32; log_space.n_bins()];
                for (i, &amp) in latest_spec_amps.iter().enumerate() {
                    let f = i as f32 * fs / nfft as f32;
                    if let Some(idx) = log_space.index_of_freq(f)
                        && let Some(slot) = ui_log_amps.get_mut(idx)
                    {
                        *slot += amp;
                    }
                }
                wave_frame = Some(WaveFrame {
                    fs,
                    samples: time_chunk_vec.clone(),
                });
                spec_frame = Some(SpecFrame {
                    spec_hz: log_space.centers_hz.clone(),
                    amps: ui_log_amps.iter().map(|&x| x.sqrt()).collect(),
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
                    .map(|agent| match agent {
                        crate::life::individual::IndividualWrapper::PureTone(ind) => {
                            let f = ind.body.base_freq_hz();
                            AgentStateInfo {
                                id: ind.id,
                                freq_hz: f,
                                target_freq: ind.target_freq,
                                integration_window: ind.integration_window,
                                breath_gain: ind.breath_gain,
                                consonance: current_landscape.evaluate_pitch(f),
                                habituation: current_landscape.get_habituation_at(f),
                            }
                        }
                        crate::life::individual::IndividualWrapper::Harmonic(ind) => {
                            let f = ind.body.base_freq_hz();
                            AgentStateInfo {
                                id: ind.id,
                                freq_hz: f,
                                target_freq: ind.target_freq,
                                integration_window: ind.integration_window,
                                breath_gain: ind.breath_gain,
                                consonance: current_landscape.evaluate_pitch(f),
                                habituation: current_landscape.get_habituation_at(f),
                            }
                        }
                    })
                    .collect();
                let ui_frame = UiFrame {
                    wave: wave_frame,
                    spec: spec_frame,
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
    if let Some(w) = upd.habituation_weight {
        params.habituation_weight = w;
    }
    if let Some(tau) = upd.habituation_tau {
        params.habituation_tau = tau;
    }
    if let Some(max_d) = upd.habituation_max_depth {
        params.habituation_max_depth = max_d;
    }
}
