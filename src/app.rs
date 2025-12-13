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
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeParams};
use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::life::analysis_worker;
use crate::life::conductor::Conductor;
use crate::life::population::{Population, PopulationParams};
use crate::life::scenario::Scenario;
use crate::life::scripting::ScriptHost;
use crate::ui::viewdata::{SimulationMeta, SpecFrame, UiFrame, WaveFrame};
use crate::{
    audio::output::AudioOutput, config::AppConfig, core::harmonicity_kernel::HarmonicityParams,
};

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
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        args: crate::Args,
        config: AppConfig,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
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
        let space = Log2Space::new(55.0, 8000.0, 200);
        let lparams = LandscapeParams {
            fs,
            max_hist_cols: 256,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005), // ΔERB LUT step
            harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
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

        let landscape = Landscape::new(lparams.clone(), nsgt.clone());
        let analysis_landscape = Landscape::new(lparams, nsgt.clone());

        // Analysis pipeline channels
        let (audio_to_analysis_tx, audio_to_analysis_rx) = bounded::<(u64, Vec<f32>)>(64);
        let (landscape_from_analysis_tx, landscape_from_analysis_rx) =
            bounded::<(u64, LandscapeFrame)>(4);

        // Spawn analysis thread
        {
            std::thread::Builder::new()
                .name("analysis".into())
                .spawn(move || {
                    analysis_worker::run(
                        analysis_landscape,
                        audio_to_analysis_rx,
                        landscape_from_analysis_tx,
                    )
                })
                .expect("spawn analysis worker");
        }

        let path = args.scenario_path.clone();
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
        let worker_handle = Some(
            thread::Builder::new()
                .name("worker".into())
                .spawn(move || {
                    worker_loop(
                        ui_frame_tx,
                        pop,
                        conductor,
                        landscape,
                        audio_prod,
                        wav_tx_for_worker,
                        stop_flag_worker,
                        audio_to_analysis_tx,
                        landscape_from_analysis_rx,
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
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.exiting.load(Ordering::SeqCst) {
            debug!("SIGINT received: closing window.");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        // Drain all frames for high-frequency rhythm updates.
        while let Ok(frame) = self.ui_frame_rx.try_recv() {
            let t = frame.time_sec as f64;
            self.rhythm_history.push_back((t, frame.landscape.rhythm));
            self.ui_queue.push_back(frame);
        }
        if let Some((t_last, _)) = self.rhythm_history.back().copied() {
            while self
                .rhythm_history
                .front()
                .map_or(false, |(time, _)| *time < t_last - 5.0)
            {
                self.rhythm_history.pop_front();
            }
        }
        while !self.ui_queue.is_empty() {
            if let Some(frame) = self.ui_queue.pop_front() {
                self.last_frame = frame;
            }
        }
        crate::ui::windows::main_window(
            ctx,
            &self.last_frame,
            &self.rhythm_history,
            self.audio_init_error.as_deref(),
        );
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        debug!("App drop. Finalizing..");

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
    ui_tx: Sender<UiFrame>,
    mut pop: Population,
    mut conductor: Conductor,
    mut landscape: Landscape,
    mut audio_prod: Option<ringbuf::HeapProd<f32>>,
    wav_tx: Option<Sender<Vec<f32>>>,
    exiting: Arc<AtomicBool>,
    audio_to_analysis_tx: Sender<(u64, Vec<f32>)>,
    landscape_from_analysis_rx: Receiver<(u64, LandscapeFrame)>,
    hop: usize,
    hop_duration: Duration,
    fs: f32,
    n_bins: usize,
    nfft: usize,
) {
    let mut current_landscape: LandscapeFrame = LandscapeFrame::default();

    let mut current_time: f32 = 0.0;
    let mut frame_idx: u64 = 0;
    let mut last_clip_log = Instant::now();
    let mut interval_start = Instant::now();
    let mut min_occupancy: Option<usize> = None;
    let mut max_peak: f32 = 0.0;
    let mut slow_chunks: u32 = 0;
    let mut last_lag_warn = Instant::now();

    loop {
        if exiting.load(Ordering::SeqCst) {
            eprintln!("Stopping worker thread.");
            break;
        }

        if audio_prod.is_none() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let prod = audio_prod.as_mut().expect("audio producer exists");

        // Diagnostics: monitor buffer occupancy (producer side).
        let free = prod.vacant_len();
        let cap = prod.capacity().get();
        let occ = cap.saturating_sub(free);
        min_occupancy = Some(min_occupancy.map_or(occ, |m| m.min(occ)));

        let mut produced_any = false;
        while prod.vacant_len() >= hop {
            produced_any = true;
            pop.set_current_frame(frame_idx);

            let t_start = Instant::now();
            conductor.dispatch_until(current_time, frame_idx, &current_landscape, &mut pop);
            landscape.set_vitality(pop.global_vitality);

            let (time_chunk_vec, max_abs) = {
                let time_chunk =
                    pop.process_audio(hop, fs, frame_idx, hop_duration.as_secs_f32(), &landscape);

                let max_abs = time_chunk
                    .iter()
                    .map(|v| v.abs())
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(0.0);

                if last_clip_log.elapsed() > Duration::from_millis(200) {
                    max_peak = max_peak.max(max_abs);
                    if max_abs > 0.98 {
                        warn!(
                            "[t={:.6}] Audio peak high: {:.3} at frame_idx={}. Consider more headroom.",
                            current_time, max_abs, frame_idx
                        );
                        last_clip_log = Instant::now();
                    } else if max_abs > 0.9 {
                        warn!(
                            "[t={:.6}] Audio peak nearing clip: {:.3} at frame_idx={}",
                            current_time, max_abs, frame_idx
                        );
                        last_clip_log = Instant::now();
                    }
                }

                AudioOutput::push_samples(prod, time_chunk);

                let chunk_vec = time_chunk.to_vec();
                if let Some(tx) = &wav_tx {
                    let _ = tx.try_send(chunk_vec.clone());
                }
                (chunk_vec, max_abs)
            };

            landscape.update_rhythm(&time_chunk_vec);

            // Build high-resolution spectrum for analysis (linear nfft, mapped to log space in worker).
            {
                let spectrum_body =
                    pop.process_frame(frame_idx, n_bins, fs, nfft, hop_duration.as_secs_f32());
                let _ = audio_to_analysis_tx.try_send((frame_idx, spectrum_body.to_vec()));
            }

            let mut latest_analysis: Option<(u64, LandscapeFrame)> = None;
            while let Ok((analyzed_id, lframe)) = landscape_from_analysis_rx.try_recv() {
                latest_analysis = Some((analyzed_id, lframe));
            }
            if let Some((analyzed_id, lframe)) = latest_analysis {
                current_landscape = lframe;
                landscape.apply_frame(&current_landscape);
                let lag = frame_idx.saturating_sub(analyzed_id);
                if lag >= 2 && last_lag_warn.elapsed() > Duration::from_secs(1) {
                    warn!(
                        "[t={:.3}] Analysis lag: {} frames (Audio={}, Analysis={})",
                        current_time, lag, frame_idx, analyzed_id
                    );
                    last_lag_warn = Instant::now();
                }
            }

            let mut ui_landscape = current_landscape.clone();
            ui_landscape.rhythm = landscape.rhythm;

            let ui_frame = UiFrame {
                wave: WaveFrame {
                    fs,
                    samples: time_chunk_vec,
                },
                spec: SpecFrame {
                    spec_hz: current_landscape.space.centers_hz.clone(),
                    amps: current_landscape.amps_last.clone(),
                },
                landscape: ui_landscape,
                time_sec: current_time,
                meta: SimulationMeta {
                    time_sec: current_time,
                    duration_sec: conductor.total_duration(),
                    agent_count: pop.individuals.len(),
                    event_queue_len: conductor.remaining_events(),
                    peak_level: max_abs,
                },
            };
            let _ = ui_tx.try_send(ui_frame);

            if pop.abort_requested || (conductor.is_done() && pop.individuals.is_empty()) {
                info!("[t={:.6}] Scenario finished. Exiting.", current_time);
                exiting.store(true, Ordering::SeqCst);
                break;
            }

            current_time += hop_duration.as_secs_f32();
            frame_idx += 1;

            let elapsed = t_start.elapsed();
            if elapsed > hop_duration {
                slow_chunks += 1;
                warn!(
                    "[t={:.6}] Audio chunk compute slow: {:?} (hop {:?}) frame_idx={}",
                    current_time, elapsed, hop_duration, frame_idx
                );
            }
        }

        if exiting.load(Ordering::SeqCst) {
            break;
        }

        if !produced_any {
            thread::sleep(Duration::from_millis(1));
        }

        if interval_start.elapsed() > Duration::from_secs(1) {
            if let Some(min_occ) = min_occupancy.take() {
                let cap = prod.capacity().get();
                debug!(
                    "[t={:.6}] Audio stats: min_occ={}, cap={}, hop={}, max_peak={:.3}, slow_chunks={}",
                    current_time, min_occ, cap, hop, max_peak, slow_chunks
                );
            }
            max_peak = 0.0;
            slow_chunks = 0;
            interval_start = Instant::now();
        }
    }
}
