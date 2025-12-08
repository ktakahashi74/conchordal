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
use crate::ui::viewdata::{SpecFrame, UiFrame, WaveFrame};
use crate::{audio::output::AudioOutput, core::harmonicity_kernel::HarmonicityParams};

pub struct App {
    ui_frame_rx: Receiver<UiFrame>,
    _ctrl_tx: Sender<()>, // placeholder
    last_frame: UiFrame,
    ui_queue: VecDeque<UiFrame>,
    visual_delay_frames: usize,
    _audio: Option<AudioOutput>,
    wav_tx: Option<Sender<Vec<f32>>>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    wav_handle: Option<std::thread::JoinHandle<()>>,
    exiting: Arc<AtomicBool>,
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        args: crate::Args,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        // Channels
        let (ui_frame_tx, ui_frame_rx) = bounded::<UiFrame>(8);
        let (ctrl_tx, _ctrl_rx) = bounded::<()>(1);

        // Audio
        let (audio_out, audio_prod) = if args.play {
            let (out, prod) = AudioOutput::new(50.0);
            (Some(out), Some(prod))
        } else {
            (None, None)
        };

        // WAV
        let (wav_tx, wav_rx) = bounded::<Vec<f32>>(16);
        let wav_handle = if let Some(path) = args.wav.clone() {
            Some(WavOutput::run(wav_rx, path, 48000))
        } else {
            None
        };

        // Population (life)
        let pop = Population::new(PopulationParams {
            initial_tones_hz: vec![440.0],
            amplitude: 0.0,
        });

        // Analysis/NSGT setup
        let fs: f32 = 48_000.0;
        let space = Log2Space::new(100.0, 8000.0, 200);
        let lparams = LandscapeParams {
            fs,
            max_hist_cols: 256,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005), // ΔERB LUT step
            harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
            loudness_exp: 0.23, // Zwicker
            tau_ms: 80.0,
            ref_power: 1e-6,
            roughness_k: 0.1,
        };
        let nfft = 16_384usize;
        let nsgt = RtNsgtKernelLog2::new(NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(nfft),
            },
            space,
        ));
        let hop = nsgt.hop();
        let hop_duration = Duration::from_secs_f32(hop as f32 / fs);
        let n_bins = nfft / 2 + 1;

        let landscape = Landscape::new(lparams, nsgt.clone());

        // Analysis pipeline channels
        let (audio_to_analysis_tx, audio_to_analysis_rx) = bounded::<(u64, Vec<f32>)>(64);
        let (landscape_from_analysis_tx, landscape_from_analysis_rx) =
            bounded::<(u64, LandscapeFrame)>(4);

        // Spawn analysis thread
        {
            let landscape = landscape;
            std::thread::Builder::new()
                .name("analysis".into())
                .spawn(move || {
                    analysis_worker::run(
                        landscape,
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

        // Egui visuals tweak (dark)
        cc.egui_ctx.set_pixels_per_point(1.25);

        Self {
            ui_frame_rx,
            _ctrl_tx: ctrl_tx,
            last_frame: UiFrame::default(),
            ui_queue: VecDeque::new(),
            visual_delay_frames: 1,
            _audio: audio_out,
            wav_tx: Some(wav_tx),
            wav_handle,
            worker_handle,
            exiting: stop_flag,
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

        // Pull newest frame (drain to latest)
        while let Ok(frame) = self.ui_frame_rx.try_recv() {
            self.ui_queue.push_back(frame);
        }
        if self.ui_queue.len() > self.visual_delay_frames {
            if let Some(frame) = self.ui_queue.pop_front() {
                self.last_frame = frame;
            }
        }
        crate::ui::windows::main_window(ctx, &self.last_frame);
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

fn worker_loop(
    ui_tx: Sender<UiFrame>,
    mut pop: Population,
    mut conductor: Conductor,
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

            let time_chunk_vec = {
                let time_chunk = pop.process_audio(hop, fs, frame_idx, hop_duration.as_secs_f32());

                if last_clip_log.elapsed() > Duration::from_millis(200) {
                    if let Some(peak) = time_chunk
                        .iter()
                        .map(|v| v.abs())
                        .max_by(|a, b| a.total_cmp(b))
                    {
                        max_peak = max_peak.max(peak);
                        if peak > 0.98 {
                            warn!(
                                "[t={:.6}] Audio peak high: {:.3} at frame_idx={}. Consider more headroom.",
                                current_time, peak, frame_idx
                            );
                            last_clip_log = Instant::now();
                        } else if peak > 0.9 {
                            warn!(
                                "[t={:.6}] Audio peak nearing clip: {:.3} at frame_idx={}",
                                current_time, peak, frame_idx
                            );
                            last_clip_log = Instant::now();
                        }
                    }
                }

                AudioOutput::push_samples(prod, time_chunk);

                let chunk_vec = time_chunk.to_vec();
                if let Some(tx) = &wav_tx {
                    let _ = tx.try_send(chunk_vec.clone());
                }
                chunk_vec
            };

            // Build high-resolution spectrum for analysis (linear nfft, mapped to log space in worker).
            {
                let spectrum_body =
                    pop.process_frame(frame_idx, n_bins, fs, nfft, hop_duration.as_secs_f32());
                let _ = audio_to_analysis_tx.try_send((frame_idx, spectrum_body.to_vec()));
            }

            while let Ok((analyzed_id, lframe)) = landscape_from_analysis_rx.try_recv() {
                current_landscape = lframe;
                let lag = frame_idx.saturating_sub(analyzed_id);
                if lag >= 2 && last_lag_warn.elapsed() > Duration::from_secs(1) {
                    warn!(
                        "[t={:.3}] Analysis lag: {} frames (Audio={}, Analysis={})",
                        current_time, lag, frame_idx, analyzed_id
                    );
                    last_lag_warn = Instant::now();
                }
            }

            let ui_frame = UiFrame {
                wave: WaveFrame {
                    fs,
                    samples: time_chunk_vec,
                },
                spec: SpecFrame {
                    spec_hz: current_landscape.space.centers_hz.clone(),
                    amps: current_landscape.amps_last.clone(),
                },
                landscape: current_landscape.clone(),
            };
            let _ = ui_tx.try_send(ui_frame);

            if pop.abort_requested || (conductor.is_done() && pop.agents.is_empty()) {
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
