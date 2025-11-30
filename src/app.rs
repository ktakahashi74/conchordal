use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};
use tracing::*;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::audio::writer::WavOutput;
use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeParams};
use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::life::conductor::Conductor;
use crate::life::population::{Population, PopulationParams};
use crate::life::scenario::Scenario;
use crate::ui::viewdata::{SpecFrame, UiFrame, WaveFrame};
use crate::{audio::output::AudioOutput, core::harmonicity_kernel::HarmonicityParams};

pub struct App {
    ui_frame_rx: Receiver<UiFrame>,
    _ctrl_tx: Sender<()>, // placeholder
    last_frame: UiFrame,
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

        let path = args.scenario_path.clone();
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|err| {
            eprintln!("Failed to read scenario file {path}: {err}");
            std::process::exit(1);
        });
        let scenario = json5::from_str::<Scenario>(&contents).unwrap_or_else(|e| {
            eprintln!("Failed to parse scenario file {path}: {e}");
            std::process::exit(1);
        });
        let conductor = Conductor::from_scenario(scenario);

        // worker に渡すのは wav_tx.clone()
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
            eprintln!("SIGINT received: closing window.");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        // Pull newest frame (drain to latest)
        while let Ok(f) = self.ui_frame_rx.try_recv() {
            self.last_frame = f;
        }
        crate::ui::windows::main_window(ctx, &self.last_frame);
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        eprintln!("App drop. Finalizing..");

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
) {
    // --- Parameters ---
    let fs: f32 = 48_000.0;

    // === NSGT (log2) analyzer & Landscape parameters ===
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

    let nsgt = RtNsgtKernelLog2::new(NsgtKernelLog2::new(
        NsgtLog2Config {
            fs,
            overlap: 0.5,
            nfft_override: Some(16384),
        },
        space,
    ));

    let mut next_deadline = Instant::now();

    let nfft = nsgt.nfft();
    let hop = nsgt.hop();
    let n_bins = nfft / 2 + 1;

    println!("nfft {nfft}, hop {hop}");

    let hop_duration = Duration::from_secs_f32(hop as f32 / fs);

    let mut landscape = Landscape::new(lparams, nsgt);
    let mut current_landscape: LandscapeFrame = landscape.snapshot();

    let mut current_time: f32 = 0.0;
    let mut frame_idx: u64 = 0;

    loop {
        if exiting.load(Ordering::SeqCst) {
            eprintln!("Stopping worker thread.");
            break;
        }

        pop.set_current_frame(frame_idx);
        next_deadline += hop_duration;

        conductor.dispatch_until(current_time, frame_idx, &current_landscape, &mut pop);

        // 1) population → waveform
        let time_chunk = pop.process_audio(hop, fs, frame_idx, hop_duration.as_secs_f32());

        // send out audio signal
        if let Some(prod) = audio_prod.as_mut() {
            AudioOutput::push_samples(prod, &time_chunk);
        }

        if let Some(tx) = &wav_tx {
            let _ = tx.try_send(time_chunk.clone());
        }

        // 2) spectral body for landscape
        let body = pop.process_frame(frame_idx, n_bins, fs, nfft, hop_duration.as_secs_f32());

        // 3) landscape update using painted spectrum
        let lframe = landscape.process_precomputed_spectrum(&body);
        current_landscape = lframe.clone();

        // 4) package for UI
        let ui_frame = UiFrame {
            wave: WaveFrame {
                fs,
                samples: time_chunk,
            },
            spec: SpecFrame {
                spec_hz: lframe.space.centers_hz.clone(),
                amps: lframe.amps_last.clone(),
            },
            landscape: lframe,
        };
        let _ = ui_tx.try_send(ui_frame);

        // Check for scenario completion or explicit finish
        if pop.abort_requested || (conductor.is_done() && pop.agents.is_empty()) {
            info!("Scenario finished. Exiting.");
            exiting.store(true, Ordering::SeqCst);
            break;
        }

        // Simple timing
        let now = Instant::now();
        if now < next_deadline {
            std::thread::sleep(next_deadline - now);
        } else {
            next_deadline = now;
            trace!("worker overrun");
        }

        current_time += hop_duration.as_secs_f32();
        frame_idx += 1;
    }
}
