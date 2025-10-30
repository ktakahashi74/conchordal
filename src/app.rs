use egui::{Order, ViewportCommand};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};
use tracing::*;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::audio::output::AudioOutput;
use crate::audio::writer::WavOutput;
use crate::core::consonance_kernel::ConsonanceKernel;
use crate::core::landscape::{CVariant, Landscape, LandscapeFrame, LandscapeParams, RVariant};
use crate::core::log2::Log2Space;
use crate::core::nsgt_kernel::{BandCoeffs, NsgtKernelLog2, NsgtLog2Config};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::life::population::{Population, PopulationParams};
use crate::synth::engine::{SynthConfig, SynthEngine};
use crate::ui::viewdata::{SpecFrame, UiFrame, WaveFrame};

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
        tones: Vec<(f32, f32)>,
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

        let initial_tones_hz: Vec<f32> = tones.iter().map(|(f, _)| *f).collect();
        let amplitude = tones.iter().map(|(_, a)| *a).fold(0.0, f32::max);

        // Population (life)
        let mut pop = Population::new(PopulationParams {
            initial_tones_hz,
            amplitude,
        });

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
    pop: Population,
    mut audio_prod: Option<ringbuf::HeapProd<f32>>,
    wav_tx: Option<Sender<Vec<f32>>>,
    exiting: Arc<AtomicBool>,
) {
    // --- Parameters ---
    let fs: f32 = 48_000.0;
    //    let fft_size: usize = 16_384;
    //let hop: usize = fft_size / 2;

    // === NSGT (log2) analyzer & Landscape parameters ===
    let space = Log2Space::new(100.0, 4000.0, 128);

    let mut nsgt = RtNsgtKernelLog2::new(NsgtKernelLog2::new(
        NsgtLog2Config {
            fs,
            overlap: 0.5, // or your desired overlap (0.5 = 50%)
            nfft_override: Some(8192),
        },
        space,
    ));

    let lparams = LandscapeParams {
        fs,
        max_hist_cols: 256,
        gamma: 1.0,
        alpha: 0.0,
        r_variant: RVariant::NsgtRt,
        c_variant: CVariant::Dummy,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005), // ΔERB LUT step
        consonance_kernel: ConsonanceKernel::default(),
    };

    let mut next_deadline = Instant::now();

    let nfft = nsgt.nsgt.nfft();
    let hop = nsgt.hop();
    let n_bins = nfft / 2 + 1;

    println!("nfft {nfft}, hop {hop}");

    let hop_duration = Duration::from_secs_f32(hop as f32 / fs);

    let mut landscape = Landscape::new(lparams, nsgt);

    // Synth engine
    let mut synth = SynthEngine::new(SynthConfig {
        fs,
        hop,
        n_bins,
        fft_size: nfft,
    });

    println!("nfft = {nfft}, hop = {hop},");

    loop {
        if exiting.load(Ordering::SeqCst) {
            eprintln!("Stopping worker thread.");
            break;
        }

        next_deadline += hop_duration;

        // 1) population → spectral amplitude A[k]
        let amps = pop.project_spectrum(n_bins, fs, nfft);

        // 2) synth: render hop
        let time_chunk = synth.render_hop(&amps);

        // send out audio signal
        if let Some(prod) = audio_prod.as_mut() {
            AudioOutput::push_samples(prod, &time_chunk);
        }

        if let Some(tx) = &wav_tx {
            let _ = tx.try_send(time_chunk.clone());
        }

        // 3) landscape update (NSGT/log2-domain)
        landscape.process_frame(&time_chunk);
        let lframe: LandscapeFrame = landscape.snapshot();

        // 4) package for UI
        let ui_frame = UiFrame {
            wave: WaveFrame {
                fs,
                samples: time_chunk,
            },
            spec: SpecFrame {
                spec_hz: synth.bin_freqs_hz(),
                amps,
            },
            landscape: lframe,
        };
        let _ = ui_tx.try_send(ui_frame);

        // Simple timing
        let now = Instant::now();
        if now < next_deadline {
            std::thread::sleep(next_deadline - now);
        } else {
            next_deadline = now;
            trace!("worker overrun");
        }
    }
}
