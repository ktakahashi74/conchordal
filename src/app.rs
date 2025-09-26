use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use egui::{Order, ViewportCommand};
use tracing::*;

use crate::audio::output::AudioOutput;
use crate::audio::writer::WavOutput;
use clap::Parser;
use ringbuf::HeapProd;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::core::landscape::{LandscapeFrame, LandscapeParams};
use crate::life::population::{Population, PopulationParams};
use crate::synth::engine::{SynthConfig, SynthEngine};
use crate::ui::viewdata::{UiFrame, WaveChunk};

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
    pub fn new(cc: &eframe::CreationContext<'_>, args: crate::Args, stop_flag: Arc<AtomicBool>) -> Self {
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
	
        // worker に渡すのは wav_tx.clone()
        let wav_tx_for_worker = if args.wav.is_some() {
            Some(wav_tx.clone())
        } else {
            None
        };
	

        // Spawn worker thread
	let stop_flag_worker = stop_flag.clone();
        let worker_handle = Some(thread::Builder::new()
            .name("worker".into())
            .spawn(move || worker_loop(ui_frame_tx, audio_prod, wav_tx_for_worker, stop_flag_worker))
            .expect("spawn worker"));

        // Egui visuals tweak (dark)
        cc.egui_ctx.set_pixels_per_point(1.25);

        Self {
            ui_frame_rx,
            _ctrl_tx: ctrl_tx,
            last_frame: UiFrame::empty(),
	    _audio: audio_out,
	    wav_tx: Some(wav_tx),
	    wav_handle: wav_handle,
	    worker_handle: worker_handle,
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


fn worker_loop(ui_tx: Sender<UiFrame>,
	       mut audio_prod: Option<ringbuf::HeapProd<f32>>,
	       wav_tx: Option<Sender<Vec<f32>>>,
	       exiting: Arc<AtomicBool>) {
    // --- Parameters ---
    let fs: f32 = 48_000.0;
    let fft_size: usize = 4096;
    let hop: usize = fft_size / 2;
    let n_bins = fft_size / 2 + 1;

    // Population (life) — single pure tone at 440 Hz as requested
    let mut pop = Population::new(PopulationParams {
        initial_tones_hz: vec![440.0],
        amplitude: 100.0,
    });

    // Synth engine (phase-accumulator iSTFT skeleton)
    let mut synth = SynthEngine::new(SynthConfig {
        fs,
        fft_size,
        hop,
        n_bins,
    });

    // Landscape parameters (R_v1 roughness only / C=0 / K = alpha*C - beta*R)
    let lparams = LandscapeParams {
        alpha: 1.0,
        beta: 1.0,
        fmin: 20.0,
        fmax: 20_000.0,
    };

    let mut next_deadline = Instant::now();
    let hop_duration = Duration::from_secs_f32(hop as f32 / fs);

    loop {
	if exiting.load(Ordering::SeqCst) {
	    eprintln!("Stopping worker thread.");
	    break;
	}
	
        next_deadline += hop_duration;

        // 1) population → spectral amplitude A[k] (real, nonnegative)
        let amps = pop.project_spectrum(n_bins, fs, fft_size);

        // 2) synth: update phase & build complex spectrum, do iSTFT OLA → time chunk
        let time_chunk = synth.render_hop(&amps);
	println!("Generated chunk: {} samples, first={:.3}", 
		 time_chunk.len(), time_chunk.get(0).cloned().unwrap_or(0.0));
	
	// send out audio signal
	if let Some(prod) = audio_prod.as_mut() {
	    AudioOutput::push_samples(prod, &time_chunk);
	}

	if let Some(tx) = &wav_tx {
	    let _ = tx.try_send(time_chunk.clone());
	}
	
        // let mut buffered = AudioOutput::buffered_samples(audio_out);
        // let capacity = audio_out.capacity();
        // let target = capacity / 2; 
        // while buffered > target * 3 / 2 {
        //     eprintln!("AudioOutput buffer full. sleep 1ms.");
        //     std::thread::sleep(std::time::Duration::from_millis(1));
	//     buffered = AudioOutput::buffered_samples(audio_out);
        // }
	
        // 3) landscape from current spectrum (rough, consonance=0, K)
        let lframe: LandscapeFrame = crate::core::landscape::compute_landscape(&amps, fs, fft_size, &lparams);

        // 4) package for UI
        let ui_frame = UiFrame {
            wave: WaveChunk {
                fs: fs as f64,
                samples: time_chunk,
            },
            spec_hz: synth.bin_freqs_hz(),
            amps,
            landscape: lframe,
        };
        let _ = ui_tx.try_send(ui_frame);

        // // Simple timing (best-effort real-time-ish)
        let now = Instant::now();
        if now < next_deadline {
            std::thread::sleep(next_deadline - now);
        } else {
            next_deadline = now;
            trace!("worker overrun");
        }
    }
}
