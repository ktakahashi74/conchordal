use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use tracing::debug;

use crate::audio::output::AudioOutput;
use crate::config::AppConfig;
use crate::ui::viewdata::{DorsalFrame, PlaybackState, UiFrame};

pub use crate::runtime::{
    compile_scenario_from_script, run_compile_only, run_headless, run_render, validate_scenario,
    validate_scenario_script_extension,
};

pub struct App {
    ui_frame_rx: Receiver<UiFrame>,
    _ctrl_tx: Sender<()>, // placeholder
    last_frame: UiFrame,
    ui_queue: VecDeque<UiFrame>,
    visual_delay_frames: usize,
    _audio: Option<AudioOutput>,
    audio_init_error: Option<String>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    analysis_handle: Option<std::thread::JoinHandle<()>>,
    exiting: Arc<AtomicBool>,
    rhythm_history: VecDeque<(f64, crate::core::modulation::NeuralRhythms)>,
    dorsal_history: VecDeque<(f64, DorsalFrame)>,
    start_flag: Arc<AtomicBool>,
    level_history: VecDeque<(std::time::Instant, [f32; 2])>,
    show_raw_nsgt_power: bool,
    has_ui_frame: bool,
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

        let rt = crate::runtime::init_runtime(args, config, stop_flag.clone());

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
            worker_handle: rt.worker_handle,
            analysis_handle: rt.analysis_handle,
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

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.viewport().close_requested()) {
            self.exiting.store(true, Ordering::SeqCst);
        }

        if self.exiting.load(Ordering::SeqCst) {
            debug!("SIGINT received: closing window.");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

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
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        debug!("App drop. Finalizing..");
        self.exiting.store(true, Ordering::SeqCst);

        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.analysis_handle.take() {
            let _ = handle.join();
        }
    }
}
