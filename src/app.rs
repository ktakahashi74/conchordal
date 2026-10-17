use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use crossbeam_channel::Receiver;
use tracing::debug;

use crate::audio::output::AudioOutput;
use crate::config::AppConfig;
use crate::viewdata::{DorsalFrame, PlaybackState, UiFrame};

pub use crate::runtime::{
    compile_scenario_from_script, run_compile_only, run_headless, run_render,
};

pub struct App {
    ui_frame_rx: Receiver<UiFrame>,
    last_frame: UiFrame,
    _audio: Option<AudioOutput>,
    runtime_error: Option<String>,
    worker_handle: Option<std::thread::JoinHandle<Result<(), String>>>,
    analysis_handle: Option<std::thread::JoinHandle<()>>,
    listener_analysis_handle: Option<std::thread::JoinHandle<()>>,
    exiting: Arc<AtomicBool>,
    drop_started: Arc<AtomicBool>,
    runtime_failed: Arc<AtomicBool>,
    listener_rhythm_history: VecDeque<(f64, crate::core::meter::MeterState)>,
    listener_attention_history: VecDeque<(f64, DorsalFrame)>,
    start_flag: Arc<AtomicBool>,
    level_history: VecDeque<(std::time::Instant, [f32; 2])>,
    show_raw_nsgt_power: bool,
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        args: crate::cli::Args,
        config: AppConfig,
        stop_flag: Arc<AtomicBool>,
        runtime_failed: Arc<AtomicBool>,
    ) -> Self {
        let drop_started = Arc::new(AtomicBool::new(false));
        let drop_watch = Arc::clone(&drop_started);
        let repaint_ctx = cc.egui_ctx.clone();
        let stop_flag_watch = stop_flag.clone();
        std::thread::spawn(move || {
            watch_for_close(
                &stop_flag_watch,
                &drop_watch,
                || {
                    repaint_ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    repaint_ctx.request_repaint();
                },
                || std::process::exit(1),
            );
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
            ui_frame_rx: rt.ui_frame_rx.expect("GUI frame receiver"),
            last_frame: UiFrame::default(),
            _audio: rt.audio_out,
            runtime_error: rt
                .audio_init_error
                .map(|err| format!("Audio init failed: {err}")),
            worker_handle: rt.worker_handle,
            analysis_handle: rt.analysis_handle,
            listener_analysis_handle: rt.listener_analysis_handle,
            exiting: stop_flag,
            drop_started,
            runtime_failed,
            listener_rhythm_history: VecDeque::with_capacity(4096),
            listener_attention_history: VecDeque::with_capacity(4096),
            start_flag: rt.start_flag,
            level_history: VecDeque::with_capacity(256),
            show_raw_nsgt_power: false,
        }
    }

    fn apply_ui_frame(&mut self, frame: UiFrame) {
        self.last_frame = frame;

        let t = self.last_frame.meta.time_sec as f64;
        if self.last_frame.meta.playback_state != PlaybackState::Finished
            && self.last_frame.listener.has_fast_state
        {
            self.listener_rhythm_history
                .push_back((t, self.last_frame.listener.meter));
            self.listener_attention_history
                .push_back((t, self.last_frame.listener.attention));
        }

        let t_last = self
            .listener_rhythm_history
            .back()
            .map(|(t, _)| *t)
            .or_else(|| self.listener_attention_history.back().map(|(t, _)| *t));
        if let Some(t_last) = t_last {
            while self
                .listener_rhythm_history
                .front()
                .is_some_and(|(time, _)| *time < t_last - 5.0)
            {
                self.listener_rhythm_history.pop_front();
            }
            while self
                .listener_attention_history
                .front()
                .is_some_and(|(time, _)| *time < t_last - 5.0)
            {
                self.listener_attention_history.pop_front();
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
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.viewport().close_requested()) {
            self.exiting.store(true, Ordering::SeqCst);
        }

        if self
            .worker_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
        {
            let handle = self.worker_handle.take().expect("finished worker handle");
            if let Err(err) =
                crate::runtime::join_thread("worker", handle).and_then(|result| result)
            {
                tracing::error!("{err}");
                self.runtime_failed.store(true, Ordering::SeqCst);
                self.runtime_error = Some(err);
            }
        }

        if self.exiting.load(Ordering::SeqCst) {
            debug!("SIGINT received: closing window.");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        while let Ok(frame) = self.ui_frame_rx.try_recv() {
            self.apply_ui_frame(frame);
        }

        ctx.request_repaint_after(Duration::from_millis(16));
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        crate::ui::windows::main_window(
            ui,
            &self.last_frame,
            &self.listener_rhythm_history,
            &self.listener_attention_history,
            self.runtime_error.as_deref(),
            &self.exiting,
            &self.start_flag,
            &mut self.show_raw_nsgt_power,
        );
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.drop_started.store(true, Ordering::SeqCst);
        debug!("App drop. Finalizing..");
        self.exiting.store(true, Ordering::SeqCst);

        let mut failures = Vec::new();
        if let Some(handle) = self.worker_handle.take()
            && let Err(err) =
                crate::runtime::join_thread("worker", handle).and_then(|result| result)
        {
            failures.push(err);
        }
        for (name, handle) in [
            ("analysis", self.analysis_handle.take()),
            ("listener-analysis", self.listener_analysis_handle.take()),
        ] {
            if let Some(handle) = handle
                && let Err(err) = crate::runtime::join_thread(name, handle)
            {
                failures.push(err);
            }
        }
        for err in failures {
            tracing::error!("{err}");
            self.runtime_failed.store(true, Ordering::SeqCst);
        }
    }
}

fn watch_for_close(
    stop_flag: &AtomicBool,
    drop_started: &AtomicBool,
    close: impl FnOnce(),
    force_exit: impl FnOnce(),
) {
    while !stop_flag.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(50));
    }
    close();
    std::thread::sleep(Duration::from_millis(200));
    if !drop_started.load(Ordering::SeqCst) {
        force_exit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_watchdog_only_forces_exit_when_drop_has_not_started() {
        for dropping in [false, true] {
            let stop = AtomicBool::new(true);
            let drop_started = AtomicBool::new(false);
            let forced = AtomicBool::new(false);
            let closed = AtomicBool::new(false);
            watch_for_close(
                &stop,
                &drop_started,
                || {
                    closed.store(true, Ordering::SeqCst);
                    drop_started.store(dropping, Ordering::SeqCst);
                },
                || forced.store(true, Ordering::SeqCst),
            );
            assert!(closed.load(Ordering::SeqCst));
            assert_eq!(forced.load(Ordering::SeqCst), !dropping);
        }
    }

    #[test]
    fn app_drop_records_worker_errors_and_panics_after_disabling_watchdog() {
        for panic in [false, true] {
            let drop_started = Arc::new(AtomicBool::new(false));
            let dropped = Arc::clone(&drop_started);
            let failed = Arc::new(AtomicBool::new(false));
            let stop = Arc::new(AtomicBool::new(false));
            let stop_worker = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                while !stop_worker.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                assert!(dropped.load(Ordering::SeqCst));
                assert!(!panic, "worker panic fixture");
                Err("report write failed".to_owned())
            });
            let (_, ui_frame_rx) = crossbeam_channel::bounded(1);
            drop(App {
                ui_frame_rx,
                last_frame: UiFrame::default(),
                _audio: None,
                runtime_error: None,
                worker_handle: Some(handle),
                analysis_handle: None,
                listener_analysis_handle: None,
                exiting: stop,
                drop_started,
                runtime_failed: Arc::clone(&failed),
                listener_rhythm_history: VecDeque::new(),
                listener_attention_history: VecDeque::new(),
                start_flag: Arc::new(AtomicBool::new(false)),
                level_history: VecDeque::new(),
                show_raw_nsgt_power: false,
            });
            assert!(failed.load(Ordering::SeqCst));
        }
    }
}
