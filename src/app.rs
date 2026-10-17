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
    shutdown_handle: Option<std::thread::JoinHandle<()>>,
    runtime_result_rx: Receiver<Result<(), String>>,
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
        let rt = crate::runtime::init_runtime(args, config, stop_flag.clone());
        let (runtime_result_tx, runtime_result_rx) = crossbeam_channel::bounded(1);
        let drop_watch = Arc::clone(&drop_started);
        let failed_watch = Arc::clone(&runtime_failed);
        let repaint_ctx = cc.egui_ctx.clone();
        let stop_flag_watch = stop_flag.clone();
        let shutdown_handle = std::thread::spawn(move || {
            watch_for_close(
                &stop_flag_watch,
                &drop_watch,
                rt.worker_handle.expect("GUI worker handle"),
                |worker| {
                    let result = crate::runtime::join_runtime(
                        Some(worker),
                        [
                            ("analysis", rt.analysis_handle),
                            ("listener-analysis", rt.listener_analysis_handle),
                        ],
                    );
                    if let Err(err) = &result {
                        tracing::error!("{err}");
                        failed_watch.store(true, Ordering::SeqCst);
                    }
                    let _ = runtime_result_tx.try_send(result);
                },
                || {
                    repaint_ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    repaint_ctx.request_repaint();
                },
                || std::process::exit(1),
            );
        });

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
            shutdown_handle: Some(shutdown_handle),
            runtime_result_rx,
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

        if let Ok(Err(err)) = self.runtime_result_rx.try_recv() {
            self.runtime_error = Some(err);
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

        if let Some(handle) = self.shutdown_handle.take()
            && let Err(err) = crate::runtime::join_thread("shutdown", handle)
        {
            tracing::error!("{err}");
            self.runtime_failed.store(true, Ordering::SeqCst);
        }
    }
}

fn watch_for_close(
    stop_flag: &AtomicBool,
    drop_started: &AtomicBool,
    worker: std::thread::JoinHandle<Result<(), String>>,
    cleanup: impl FnOnce(std::thread::JoinHandle<Result<(), String>>),
    close: impl FnOnce(),
    force_exit: impl FnOnce(),
) {
    while !stop_flag.load(Ordering::SeqCst) && !worker.is_finished() {
        std::thread::sleep(Duration::from_millis(50));
    }
    stop_flag.store(true, Ordering::SeqCst);
    close();
    // Cleanup must progress even when the GUI event loop never reaches Drop.
    cleanup(worker);
    std::thread::sleep(Duration::from_millis(200));
    if !drop_started.load(Ordering::SeqCst) {
        force_exit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_watchdog_waits_for_finalization_and_observer_join_before_forcing_exit() {
        for dropping in [false, true] {
            let stop = Arc::new(AtomicBool::new(true));
            let drop_started = Arc::new(AtomicBool::new(false));
            let forced = Arc::new(AtomicBool::new(false));
            let (finish_tx, finish_rx) = crossbeam_channel::bounded(0);
            let (observer_tx, observer_rx) = crossbeam_channel::bounded(0);
            let (close_tx, close_rx) = crossbeam_channel::bounded(0);
            let (joined_tx, joined_rx) = crossbeam_channel::bounded(0);
            let worker = std::thread::spawn(move || {
                finish_rx.recv().unwrap();
                Ok(())
            });
            let observer = std::thread::spawn(move || observer_rx.recv().unwrap());
            let forced_watch = Arc::clone(&forced);
            let dropped_watch = Arc::clone(&drop_started);
            let watchdog = std::thread::spawn(move || {
                watch_for_close(
                    &stop,
                    &dropped_watch,
                    worker,
                    |worker| {
                        crate::runtime::join_thread("worker", worker)
                            .unwrap()
                            .unwrap();
                        joined_tx.send(()).unwrap();
                        crate::runtime::join_thread("analysis", observer).unwrap();
                    },
                    || close_tx.send(()).unwrap(),
                    || forced_watch.store(true, Ordering::SeqCst),
                );
            });
            close_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            // The GUI still has not entered Drop after the old 200ms deadline.
            std::thread::sleep(Duration::from_millis(250));
            assert!(!forced.load(Ordering::SeqCst));
            finish_tx.send(()).unwrap();
            joined_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            std::thread::sleep(Duration::from_millis(250));
            assert!(!forced.load(Ordering::SeqCst));
            drop_started.store(dropping, Ordering::SeqCst);
            observer_tx.send(()).unwrap();
            watchdog.join().unwrap();
            assert_eq!(forced.load(Ordering::SeqCst), !dropping);
        }
    }

    #[test]
    fn app_drop_waits_for_shutdown_and_records_panics() {
        let failed = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop);
        let dropping = Arc::new(AtomicBool::new(false));
        let dropped = Arc::clone(&dropping);
        let shutdown = std::thread::spawn(move || {
            while !stop_worker.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            assert!(dropped.load(Ordering::SeqCst));
            panic!("shutdown panic fixture");
        });
        let (_, ui_frame_rx) = crossbeam_channel::bounded(1);
        let (_, runtime_result_rx) = crossbeam_channel::bounded(1);
        drop(App {
            ui_frame_rx,
            last_frame: UiFrame::default(),
            _audio: None,
            runtime_error: None,
            shutdown_handle: Some(shutdown),
            runtime_result_rx,
            exiting: stop,
            drop_started: dropping,
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
