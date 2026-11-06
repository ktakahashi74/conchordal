use crate::{frame::*, ui};
use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use wasm_bindgen::prelude::*;

#[derive(Default)]
struct State {
    frame: UiFrame,
    rhythm: VecDeque<ListenerRhythmHop>,
    attention: VecDeque<(f64, DorsalFrame)>,
    harmony: VecDeque<(f64, Option<egui::Vec2>)>,
    peaks: VecDeque<(f64, [f32; 2])>,
    commands: Vec<&'static str>,
    raw: bool,
    exit: Arc<AtomicBool>,
    start: Arc<AtomicBool>,
    end: Arc<AtomicBool>,
    decode_ms: Vec<f64>,
    encode_ms: Vec<f64>,
    ui_ms: Vec<f64>,
    cpu_ms: Vec<f64>,
    arrivals: VecDeque<(f64, f64)>,
    shown: f64,
    measure: bool,
}
fn now() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

#[wasm_bindgen]
pub struct WebGui {
    runner: eframe::WebRunner,
    state: Rc<RefCell<State>>,
}
#[wasm_bindgen]
impl WebGui {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            runner: eframe::WebRunner::new(),
            state: Rc::default(),
        }
    }
    pub async fn start(&self, canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
        let state = self.state.clone();
        self.runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |_| Ok(Box::new(BrowserApp(state)))),
            )
            .await
    }
    /// Host passes UTF-8 bytes directly; JS must not round u64 identities through Number.
    pub fn push_snapshot(&self, bytes: &[u8]) -> Result<(), JsValue> {
        let start = now();
        let frame = UiFrame::decode(bytes).map_err(|e| JsValue::from_str(&e))?;
        let elapsed = now() - start;
        let mut s = self.state.borrow_mut();
        if s.measure {
            s.decode_ms.push(elapsed);
        }
        if frame.meta.time_sec < s.frame.meta.time_sec {
            s.rhythm.clear();
            s.attention.clear();
            s.peaks.clear();
            s.arrivals.clear();
            s.shown = 0.;
        }
        let t = frame.meta.time_sec;
        for hop in &frame.listener_rhythm {
            s.rhythm.push_back(*hop);
            s.attention.push_back((hop.time_sec, hop.attention));
        }
        while s
            .rhythm
            .front()
            .is_some_and(|h| h.time_sec < t - RHYTHM_HISTORY_SEC)
        {
            s.rhythm.pop_front();
        }
        while s
            .attention
            .front()
            .is_some_and(|h| h.0 < t - RHYTHM_HISTORY_SEC)
        {
            s.attention.pop_front();
        }
        ui::plots::observe_harmony(&mut s.harmony, &frame);
        s.peaks.push_back((t, frame.meta.channel_peak));
        while s.peaks.front().is_some_and(|h| h.0 < t - 1.) {
            s.peaks.pop_front();
        }
        s.frame = frame;
        s.frame.meta.window_peak = s
            .peaks
            .iter()
            .fold([0.0_f32; 2], |p, (_, v)| [p[0].max(v[0]), p[1].max(v[1])]);
        if let Some(hop) = s.frame.listener_rhythm.last().copied() {
            s.arrivals.push_back((hop.time_sec, now() / 1000.));
        }
        while s.arrivals.len() >= 2 && s.arrivals[1].0 <= t - RHYTHM_HISTORY_SEC {
            s.arrivals.pop_front();
        }
        Ok(())
    }
    /// Local codec benchmark only: no snapshot or waveform is exported to the host.
    pub fn benchmark_codec(&self, bytes: &[u8]) -> Result<(), JsValue> {
        let start = now();
        let decoded = UiFrame::decode(bytes).map_err(|e| JsValue::from_str(&e))?;
        let mid = now();
        let encoded = decoded
            .encode()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let end = now();
        let mut s = self.state.borrow_mut();
        if s.measure {
            s.encode_ms.push(end - mid);
        }
        if s.measure {
            s.decode_ms.push(mid - start);
        }
        std::hint::black_box(encoded);
        Ok(())
    }
    pub fn reset(&self) {
        let mut state = self.state.borrow_mut();
        let measure = state.measure;
        *state = State {
            measure,
            ..Default::default()
        };
    }
    pub fn set_playback_state(&self, state: &str) {
        self.state.borrow_mut().frame.meta.playback_state = match state {
            "initializing" | "playing" => PlaybackState::Playing,
            "ending" => PlaybackState::Ending,
            "ended" => PlaybackState::Finished,
            _ => PlaybackState::NotStarted,
        };
    }
    pub fn take_commands(&self) -> String {
        serde_json::to_string(&std::mem::take(&mut self.state.borrow_mut().commands)).unwrap()
    }
    pub fn set_measurement(&self, enabled: bool) {
        let mut s = self.state.borrow_mut();
        s.measure = enabled;
        s.decode_ms.clear();
        s.encode_ms.clear();
        s.ui_ms.clear();
        s.cpu_ms.clear();
    }
    pub fn layout_check(&self) -> String {
        let mut s = self.state.borrow_mut();
        let s = &mut *s;
        let mut output = egui::Context::default().run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1200., 900.),
                )),
                ..Default::default()
            },
            |root| {
                ui::windows::main_window(
                    root,
                    &s.frame,
                    &s.rhythm,
                    &s.attention,
                    &s.harmony,
                    s.frame.meta.time_sec,
                    None,
                    &s.exit,
                    &s.start,
                    &s.end,
                    &mut s.raw,
                )
            },
        );
        output.textures_delta.clear();
        let mut bottom = None;
        let mut right = None;
        let mut exit_y = None;
        for shape in output.shapes {
            if let egui::epaint::Shape::Text(text) = shape.shape {
                match text.galley.text() {
                    "Roughness / Harmonicity" => {
                        bottom = Some(text.pos.y + text.galley.size().y + 102.)
                    }
                    "Exit" => {
                        right = Some(text.pos.x + text.galley.size().x);
                        exit_y = Some(text.pos.y + text.galley.size().y / 2.);
                    }
                    _ => {}
                }
            }
        }
        serde_json::json!({"rh_bottom":bottom,"exit_right":right,"exit_y":exit_y,"pass":bottom.is_some_and(|v|v<=900.) && right.is_some_and(|v|v>1150.)}).to_string()
    }
    pub fn metrics(&self) -> String {
        let s = self.state.borrow();
        serde_json::json!({"decode_ms":s.decode_ms,"encode_ms":s.encode_ms,"ui_ms":s.ui_ms,"cpu_ms":s.cpu_ms}).to_string()
    }
    pub fn destroy(&self) {
        self.runner.destroy();
    }
}
struct BrowserApp(Rc<RefCell<State>>);
impl eframe::App for BrowserApp {
    fn logic(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
    fn ui(&mut self, root: &mut egui::Ui, frame: &mut eframe::Frame) {
        let began = now();
        let mut s = self.0.borrow_mut();
        let s = &mut *s;
        let running = matches!(
            s.frame.meta.playback_state,
            PlaybackState::Playing | PlaybackState::Ending
        );
        let latest = s.arrivals.back().map_or(s.frame.meta.time_sec, |a| a.0);
        let mut offset = f64::INFINITY;
        let mut gap = 0.0_f64;
        let mut previous = None;
        for &(t, w) in &s.arrivals {
            offset = offset.min(t - w);
            if let Some(p) = previous {
                gap = gap.max(w - p);
            }
            previous = Some(w);
        }
        let estimate = if running && !s.arrivals.is_empty() {
            (began / 1000. + offset - gap).clamp(latest - ui::windows::TIME_WINDOW_SEC, latest)
        } else {
            latest
        };
        s.shown = if !running || s.shown > latest {
            estimate
        } else {
            estimate.max(s.shown)
        };
        let display_time = s.shown;
        let start = display_time - ui::windows::TIME_WINDOW_SEC;
        // Keep the boundary sample, as the native display clock does.
        while s.attention.len() >= 2 && s.attention[1].0 <= start {
            s.attention.pop_front();
        }
        while s.arrivals.len() >= 2 && s.arrivals[1].0 <= start {
            s.arrivals.pop_front();
        }
        ui::windows::main_window(
            root,
            &s.frame,
            &s.rhythm,
            &s.attention,
            &s.harmony,
            display_time,
            None,
            &s.exit,
            &s.start,
            &s.end,
            &mut s.raw,
        );
        for (flag, command) in [(&s.exit, "exit"), (&s.start, "start"), (&s.end, "end")] {
            if flag.swap(false, Ordering::SeqCst) {
                s.commands.push(command);
            }
        }
        if s.measure {
            s.ui_ms.push(now() - began);
        }
        if let Some(cpu) = frame.info().cpu_usage
            && s.measure
        {
            s.cpu_ms.push(f64::from(cpu) * 1000.);
        }
    }
}
