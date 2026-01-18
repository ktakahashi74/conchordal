use std::sync::Arc;

use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::{Log2Space, sample_scan_linear_log2};
use crate::core::modulation::NeuralRhythms;
use crate::core::stream::dorsal::DorsalMetrics;
use crate::core::timebase::{Tick, Timebase};
use crate::life::gate_clock::next_gate_tick;
use crate::life::note_event::{NoteBoard, NoteEvent};
use tracing::debug;

const PRED_DAMPING: f32 = 0.5;

#[derive(Clone, Debug)]
pub struct NoteView {
    pub onset_tick: Tick,
    pub dur_tick: Tick,
    pub freq_hz: f32,
    pub amp: f32,
    pub source_id: u64,
    pub tag: Option<String>,
}

impl From<&NoteEvent> for NoteView {
    fn from(note: &NoteEvent) -> Self {
        Self {
            onset_tick: note.onset,
            dur_tick: note.duration,
            freq_hz: note.freq_hz,
            amp: note.amp,
            source_id: note.source_id,
            tag: note.tag.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorldView {
    pub now_tick: Tick,
    pub fs: f32,
    pub past_ticks: Tick,
    pub future_ticks: Tick,
    pub notes: Vec<NoteView>,
    pub next_gate_tick_est: Option<Tick>,
    pub next_gate_sec_est: Option<f64>,
    pub planned_next_live: Vec<PlannedNoteView>,
    pub planned_last_committed: Vec<PlannedNoteView>,
    pub planned_last_gate_tick: Option<Tick>,
}

impl Default for WorldView {
    fn default() -> Self {
        Self {
            now_tick: 0,
            fs: 0.0,
            past_ticks: 0,
            future_ticks: 0,
            notes: Vec::new(),
            next_gate_tick_est: None,
            next_gate_sec_est: None,
            planned_next_live: Vec::new(),
            planned_last_committed: Vec::new(),
            planned_last_gate_tick: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlannedNoteView {
    pub source_id: u64,
    pub freq_hz: f32,
    pub amp: f32,
    pub confidence: f32,
    pub tag: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct TerrainPredictor {
    prev_obs: Option<(Tick, Arc<[f32]>)>,
    last_obs: Option<(Tick, Arc<[f32]>)>,
    cache_pred_next_gate: Option<(Tick, Arc<[f32]>)>,
}

impl TerrainPredictor {
    fn reset(&mut self) {
        self.prev_obs = None;
        self.last_obs = None;
        self.cache_pred_next_gate = None;
    }

    fn observe_consonance01(&mut self, tick: Tick, scan: Arc<[f32]>, space: &Log2Space) {
        space.assert_scan_len_named(&scan, "perc_c_state01_scan");
        self.prev_obs = self.last_obs.take();
        self.last_obs = Some((tick, scan));
        self.cache_pred_next_gate = None;
    }

    fn predict_consonance01_at(&self, tick: Tick, space: &Log2Space) -> Option<Arc<[f32]>> {
        let (last_tick, last_scan) = self.last_obs.as_ref()?;
        space.assert_scan_len_named(last_scan, "pred_c_state01_scan");
        if let Some((prev_tick, prev_scan)) = self.prev_obs.as_ref() {
            space.assert_scan_len_named(prev_scan, "pred_c_state01_scan_prev");
            if prev_scan.len() != last_scan.len() {
                return Some(Arc::clone(last_scan));
            }
            let dt = last_tick.saturating_sub(*prev_tick).max(1) as f32;
            let a = tick.saturating_sub(*last_tick) as f32;
            let scale = a / dt;
            let mut out = Vec::with_capacity(last_scan.len());
            for (&last, &prev) in last_scan.iter().zip(prev_scan.iter()) {
                let pred = last + PRED_DAMPING * (last - prev) * scale;
                let val = if pred.is_finite() {
                    pred.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                out.push(val);
            }
            Some(Arc::from(out))
        } else {
            Some(Arc::clone(last_scan))
        }
    }
}

pub struct WorldModel {
    pub time: Timebase,
    pub space: Log2Space,
    pub now: Tick,
    pub board: NoteBoard,
    pub next_note_id: u64,
    pub percept_landscape: Option<LandscapeFrame>,
    pub dorsal_metrics: Option<DorsalMetrics>,
    pub next_gate_tick_est: Option<Tick>,
    terrain_predictor: TerrainPredictor,
}

impl WorldModel {
    pub fn new(time: Timebase, space: Log2Space) -> Self {
        let retention_past = time.sec_to_tick(2.0);
        let horizon_future = time.sec_to_tick(8.0);
        Self {
            time,
            space,
            now: 0,
            board: NoteBoard::new(retention_past, horizon_future),
            next_note_id: 0,
            percept_landscape: None,
            dorsal_metrics: None,
            next_gate_tick_est: None,
            terrain_predictor: TerrainPredictor::default(),
        }
    }

    pub fn advance_to(&mut self, now_tick: Tick) {
        self.now = now_tick;
        self.board.prune(now_tick);
    }

    pub fn set_space(&mut self, space: Log2Space) {
        self.space = space;
        self.terrain_predictor.reset();
    }

    pub fn ui_view(&self) -> WorldView {
        let past = self.board.retention_past;
        let future = self.board.horizon_future;
        let notes = self
            .board
            .snapshot(self.now, past, future)
            .iter()
            .map(NoteView::from)
            .collect();
        let next_gate_sec_est = self.next_gate_tick_est.and_then(|tick| {
            if self.time.fs.is_finite() && self.time.fs > 0.0 {
                Some(tick as f64 / self.time.fs as f64)
            } else {
                None
            }
        });
        WorldView {
            now_tick: self.now,
            fs: self.time.fs,
            past_ticks: past,
            future_ticks: future,
            notes,
            next_gate_tick_est: self.next_gate_tick_est,
            next_gate_sec_est,
            planned_next_live: Vec::new(),
            planned_last_committed: Vec::new(),
            planned_last_gate_tick: None,
        }
    }

    pub fn update_gate_from_rhythm(&mut self, now_tick: Tick, rhythm: &NeuralRhythms) {
        self.next_gate_tick_est = next_gate_tick(now_tick, self.time.fs, rhythm.theta, 0.0);
        if self.next_gate_tick_est.is_none() && cfg!(debug_assertions) {
            debug!(
                target: "gate",
                "next_gate_tick_est None: now_tick={} fs={:.3} theta_hz={:.3} theta_phase={:.3}",
                now_tick,
                self.time.fs,
                rhythm.theta.freq_hz,
                rhythm.theta.phase
            );
        }
    }

    pub fn observe_consonance01(&mut self, tick: Tick, scan: Arc<[f32]>) {
        // NSGT is right-aligned, so observations are stamped at frame_end_tick.
        self.terrain_predictor
            .observe_consonance01(tick, scan, &self.space);
    }

    pub fn predict_consonance01_at(&self, tick: Tick) -> Option<Arc<[f32]>> {
        self.terrain_predictor
            .predict_consonance01_at(tick, &self.space)
    }

    pub fn predict_consonance01_next_gate(&mut self) -> Option<(Tick, Arc<[f32]>)> {
        let eval_tick = self.next_gate_tick_est?;
        if let Some((_, cached_scan)) = self
            .terrain_predictor
            .cache_pred_next_gate
            .as_ref()
            .filter(|(cached_tick, _)| *cached_tick == eval_tick)
        {
            return Some((eval_tick, Arc::clone(cached_scan)));
        }
        let scan = self.predict_consonance01_at(eval_tick)?;
        self.terrain_predictor.cache_pred_next_gate = Some((eval_tick, Arc::clone(&scan)));
        Some((eval_tick, scan))
    }

    pub fn sample_scan01(&self, scan: &[f32], freq_hz: f32) -> f32 {
        self.space
            .assert_scan_len_named(scan, "pred_c_state01_scan");
        let raw = sample_scan_linear_log2(&self.space, scan, freq_hz);
        if raw.is_finite() {
            raw.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn apply_action(&mut self, action: &crate::life::scenario::Action) {
        if let crate::life::scenario::Action::PostNote {
            source_id,
            onset_sec,
            duration_sec,
            freq_hz,
            amp,
            tag,
            confidence,
        } = action
        {
            let onset_tick = self.time.sec_to_tick(onset_sec.max(0.0));
            let mut dur_tick = self.time.sec_to_tick(duration_sec.max(0.0));
            if dur_tick == 0 && *duration_sec > 0.0 {
                dur_tick = 1;
            }
            let note = NoteEvent {
                source_id: *source_id,
                note_id: self.next_note_id,
                onset: onset_tick,
                duration: dur_tick,
                freq_hz: *freq_hz,
                amp: *amp,
                tag: tag.clone(),
                confidence: *confidence,
                body: None,
                articulation: None,
            };
            self.next_note_id = self.next_note_id.wrapping_add(1);
            self.board.publish(note);
        }
    }
}
