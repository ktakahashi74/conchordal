use std::sync::Arc;

use crate::core::landscape::{LandscapeFrame, LandscapeParams};
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::stream::dorsal::DorsalMetrics;
use crate::core::timebase::{Tick, Timebase};
use crate::life::gate_clock::next_gate_tick;
use crate::life::intent::{Intent, IntentBoard};
use crate::life::plan::{PlanBoard, PlannedIntent};
use crate::life::predictive_spectrum::{
    PredKernelInputs, PredTerrain, build_pred_kernel_inputs_from_intents,
    build_pred_terrain_from_intents,
};
use tracing::debug;

#[derive(Clone, Debug)]
pub struct IntentView {
    pub onset_tick: Tick,
    pub dur_tick: Tick,
    pub freq_hz: f32,
    pub amp: f32,
    pub source_id: u64,
    pub tag: Option<String>,
}

impl From<&Intent> for IntentView {
    fn from(intent: &Intent) -> Self {
        Self {
            onset_tick: intent.onset,
            dur_tick: intent.duration,
            freq_hz: intent.freq_hz,
            amp: intent.amp,
            source_id: intent.source_id,
            tag: intent.tag.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorldView {
    pub now_tick: Tick,
    pub fs: f32,
    pub past_ticks: Tick,
    pub future_ticks: Tick,
    pub intents: Vec<IntentView>,
    pub next_gate_tick_est: Option<Tick>,
    pub next_gate_sec_est: Option<f64>,
    pub planned_next: Vec<PlannedIntent>,
}

impl Default for WorldView {
    fn default() -> Self {
        Self {
            now_tick: 0,
            fs: 0.0,
            past_ticks: 0,
            future_ticks: 0,
            intents: Vec::new(),
            next_gate_tick_est: None,
            next_gate_sec_est: None,
            planned_next: Vec::new(),
        }
    }
}

pub struct WorldModel {
    pub time: Timebase,
    pub space: Log2Space,
    pub now: Tick,
    pub board: IntentBoard,
    pub next_intent_id: u64,
    pub percept_landscape: Option<LandscapeFrame>,
    pub dorsal_metrics: Option<DorsalMetrics>,
    pub pred_params: Option<LandscapeParams>,
    pub plan_board: PlanBoard,
    pub next_gate_tick_est: Option<Tick>,
    pub last_committed_gate_tick: Option<Tick>,
    #[allow(dead_code)]
    pub phi_epsilon: f32,
}

impl WorldModel {
    pub fn new(time: Timebase, space: Log2Space) -> Self {
        let retention_past = time.sec_to_tick(2.0);
        let horizon_future = time.sec_to_tick(8.0);
        Self {
            time,
            space,
            now: 0,
            board: IntentBoard::new(retention_past, horizon_future),
            next_intent_id: 0,
            percept_landscape: None,
            dorsal_metrics: None,
            pred_params: None,
            plan_board: PlanBoard::new(),
            next_gate_tick_est: None,
            last_committed_gate_tick: None,
            phi_epsilon: 1e-6,
        }
    }

    pub fn advance_to(&mut self, now_tick: Tick) {
        self.now = now_tick;
        self.board.prune(now_tick);
    }

    pub fn set_pred_params(&mut self, params: LandscapeParams) {
        self.pred_params = Some(params);
    }

    pub fn set_space(&mut self, space: Log2Space) {
        self.space = space;
    }

    pub fn ui_view(&self) -> WorldView {
        let past = self.board.retention_past;
        let future = self.board.horizon_future;
        let intents = self
            .board
            .snapshot(self.now, past, future)
            .iter()
            .map(IntentView::from)
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
            intents,
            next_gate_tick_est: self.next_gate_tick_est,
            next_gate_sec_est,
            planned_next: self.plan_board.snapshot_next(),
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

    pub fn commit_plans_if_due(&mut self, now: Tick, frame_end: Tick) {
        let gate_tick = match self.next_gate_tick_est {
            Some(tick) => tick,
            None => return,
        };
        if gate_tick < now || gate_tick >= frame_end {
            return;
        }
        if self.last_committed_gate_tick == Some(gate_tick) {
            return;
        }

        let planned = self.plan_board.snapshot_next();
        for p in planned {
            let intent = Intent {
                source_id: p.source_id,
                intent_id: self.next_intent_id,
                onset: gate_tick,
                duration: p.duration,
                freq_hz: p.freq_hz,
                amp: p.amp,
                tag: p.tag.clone(),
                confidence: p.confidence,
                body: p.body.clone(),
            };
            self.next_intent_id = self.next_intent_id.wrapping_add(1);
            self.board.publish(intent);
        }
        self.plan_board.clear_next();
        self.last_committed_gate_tick = Some(gate_tick);
    }

    pub fn pred_c_next_gate(&self, params: &LandscapeParams) -> Option<Arc<[f32]>> {
        let eval_tick = self.next_gate_tick_est?;
        let past = self
            .board
            .retention_past
            .saturating_add(self.now.saturating_sub(eval_tick));
        let future = self
            .board
            .horizon_future
            .saturating_add(eval_tick.saturating_sub(self.now));
        let mut intents = self.board.snapshot(self.now, past, future);
        for planned in self.plan_board.snapshot_next() {
            intents.push(Intent {
                source_id: planned.source_id,
                intent_id: planned.plan_id,
                onset: eval_tick,
                duration: planned.duration,
                freq_hz: planned.freq_hz,
                amp: planned.amp,
                tag: planned.tag.clone(),
                confidence: planned.confidence,
                body: planned.body.clone(),
            });
        }
        let terrain = build_pred_terrain_from_intents(&self.space, params, &intents, eval_tick);
        if terrain.pred_c_statepm1_scan.is_empty() {
            if cfg!(debug_assertions) {
                debug!(target: "pred_c", "pred_c_next_gate empty scan");
            }
            return None;
        }
        Some(Arc::from(terrain.pred_c_statepm1_scan))
    }

    pub fn pred_kernel_inputs_at(&self, eval_tick: Tick) -> PredKernelInputs {
        let past = self
            .board
            .retention_past
            .saturating_add(self.now.saturating_sub(eval_tick));
        let future = self
            .board
            .horizon_future
            .saturating_add(eval_tick.saturating_sub(self.now));
        let intents = self.board.snapshot(self.now, past, future);
        build_pred_kernel_inputs_from_intents(&self.space, &intents, eval_tick)
    }

    pub fn pred_terrain_at(&self, eval_tick: Tick) -> Option<PredTerrain> {
        let params = match self.pred_params.as_ref() {
            Some(params) => params,
            None => {
                if cfg!(debug_assertions) {
                    debug!(target: "pred_c", "pred_terrain missing pred_params");
                }
                return None;
            }
        };
        let past = self
            .board
            .retention_past
            .saturating_add(self.now.saturating_sub(eval_tick));
        let future = self
            .board
            .horizon_future
            .saturating_add(eval_tick.saturating_sub(self.now));
        let intents = self.board.snapshot(self.now, past, future);
        let terrain = build_pred_terrain_from_intents(&self.space, params, &intents, eval_tick);
        if cfg!(debug_assertions) && terrain.pred_c_statepm1_scan.is_empty() {
            debug!(target: "pred_c", "pred_terrain empty scan");
        }
        Some(terrain)
    }

    pub fn pred_c_statepm1_scan_at(&self, eval_tick: Tick) -> Vec<f32> {
        let scan = self
            .pred_terrain_at(eval_tick)
            .map(|terrain| terrain.pred_c_statepm1_scan)
            .unwrap_or_default();
        if cfg!(debug_assertions) {
            let non_finite = scan.iter().filter(|v| !v.is_finite()).count();
            if non_finite > 0 {
                debug!(
                    target: "pred_c",
                    "pred_c_scan non_finite={} len={} eval_tick={}",
                    non_finite,
                    scan.len(),
                    eval_tick
                );
            }
        }
        scan
    }

    pub fn apply_action(&mut self, action: &crate::life::scenario::Action) {
        if let crate::life::scenario::Action::PostIntent {
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
            let intent = Intent {
                source_id: *source_id,
                intent_id: self.next_intent_id,
                onset: onset_tick,
                duration: dur_tick,
                freq_hz: *freq_hz,
                amp: *amp,
                tag: tag.clone(),
                confidence: *confidence,
                body: None,
            };
            self.next_intent_id = self.next_intent_id.wrapping_add(1);
            self.board.publish(intent);
        }
    }
}
