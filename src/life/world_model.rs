use crate::core::landscape::{LandscapeFrame, LandscapeParams};
use crate::core::log2space::Log2Space;
use crate::core::stream::dorsal::DorsalMetrics;
use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::{Intent, IntentBoard};
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
}

impl Default for WorldView {
    fn default() -> Self {
        Self {
            now_tick: 0,
            fs: 0.0,
            past_ticks: 0,
            future_ticks: 0,
            intents: Vec::new(),
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
        WorldView {
            now_tick: self.now,
            fs: self.time.fs,
            past_ticks: past,
            future_ticks: future,
            intents,
        }
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
