use crate::core::landscape::LandscapeFrame;
use crate::core::stream::dorsal::DorsalMetrics;
use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::{Intent, IntentBoard};
use crate::life::predictive_spectrum::{PredKernelInputs, build_pred_kernel_inputs_from_intents};

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
    pub now: Tick,
    pub board: IntentBoard,
    pub next_intent_id: u64,
    pub percept_landscape: Option<LandscapeFrame>,
    pub dorsal_metrics: Option<DorsalMetrics>,
}

impl WorldModel {
    pub fn new(time: Timebase) -> Self {
        let retention_past = time.sec_to_tick(2.0);
        let horizon_future = time.sec_to_tick(8.0);
        Self {
            time,
            now: 0,
            board: IntentBoard::new(retention_past, horizon_future),
            next_intent_id: 0,
            percept_landscape: None,
            dorsal_metrics: None,
        }
    }

    pub fn advance_to(&mut self, now_tick: Tick) {
        self.now = now_tick;
        self.board.prune(now_tick);
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
        let Some(landscape) = self.percept_landscape.as_ref() else {
            debug_assert!(false, "pred_kernel_inputs_at requires percept_landscape");
            return PredKernelInputs {
                eval_tick,
                pred_env_scan: Vec::new(),
                pred_den_scan: Vec::new(),
            };
        };
        let past = self.board.retention_past;
        let future = self.board.horizon_future;
        let intents = self.board.snapshot(eval_tick, past, future);
        build_pred_kernel_inputs_from_intents(&landscape.space, &intents, eval_tick)
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
