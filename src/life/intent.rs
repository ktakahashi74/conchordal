use std::collections::VecDeque;
use std::ops::Range;

use crate::core::timebase::Tick;

#[derive(Clone, Debug)]
pub struct Intent {
    pub source_id: u64,
    pub intent_id: u64,
    pub onset: Tick,
    pub duration: Tick,
    pub freq_hz: f32,
    pub amp: f32,
    pub tag: Option<String>,
    pub confidence: f32,
}

pub struct IntentBoard {
    pub retention_past: Tick,
    pub horizon_future: Tick,
    intents: VecDeque<Intent>,
}

impl IntentBoard {
    pub fn new(retention_past: Tick, horizon_future: Tick) -> Self {
        Self {
            retention_past,
            horizon_future,
            intents: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.intents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.intents.is_empty()
    }

    pub fn publish(&mut self, intent: Intent) {
        if let Some(last) = self.intents.back() {
            debug_assert!(
                last.onset <= intent.onset,
                "IntentBoard expects non-decreasing onset order: last={} new={}",
                last.onset,
                intent.onset
            );
        }
        self.intents.push_back(intent);
    }

    pub fn prune(&mut self, now_tick: Tick) {
        let min_keep_end = now_tick.saturating_sub(self.retention_past);
        while let Some(front) = self.intents.front() {
            let end = front.onset.saturating_add(front.duration);
            if end < min_keep_end {
                self.intents.pop_front();
            } else {
                break;
            }
        }

        let max_keep_onset = now_tick.saturating_add(self.horizon_future);
        while let Some(back) = self.intents.back() {
            if back.onset > max_keep_onset {
                self.intents.pop_back();
            } else {
                break;
            }
        }
    }

    pub fn query_range<'a>(&'a self, range: Range<Tick>) -> impl Iterator<Item = &'a Intent> + 'a {
        let start = range.start;
        let end = range.end;
        self.intents
            .iter()
            .take_while(move |intent| intent.onset < end)
            .filter(move |intent| {
                let intent_end = intent.onset.saturating_add(intent.duration);
                intent.onset < end && intent_end > start
            })
    }

    pub fn snapshot(&self, now_tick: Tick, past: Tick, future: Tick) -> Vec<Intent> {
        let t0 = now_tick.saturating_sub(past);
        let t1 = now_tick.saturating_add(future);
        self.query_range(t0..t1).cloned().collect()
    }
}
