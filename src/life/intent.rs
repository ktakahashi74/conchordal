use std::collections::VecDeque;
use std::ops::Range;

use crate::core::timebase::Tick;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum IntentKind {
    #[default]
    Normal,
    BirthOnce,
}

#[derive(Clone, Debug)]
pub struct Intent {
    pub source_id: u64,
    pub intent_id: u64,
    pub kind: IntentKind,
    pub onset: Tick,
    pub duration: Tick,
    pub freq_hz: f32,
    pub amp: f32,
    pub tag: Option<String>,
    pub confidence: f32,
    pub body: Option<BodySnapshot>,
    pub articulation: Option<crate::life::individual::ArticulationWrapper>,
}

#[derive(Clone, Debug)]
pub struct BodySnapshot {
    pub kind: String,
    pub amp_scale: f32,
    pub brightness: f32,
    pub noise_mix: f32,
}

/// Schedule of committed intents (definitive timeline).
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
        if self.intents.is_empty() {
            self.intents.push_back(intent);
            return;
        }

        let push_back = matches!(
            self.intents.back(),
            Some(last) if last.onset <= intent.onset
        );
        if push_back {
            self.intents.push_back(intent);
        } else if matches!(
            self.intents.front(),
            Some(first) if intent.onset < first.onset
        ) {
            self.intents.push_front(intent);
        } else {
            let insert_at = self
                .intents
                .iter()
                .position(|existing| existing.onset > intent.onset)
                .unwrap_or(self.intents.len());
            self.intents.insert(insert_at, intent);
        }
        debug_assert!(self.is_sorted_by_onset());
    }

    fn is_sorted_by_onset(&self) -> bool {
        self.intents
            .iter()
            .zip(self.intents.iter().skip(1))
            .all(|(a, b)| a.onset <= b.onset)
    }

    pub fn prune(&mut self, now_tick: Tick) {
        let min_keep_end = now_tick.saturating_sub(self.retention_past);
        self.intents.retain(|intent| {
            let end = intent.onset.saturating_add(intent.duration);
            end >= min_keep_end
        });

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

    pub fn remove_onset_from(&mut self, cutoff: Tick) {
        self.intents.retain(|intent| intent.onset < cutoff);
    }
}
