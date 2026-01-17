use std::collections::VecDeque;
use std::ops::Range;

use crate::core::timebase::Tick;

#[derive(Clone, Debug)]
pub struct NoteEvent {
    pub source_id: u64,
    pub note_id: u64,
    pub onset: Tick,
    pub duration: Tick,
    pub freq_hz: f32,
    pub amp: f32,
    pub tag: Option<String>,
    pub confidence: f32,
    pub body: Option<BodySnapshot>,
    pub articulation: Option<crate::life::individual::ArticulationWrapper>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BodySnapshot {
    pub kind: String,
    pub amp_scale: f32,
    pub brightness: f32,
    pub noise_mix: f32,
}

/// Schedule of committed notes (definitive timeline).
pub struct NoteBoard {
    pub retention_past: Tick,
    pub horizon_future: Tick,
    notes: VecDeque<NoteEvent>,
}

impl NoteBoard {
    pub fn new(retention_past: Tick, horizon_future: Tick) -> Self {
        Self {
            retention_past,
            horizon_future,
            notes: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.notes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }

    pub fn publish(&mut self, note: NoteEvent) {
        if self.notes.is_empty() {
            self.notes.push_back(note);
            return;
        }

        let push_back = matches!(
            self.notes.back(),
            Some(last) if last.onset <= note.onset
        );
        if push_back {
            self.notes.push_back(note);
        } else if matches!(
            self.notes.front(),
            Some(first) if note.onset < first.onset
        ) {
            self.notes.push_front(note);
        } else {
            let insert_at = self
                .notes
                .iter()
                .position(|existing| existing.onset > note.onset)
                .unwrap_or(self.notes.len());
            self.notes.insert(insert_at, note);
        }
        debug_assert!(self.is_sorted_by_onset());
    }

    fn is_sorted_by_onset(&self) -> bool {
        self.notes
            .iter()
            .zip(self.notes.iter().skip(1))
            .all(|(a, b)| a.onset <= b.onset)
    }

    pub fn prune(&mut self, now_tick: Tick) {
        let min_keep_end = now_tick.saturating_sub(self.retention_past);
        self.notes.retain(|note| {
            let end = note.onset.saturating_add(note.duration);
            end >= min_keep_end
        });

        let max_keep_onset = now_tick.saturating_add(self.horizon_future);
        while let Some(back) = self.notes.back() {
            if back.onset > max_keep_onset {
                self.notes.pop_back();
            } else {
                break;
            }
        }
    }

    pub fn query_range<'a>(
        &'a self,
        range: Range<Tick>,
    ) -> impl Iterator<Item = &'a NoteEvent> + 'a {
        let start = range.start;
        let end = range.end;
        self.notes
            .iter()
            .take_while(move |note| note.onset < end)
            .filter(move |note| {
                let note_end = note.onset.saturating_add(note.duration);
                note.onset < end && note_end > start
            })
    }

    pub fn snapshot(&self, now_tick: Tick, past: Tick, future: Tick) -> Vec<NoteEvent> {
        let t0 = now_tick.saturating_sub(past);
        let t1 = now_tick.saturating_add(future);
        self.query_range(t0..t1).cloned().collect()
    }

    pub fn remove_onset_from(&mut self, cutoff: Tick) {
        self.notes.retain(|note| note.onset < cutoff);
    }
}
