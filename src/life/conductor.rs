use std::collections::VecDeque;

use tracing::debug;

use super::population::Population;
use super::scenario::{Action, Episode, Scenario};
use crate::core::landscape::LandscapeFrame;

#[derive(Debug, Clone)]
struct QueuedEvent {
    time: f32,
    actions: Vec<Action>,
}

#[derive(Debug)]
pub struct Conductor {
    event_queue: VecDeque<QueuedEvent>,
}

impl Conductor {
    pub fn from_scenario(s: Scenario) -> Self {
        let mut events: Vec<QueuedEvent> = s
            .episodes
            .into_iter()
            .flat_map(|ep| flatten_episode(ep))
            .collect();

        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            event_queue: events.into(),
        }
    }

    /// Apply any events scheduled up to and including current time.
    pub fn dispatch_until(
        &mut self,
        time_sec: f32,
        _current_frame: u64,
        landscape: &LandscapeFrame,
        population: &mut Population,
    ) {
        while let Some(ev) = self.event_queue.front() {
            if ev.time > time_sec {
                break;
            }

            let ev = self.event_queue.pop_front().expect("front exists");
            debug!("Dispatching event at t={}", ev.time);
            for action in ev.actions {
                population.apply_action(action, landscape);
            }
        }
    }

    pub fn is_done(&self) -> bool {
        self.event_queue.is_empty()
    }
}

fn flatten_episode(ep: Episode) -> impl Iterator<Item = QueuedEvent> {
    ep.events.into_iter().map(move |ev| QueuedEvent {
        time: ep.start_time + ev.time,
        actions: ev.actions,
    })
}
