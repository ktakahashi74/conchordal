use std::collections::VecDeque;

use tracing::info;

use super::population::Population;
use super::scenario::{Action, Episode, Scenario};
use crate::core::landscape::LandscapeFrame;

#[derive(Debug, Clone)]
pub struct QueuedEvent {
    pub time: f32,
    pub actions: Vec<Action>,
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

    pub fn from_events(events: Vec<QueuedEvent>) -> Self {
        let mut events = events;
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
            info!("[t={:.6}] Dispatching event", ev.time);
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
    let mut out: Vec<QueuedEvent> = Vec::new();
    for ev in ep.events {
        if let Some(rep) = &ev.repeat {
            for i in 0..rep.count {
                out.push(QueuedEvent {
                    time: ep.start_time + ev.time + i as f32 * rep.interval,
                    actions: ev.actions.clone(),
                });
            }
        } else {
            out.push(QueuedEvent {
                time: ep.start_time + ev.time,
                actions: ev.actions,
            });
        }
    }
    out.into_iter()
}
