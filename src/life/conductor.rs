use std::collections::VecDeque;

use tracing::info;

use super::population::Population;
use super::scenario::{Action, Scenario, Scene};
use crate::core::landscape::LandscapeFrame;

#[derive(Debug, Clone)]
pub struct QueuedEvent {
    pub time: f32,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
struct SceneInfo {
    name: Option<String>,
    start_time: f32,
}

#[derive(Debug)]
pub struct Conductor {
    event_queue: VecDeque<QueuedEvent>,
    total_duration: f32,
    scenes: Vec<SceneInfo>,
}

impl Conductor {
    pub fn from_scenario(s: Scenario) -> Self {
        let mut scenes: Vec<SceneInfo> = s
            .scenes
            .iter()
            .map(|sc| SceneInfo {
                name: sc.name.clone(),
                start_time: sc.start_time,
            })
            .collect();
        scenes.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut events: Vec<QueuedEvent> = s.scenes.into_iter().flat_map(flatten_scene).collect();

        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_duration = events.last().map(|ev| ev.time).unwrap_or(0.0);
        Self {
            event_queue: events.into(),
            total_duration,
            scenes,
        }
    }

    pub fn from_events(events: Vec<QueuedEvent>) -> Self {
        let mut events = events;
        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total_duration = events.last().map(|ev| ev.time).unwrap_or(0.0);
        Self {
            event_queue: events.into(),
            total_duration,
            scenes: Vec::new(),
        }
    }

    /// Apply any events scheduled up to and including current time.
    pub fn dispatch_until(
        &mut self,
        time_sec: f32,
        _current_frame: u64,
        landscape: &LandscapeFrame,
        mut landscape_rt: Option<&mut crate::core::landscape::Landscape>,
        population: &mut Population,
    ) {
        while let Some(ev) = self.event_queue.front() {
            if ev.time > time_sec {
                break;
            }

            let ev = self.event_queue.pop_front().expect("front exists");
            let action_descs: Vec<String> = ev.actions.iter().map(ToString::to_string).collect();
            info!("[t={:.3}] Event: {}", ev.time, action_descs.join(" | "));
            for action in ev.actions {
                population.apply_action(action, landscape, landscape_rt.as_deref_mut());
            }
        }
    }

    pub fn is_done(&self) -> bool {
        self.event_queue.is_empty()
    }

    pub fn total_duration(&self) -> f32 {
        self.total_duration
    }

    pub fn remaining_events(&self) -> usize {
        self.event_queue.len()
    }

    pub fn current_scene_name(&self, time_sec: f32) -> Option<String> {
        let mut current: Option<String> = None;
        for scene in &self.scenes {
            if time_sec + f32::EPSILON >= scene.start_time {
                current = scene.name.clone();
            } else {
                break;
            }
        }
        current
    }
}

fn flatten_scene(ep: Scene) -> impl Iterator<Item = QueuedEvent> {
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
