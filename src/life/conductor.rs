use std::collections::VecDeque;

use tracing::info;

use super::population::Population;
use super::scenario::{Action, Scenario, SceneMarker, TimedEvent};
use crate::core::landscape::LandscapeFrame;

#[derive(Debug, Clone)]
pub struct QueuedEvent {
    pub time: f32,
    pub order: u64,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
struct SceneInfo {
    name: String,
    time: f32,
    order: u64,
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
            .scene_markers
            .into_iter()
            .map(|SceneMarker { name, time, order }| SceneInfo { name, time, order })
            .collect();
        scenes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.order.cmp(&b.order))
        });

        let mut events: Vec<QueuedEvent> = s
            .events
            .into_iter()
            .map(
                |TimedEvent {
                     time,
                     order,
                     actions,
                 }| QueuedEvent {
                    time,
                    order,
                    actions,
                },
            )
            .collect();

        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.order.cmp(&b.order))
        });

        let total_duration = s.duration_sec;
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
                .then_with(|| a.order.cmp(&b.order))
        });
        let total_duration = events.last().map(|ev| ev.time).unwrap_or(0.0);
        Self {
            event_queue: events.into(),
            total_duration,
            scenes: Vec::new(),
        }
    }

    /// Apply any events scheduled up to and including current time.
    ///
    /// Note: spawn placement rules (e.g. minimum ERB distance between fundamentals) are enforced
    /// inside `Population` when handling spawn actions.
    pub fn dispatch_until(
        &mut self,
        time_sec: f32,
        _current_frame: u64,
        landscape: &LandscapeFrame,
        mut roughness_rt: Option<&mut crate::core::stream::roughness::RoughnessStream>,
        population: &mut Population,
        world: &mut crate::life::world_model::WorldModel,
    ) {
        while let Some(ev) = self.event_queue.front() {
            if ev.time > time_sec {
                break;
            }

            let ev = self.event_queue.pop_front().expect("front exists");
            let action_descs: Vec<String> = ev.actions.iter().map(ToString::to_string).collect();
            info!("[t={:.3}] Event: {}", ev.time, action_descs.join(" | "));
            for action in ev.actions {
                match action {
                    crate::life::scenario::Action::PostNote { .. } => {
                        world.apply_action(&action);
                    }
                    _ => {
                        population.apply_action(action, landscape, roughness_rt.as_deref_mut());
                    }
                }
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
            if time_sec + f32::EPSILON >= scene.time {
                current = Some(scene.name.clone());
            } else {
                break;
            }
        }
        current
    }
}
