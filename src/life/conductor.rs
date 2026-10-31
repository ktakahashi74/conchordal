use std::cmp::Ordering;
use std::collections::VecDeque;

use tracing::info;

use super::community::Community;
use crate::core::landscape::Landscape;
use crate::core::timebase::{Tick, Timebase};
use crate::scenario::{Action, Scenario, SceneMarker, TimedEvent};

#[derive(Debug, Clone)]
pub struct QueuedEvent {
    pub time: f64,
    tick: Tick,
    pub order: u64,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
struct SceneInfo {
    name: String,
    time: f64,
    tick: Tick,
    order: u64,
}

#[derive(Debug)]
pub struct Conductor {
    pub(crate) expression: Option<crate::scenario::expression::Expression>,
    pub(crate) score: Option<crate::scripting::score::ScriptSession>,
    pub(crate) current_scene: Option<String>,
    event_queue: VecDeque<QueuedEvent>,
    end: Option<f64>,
    scenes: Vec<SceneInfo>,
    dispatched_through: Option<Tick>,
}

impl Conductor {
    pub fn from_scenario(s: Scenario, timebase: Timebase) -> Self {
        let mut scenes: Vec<SceneInfo> = s
            .scene_markers
            .into_iter()
            .map(|SceneMarker { name, time, order }| SceneInfo {
                name,
                time,
                tick: timebase.sec_to_tick(time),
                order,
            })
            .collect();
        scenes.sort_by(|a, b| cmp_time_order(a.time, a.order, b.time, b.order));

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
                    tick: timebase.sec_to_tick(time),
                    order,
                    actions,
                },
            )
            .collect();

        events.sort_by(|a, b| cmp_time_order(a.time, a.order, b.time, b.order));

        let end = s.end.duration_sec();
        Self {
            expression: s.expression,
            score: None,
            current_scene: None,
            event_queue: events.into(),
            end,
            scenes,
            dispatched_through: None,
        }
    }

    /// Apply any events scheduled up to and including current time.
    ///
    /// Note: spawn placement rules (e.g. minimum ERB distance between fundamentals) are enforced
    /// inside `Community` when handling spawn actions.
    pub fn dispatch_until(
        &mut self,
        tick: Tick,
        landscape: &Landscape,
        community: &mut Community,
    ) -> bool {
        let mut finish = false;
        // Audio thread: pop-then-restore avoids an `expect`, and logging one line per
        // action avoids joining descriptions into a temporary `String`.
        while let Some(ev) = self.event_queue.pop_front() {
            if ev.tick > tick {
                self.event_queue.push_front(ev);
                break;
            }

            let time = ev.time;
            for action in ev.actions {
                info!("[t={time:.3}] Event: {action}");
                if matches!(action, Action::Finish) {
                    finish = true;
                } else {
                    community.apply_action(action, landscape);
                }
            }
        }
        self.dispatched_through = Some(tick);
        finish
    }

    pub fn discard_pending(&mut self) {
        if let Some(session) = &self.score {
            session.stream.close("stopped");
        }
        self.event_queue.clear();
        self.scenes.retain(|scene| {
            self.dispatched_through
                .is_some_and(|tick| scene.tick <= tick)
        });
    }

    pub fn end(&self) -> Option<f64> {
        self.end
    }

    pub(crate) fn has_pending_population(&self, id: u64) -> bool {
        self.event_queue.iter().any(|event| {
            event.actions.iter().any(|action|
            matches!(action, Action::Spawn { population_id, .. } if *population_id == id))
        })
    }

    pub fn pending_events(&self) -> usize {
        self.score.as_ref().map_or(self.event_queue.len(), |s| {
            s.stream.state.lock().unwrap().ring.len()
        })
    }

    pub fn current_scene_name(&self, tick: Tick) -> Option<String> {
        if self.score.is_some() {
            return self.current_scene.clone();
        }
        let mut current: Option<String> = None;
        for scene in &self.scenes {
            if tick >= scene.tick {
                current = Some(scene.name.clone());
            } else {
                break;
            }
        }
        current
    }
}

#[inline]
fn cmp_time_order(a_time: f64, a_order: u64, b_time: f64, b_order: u64) -> Ordering {
    a_time
        .partial_cmp(&b_time)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a_order.cmp(&b_order))
}
