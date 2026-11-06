use super::ScriptWarning;
use std::collections::BTreeMap;
#[cfg(feature = "native")]
use std::sync::Weak;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Credit {
    pub actions: usize,
    pub bytes: usize,
}

impl Credit {
    pub fn fits(self, available: Self) -> bool {
        self.actions <= available.actions && self.bytes <= available.bytes
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScriptStatus {
    pub evaluated_through: f64,
    pub done: bool,
    pub error: Option<String>,
    pub error_position: Option<rhai::Position>,
    pub stopped: bool,
    pub warnings: Vec<ScriptWarning>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct WarningStore {
    pub(super) entries: BTreeMap<(&'static str, rhai::Position), ScriptWarning>,
    #[cfg(feature = "native")]
    pub(super) outbox: Option<Weak<super::score::ScoreStream>>,
}

impl WarningStore {
    pub fn push(&mut self, warning: ScriptWarning) {
        #[cfg(feature = "native")]
        if let Some(stream) = self.outbox.as_ref().and_then(Weak::upgrade) {
            stream
                .state
                .lock()
                .unwrap()
                .warnings
                .entry((warning.kind, warning.position))
                .and_modify(|entry| entry.count += 1)
                .or_insert(warning);
            stream.changed.notify_all();
            return;
        }
        {
            self.entries
                .entry((warning.kind, warning.position))
                .and_modify(|entry| entry.count += 1)
                .or_insert(warning);
        }
    }
    pub fn snapshot(&self) -> Vec<ScriptWarning> {
        #[cfg(feature = "native")]
        if let Some(stream) = self.outbox.as_ref().and_then(Weak::upgrade) {
            return stream
                .state
                .lock()
                .unwrap()
                .warnings
                .values()
                .cloned()
                .collect();
        }
        self.entries.values().cloned().collect()
    }
    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = &ScriptWarning> {
        self.entries.values()
    }
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
