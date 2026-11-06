use serde::Serialize;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AudioDeviceInfo {
    pub(crate) backend: String,
    pub(crate) device_name: String,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) ring_capacity_frames: usize,
}

#[derive(Default)]
pub(crate) struct AudioCallbackCounters {
    pub(crate) callback_count: AtomicU64,
    pub(crate) callback_frames_total: AtomicU64,
    pub(crate) underrun_frames: Arc<AtomicU64>,
    pub(crate) callback_errors_total: AtomicU64,
    pub(crate) xruns_total: AtomicU64,
    pub(crate) rebuilds_total: AtomicU64,
    pub(crate) recovering: AtomicBool,
    pub(crate) failed: AtomicBool,
    pub(super) stream_lost: AtomicBool,
}

pub(crate) const STATS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
