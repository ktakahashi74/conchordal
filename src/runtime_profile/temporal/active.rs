use serde::Serialize;

#[derive(Debug, Default, Serialize)]
pub(crate) struct TemporalMeterProfile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) t2_meter: Option<crate::life::report::TemporalMeterSample>,
}

#[derive(Serialize)]
pub(crate) struct SharedWorkerProfile {
    pub bus: u8,
    pub state: crate::temporal_cognition::observation::ObservationState,
    pub received_frames: u64,
    pub source_missing_samples: u64,
    pub delivery_dropped_frames: u64,
    pub rejected_frames: u64,
}

#[derive(Serialize)]
pub(crate) struct BodyWorkerProfile {
    pub finished: bool,
    pub input_end: u64,
    pub processed_frames: u64,
    pub invalid_hops: u64,
    pub capture_drops: u64,
    pub outside_voice_hops: u64,
    pub worker_resources: crate::temporal_cognition::resources::Snapshot,
}

#[derive(Serialize)]
pub(crate) struct BackgroundTemporalProfile {
    pub(crate) shared: [Option<SharedWorkerProfile>; 2],
    pub(crate) body: Option<BodyWorkerProfile>,
}
