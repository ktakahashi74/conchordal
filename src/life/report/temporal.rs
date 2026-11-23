use super::*;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct TemporalMeterSample {
    pub sample_rate: u32,
    pub hop: u64,
    pub use_cut: u64,
    pub input: crate::temporal_cognition::t2::runtime_input::Diagnostics,
    pub recurrence: crate::core::meter::ContinuousDiagnostics,
    pub state: crate::core::meter::MeterState,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TemporalRecord<'a> {
    TemporalMeter(&'a TemporalMeterSample),
    BodyObservation(&'a crate::temporal_cognition::body::Snapshot),
    BodyDescriptor {
        #[serde(flatten)]
        record: &'a crate::temporal_cognition::body::Record,
        standardized: [Option<f64>; 6],
        prototype_assignment: Option<crate::temporal_cognition::body_model::Assignment>,
    },
    TemporalObservation {
        time_sec: f64,
        observation: &'a crate::temporal_cognition::observation::Snapshot,
    },
}

impl JsonlReporter {
    pub(crate) fn write_temporal_meter(
        &mut self,
        sample: &TemporalMeterSample,
    ) -> Result<(), String> {
        self.write_record(&TemporalRecord::TemporalMeter(sample))
    }

    pub(crate) fn write_temporal_observation(
        &mut self,
        time_sec: f64,
        observation: &crate::temporal_cognition::observation::Snapshot,
    ) -> Result<(), String> {
        self.write_record(&TemporalRecord::TemporalObservation {
            time_sec,
            observation,
        })
    }

    pub(crate) fn write_body_observation(
        &mut self,
        snapshot: &crate::temporal_cognition::body::Snapshot,
    ) -> Result<(), String> {
        self.write_record(&TemporalRecord::BodyObservation(snapshot))?;
        for (index, record) in snapshot
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.active)
        {
            self.write_record(&TemporalRecord::BodyDescriptor {
                record,
                standardized: record.standardized(snapshot.config),
                prototype_assignment: snapshot.prototype_assignments[index],
            })?;
        }
        Ok(())
    }
}
