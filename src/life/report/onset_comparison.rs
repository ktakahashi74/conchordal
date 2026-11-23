use super::*;

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OnsetRecord<'a> {
    ParticipationDecision {
        voice_id: u64,
        sample_rate: u32,
        #[serde(flatten)]
        decision: &'a crate::life::temporal_participation::ParticipationDecision,
    },
    BodyFootprintWorker(&'a crate::life::onset_footprint::Stats),
    BodyFootprint {
        /// The Voice discarded the record because its request was superseded.
        superseded: bool,
        #[serde(flatten)]
        record: &'a crate::life::onset_footprint::Record,
    },
}

impl JsonlReporter {
    pub(crate) fn write_participation_decision(
        &mut self,
        voice_id: u64,
        sample_rate: u32,
        decision: &crate::life::temporal_participation::ParticipationDecision,
    ) -> Result<(), String> {
        self.write_record(&OnsetRecord::ParticipationDecision {
            voice_id,
            sample_rate,
            decision,
        })
    }

    pub(crate) fn write_body_footprint_worker(
        &mut self,
        stats: &crate::life::onset_footprint::Stats,
    ) -> Result<(), String> {
        self.write_record(&OnsetRecord::BodyFootprintWorker(stats))
    }

    pub(crate) fn write_body_footprint(
        &mut self,
        record: &crate::life::onset_footprint::Record,
        superseded: bool,
    ) -> Result<(), String> {
        self.write_record(&OnsetRecord::BodyFootprint { superseded, record })
    }
}
