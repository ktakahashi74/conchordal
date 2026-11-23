use super::*;

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FeltRecord<'a> {
    FeltRhythm {
        voice_id: u64,
        generation: u32,
        #[serde(flatten)]
        record: &'a crate::life::phonation_engine::FeltRhythmRecord,
    },
    FeltBeat {
        observed_end: u64,
        sample_rate: u32,
        #[serde(flatten)]
        state: crate::core::felt_beat::FeltBeatState,
    },
}

impl JsonlReporter {
    pub(crate) fn write_felt_rhythm(
        &mut self,
        voice_id: u64,
        generation: u32,
        record: &crate::life::phonation_engine::FeltRhythmRecord,
    ) -> Result<(), String> {
        self.write_record(&FeltRecord::FeltRhythm {
            voice_id,
            generation,
            record,
        })
    }

    pub(crate) fn write_felt_beat(
        &mut self,
        observed_end: u64,
        sample_rate: u32,
        state: crate::core::felt_beat::FeltBeatState,
    ) -> Result<(), String> {
        self.write_record(&FeltRecord::FeltBeat {
            observed_end,
            sample_rate,
            state,
        })
    }
}
