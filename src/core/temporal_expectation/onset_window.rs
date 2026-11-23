use super::*;

impl TemporalForecast {
    /// No window before this frame may inform a prediction.
    pub(crate) fn available_through_frame(&self) -> u64 {
        self.available_through_frame
    }
}
