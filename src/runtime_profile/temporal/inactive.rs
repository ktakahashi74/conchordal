use serde::Serialize;

#[derive(Debug, Default, Serialize)]
pub(crate) struct TemporalMeterProfile {}

#[derive(Serialize)]
pub(crate) struct BackgroundTemporalProfile {
    pub(crate) shared: [Option<()>; 2],
    pub(crate) body: Option<()>,
}
