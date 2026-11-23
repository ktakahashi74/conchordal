use crate::config::{AppConfig, FeltBeatParams};
use crate::core::meter::MeterNetwork;
use crate::scenario::control::PhonationControl;
use crate::scenario::{Action, Scenario};

#[derive(Default)]
pub(crate) struct FeltConfig {
    params: Option<FeltBeatParams>,
    period_guard: bool,
}

impl FeltConfig {
    pub(crate) fn from_scene(config: &AppConfig, scenario: &Scenario) -> Self {
        let period_guard = scenario.events.iter().flat_map(|e| &e.actions).any(
            |action| matches!(action, Action::Spawn { spec, .. } if spec.control.phonation.syncopate > 0.0),
        );
        let params = config
            .analysis
            .felt_beat
            .or_else(|| (scenario.observe_felt_beat || period_guard).then(FeltBeatParams::default));
        Self {
            params,
            period_guard,
        }
    }

    pub(crate) fn requested(&self) -> bool {
        self.params.is_some()
    }

    pub(crate) fn weight(control: &PhonationControl) -> f32 {
        control.syncopate
    }

    pub(crate) fn setup_meter(&self, meter: &mut MeterNetwork, fs: f32, hop: usize) {
        if let Some(params) = self.params {
            meter.enable_felt_beat(params);
            meter.enable_felt_input(fs, hop);
            if self.period_guard {
                meter.enable_felt_period_guard();
            }
        }
    }

    #[cfg(feature = "native")]
    pub(crate) fn before_action(action: &Action, meter: Option<&mut MeterNetwork>) {
        if let Action::Spawn { spec, .. } = action
            && spec.control.phonation.syncopate > 0.0
            && let Some(meter) = meter
        {
            meter.enable_felt_period_guard();
        }
    }
}
