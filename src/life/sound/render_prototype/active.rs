use crate::config::AppConfig;
use crate::core::timebase::Tick;
use crate::life::schedule_renderer::modal_phase_seed;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RenderPrototype(bool);

impl RenderPrototype {
    pub(crate) fn from_config(config: &AppConfig) -> Self {
        Self(config.render_prototype)
    }

    #[cfg(test)]
    pub(crate) fn from_enabled(enabled: bool) -> Self {
        Self(enabled)
    }

    #[inline]
    pub(crate) fn enabled(self) -> bool {
        self.0
    }

    pub(crate) fn hold_ticks(self, default: Tick) -> Tick {
        if self.0 { Tick::MAX } else { default }
    }

    pub(crate) fn seed_source(self, source: u64, generation: u32) -> u64 {
        if self.0 {
            modal_phase_seed(source, u64::from(generation), 0)
        } else {
            source
        }
    }
}
