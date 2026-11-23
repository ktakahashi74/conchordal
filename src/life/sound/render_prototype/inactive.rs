use crate::config::AppConfig;
use crate::core::timebase::Tick;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RenderPrototype;

impl RenderPrototype {
    #[inline(always)]
    pub(crate) fn from_config(_: &AppConfig) -> Self {
        Self
    }

    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn from_enabled(enabled: bool) -> Self {
        assert!(!enabled);
        Self
    }

    #[inline(always)]
    pub(crate) fn enabled(self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn hold_ticks(self, default: Tick) -> Tick {
        default
    }

    #[inline(always)]
    pub(crate) fn seed_source(self, source: u64, _: u32) -> u64 {
        source
    }
}
