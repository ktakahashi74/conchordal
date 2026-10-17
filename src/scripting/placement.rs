use super::*;

#[derive(Clone, Copy, Debug)]
pub struct Bus {
    pub(super) routing: Routing,
}

impl Bus {
    pub fn habitat() -> Self {
        Self {
            routing: Routing {
                to_presentation: false,
                to_habitat: true,
            },
        }
    }

    pub fn presentation() -> Self {
        Self {
            routing: Routing {
                to_presentation: true,
                to_habitat: false,
            },
        }
    }

    pub(super) fn set(self) -> BusSet {
        BusSet {
            routing: self.routing,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BusSet {
    pub(super) routing: Routing,
}

impl BusSet {
    pub(super) fn combine(self, other: Self) -> Self {
        Self {
            routing: Routing {
                to_presentation: self.routing.to_presentation || other.routing.to_presentation,
                to_habitat: self.routing.to_habitat || other.routing.to_habitat,
            },
        }
    }

    pub(super) fn routing(self) -> Routing {
        self.routing
    }
}

#[derive(Clone, Debug)]
pub struct Placement {
    pub(super) diagnostics: Arc<Mutex<Vec<ScriptWarning>>>,
    pub(super) position: Position,
    pub(super) count: i64,
    pub(super) strategy: Option<SpawnStrategy>,
    pub(super) freq_hz: Option<f32>,
    /// Set when built from `consonance(root)`, so `range()` reads multiples.
    pub(super) root_hz: Option<f32>,
}

impl Placement {
    pub(super) fn warn(&self, message: String) {
        self.diagnostics
            .lock()
            .expect("lock script warnings")
            .push(ScriptWarning {
                message,
                position: self.position,
            });
    }

    pub(super) fn field(
        target: FieldTarget,
        min_freq: f32,
        max_freq: f32,
        min_dist_erb: f32,
    ) -> Self {
        Self {
            diagnostics: Arc::default(),
            position: Position::NONE,
            count: 1,
            strategy: Some(SpawnStrategy::Field {
                target,
                sampling: FieldSampling::Density,
                min_freq,
                max_freq,
                min_dist_erb,
                tension: 0.0,
            }),
            freq_hz: None,
            root_hz: None,
        }
    }

    pub(super) fn at(freq_hz: f32) -> Self {
        Self {
            diagnostics: Arc::default(),
            position: Position::NONE,
            count: 1,
            strategy: None,
            freq_hz: Some(freq_hz),
            root_hz: None,
        }
    }

    pub(super) fn consonance_range(min_freq: f32, max_freq: f32) -> Self {
        Self::field(FieldTarget::Consonance, min_freq, max_freq, 1.0)
    }

    pub(super) fn consonance_root(root_freq: f32) -> Self {
        // Default harmonic window [1x, 4x]; `range(lo, hi)` overrides the multiples.
        let mut p = Self::field(FieldTarget::Consonance, root_freq, root_freq * 4.0, 1.0);
        p.root_hz = Some(root_freq);
        p
    }

    pub(super) fn dissonance(min_freq: f32, max_freq: f32) -> Self {
        Self::field(FieldTarget::Dissonance, min_freq, max_freq, 1.0)
    }

    pub(super) fn edge(min_freq: f32, max_freq: f32) -> Self {
        Self::field(FieldTarget::Edge, min_freq, max_freq, 1.0)
    }

    pub(super) fn gap(min_freq: f32, max_freq: f32) -> Self {
        Self::field(FieldTarget::Gap, min_freq, max_freq, 1.0)
    }

    pub(super) fn random(min_freq: f32, max_freq: f32) -> Self {
        // Uniform measure; no crowding spacing by default.
        Self::field(FieldTarget::Uniform, min_freq, max_freq, 0.0)
    }

    pub(super) fn line(start_freq: f32, end_freq: f32) -> Self {
        Self {
            diagnostics: Arc::default(),
            position: Position::NONE,
            count: 1,
            strategy: Some(SpawnStrategy::Linear {
                start_freq,
                end_freq,
            }),
            freq_hz: None,
            root_hz: None,
        }
    }

    pub(super) fn with_count(mut self, count: i64) -> Self {
        self.count = count;
        self
    }

    pub(super) fn with_sampling(mut self, sampling: FieldSampling) -> Self {
        if !matches!(self.strategy, Some(SpawnStrategy::Field { .. })) {
            self.warn("peak()/density() only apply to field placements; ignored".into());
            return self;
        }
        self.strategy = self.strategy.take().map(|strategy| match strategy {
            SpawnStrategy::Field {
                target,
                min_freq,
                max_freq,
                min_dist_erb,
                tension,
                ..
            } => SpawnStrategy::Field {
                target,
                sampling,
                min_freq,
                max_freq,
                min_dist_erb,
                tension,
            },
            other => {
                self.warn("peak()/density() only apply to field placements; ignored".to_string());
                other
            }
        });
        self
    }

    pub(super) fn with_tension(mut self, tension: f32) -> Self {
        let t = if tension.is_finite() {
            tension.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self.strategy {
            Some(SpawnStrategy::Field {
                target: FieldTarget::Consonance,
                tension: ref mut tn,
                ..
            }) => *tn = t,
            _ => {
                self.warn("tension() only applies to consonance() placements; ignored".to_string())
            }
        }
        self
    }

    pub(super) fn with_range(mut self, min_mul: f32, max_mul: f32) -> Self {
        match (self.root_hz, self.strategy.take()) {
            (
                Some(root),
                Some(SpawnStrategy::Field {
                    target,
                    sampling,
                    min_dist_erb,
                    tension,
                    ..
                }),
            ) => {
                self.strategy = Some(SpawnStrategy::Field {
                    target,
                    sampling,
                    min_freq: root * min_mul,
                    max_freq: root * max_mul,
                    min_dist_erb,
                    tension,
                });
            }
            (_, strategy) => {
                self.strategy = strategy;
                self.warn("range() is only supported for consonance(root); ignored".to_string());
            }
        }
        self
    }

    pub(super) fn with_spacing(mut self, spacing: f32) -> Self {
        if !matches!(self.strategy, Some(SpawnStrategy::Field { .. })) {
            self.warn("spacing() is only supported for field placements; ignored".into());
            return self;
        }
        self.strategy = self.strategy.take().map(|strategy| match strategy {
            SpawnStrategy::Field {
                target,
                sampling,
                min_freq,
                max_freq,
                tension,
                ..
            } => SpawnStrategy::Field {
                target,
                sampling,
                min_freq,
                max_freq,
                min_dist_erb: spacing,
                tension,
            },
            other => {
                self.warn("spacing() is only supported for field placements; ignored".to_string());
                other
            }
        });
        self
    }

    pub(super) fn strategy(&self) -> Option<SpawnStrategy> {
        self.strategy.clone()
    }
}
