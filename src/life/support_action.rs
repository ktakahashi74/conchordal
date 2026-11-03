//! Local C/N consequences over the consumer's next decision window.
use crate::core::harmonic_support::{Snapshot, Weights};
use crate::life::tone_energy::{ScheduledRelease, ToneEnergy};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
// Keep issued renderer state inline; refreshing this value must not allocate per hop.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Body {
    Silent,
    One {
        pitch: f32,
        energy: ToneEnergy,
    },
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    Onset,
    Continue,
    Release,
}

#[derive(Clone, Debug)]
pub(crate) struct Context {
    pub source: Option<(u64, u32)>,
    pub snapshot: Arc<Snapshot>,
    pub template: Option<Arc<[(isize, f32)]>>,
    pub single_density: f32,
    pub weights: Weights,
    pub loudness_exp: f32,
    pub nominal_amp: f32,
    pub pitch: f32,
    pub onset: Option<ToneEnergy>,
    pub body: Body,
    pub replace: bool,
    pub allow_release: bool,
    pub retention_bonus: f64,
}

pub(crate) fn active(weights: Weights) -> bool {
    weights.0[0] != 0. || weights.0[2] != 0.
}

/// At least two midpoint cells per shortest nonconstant envelope phase.
/// The window and its phase durations come from the consumer and the body.
pub(crate) fn cells(width: u64, phase: u64) -> u64 {
    width.div_ceil((phase / 2).max(1)).max(2).min(width.max(1))
}

/// Refine only when the consumer's decision changes. The final grid is the body's
/// integer sample grid, so the bound comes from the supported window, not a new cap.
pub(crate) fn resolved_choice<T: Copy>(
    width: u64,
    mut evaluate: impl FnMut(i32) -> Option<(usize, T)>,
) -> Option<(usize, T, i32)> {
    if width == 0 {
        return None;
    }
    let mut coarse = evaluate(-1)?;
    let mut middle = evaluate(0)?;
    for level in 1..=(width.ilog2() as i32 + 1) {
        let fine = evaluate(level)?;
        if coarse.0 == middle.0 && middle.0 == fine.0 {
            return Some((middle.0, middle.1, level - 1));
        }
        coarse = middle;
        middle = fine;
    }
    Some((middle.0, middle.1, width.ilog2() as i32 + 1))
}

pub(crate) fn resolution_cells(base: u64, width: u64, level: i32) -> u64 {
    if level < 0 {
        (base / 2).max(1)
    } else {
        base.saturating_mul(1u64.checked_shl(level as u32).unwrap_or(u64::MAX))
            .min(width.max(1))
    }
}

impl Context {
    pub(crate) fn choose(&self, at: u64, end: u64, baseline: Action) -> Action {
        if !active(self.weights) || self.snapshot.end_sample > at || end <= at {
            return baseline;
        }
        let Some(onset) = self.onset else {
            return baseline;
        };
        let phases = [
            onset.envelope,
            match self.body {
                Body::One { energy, .. } => energy.envelope,
                Body::Silent => onset.envelope,
                Body::Unknown => return baseline,
            },
        ];
        let phase = phases
            .iter()
            .flat_map(|e| [e.attack_ticks, e.decay_ticks, e.release_ticks])
            .filter(|&n| n > 0)
            .min()
            .unwrap_or(end - at);
        let n = cells(end - at, phase);
        let Some((_, (best, best_delta, base, selected_features), resolution)) =
            resolved_choice(end - at, |resolution| {
                let n = resolution_cells(n, end - at, resolution);
                let base = self.mean(at, end, n, baseline)?;
                let mut best = baseline;
                let mut selected_features = base;
                let mut best_delta = 0.;
                for action in [Action::Onset, Action::Continue, Action::Release] {
                    if action == Action::Onset
                        && baseline != Action::Onset
                        && !matches!(self.body, Body::Silent)
                    {
                        continue;
                    }
                    if action == baseline
                        || (action == Action::Release
                            && (!self.allow_release || matches!(self.body, Body::Silent)))
                    {
                        continue;
                    }
                    let Some(value) = self.mean(at, end, n, action) else {
                        continue;
                    };
                    let native = |a| {
                        if a == Action::Continue {
                            self.retention_bonus
                        } else {
                            0.
                        }
                    };
                    let delta = self
                        .weights
                        .value([value[0] - base[0], 0., value[1] - base[1]])
                        + native(action)
                        - native(baseline);
                    if delta > best_delta {
                        best = action;
                        selected_features = value;
                        best_delta = delta;
                    }
                }
                Some((best as usize, (best, best_delta, base, selected_features)))
            })
        else {
            return baseline;
        };
        let n = resolution_cells(n, end - at, resolution);
        tracing::trace!(
            source=?self.source,
            at,
            end,
            cells = n,
            ?baseline,
            ?best,
            best_delta,
            base_c=base[0], base_n=base[1], selected_c=selected_features[0], selected_n=selected_features[1],
            "support action selection"
        );
        best
    }

    fn mean(&self, at: u64, end: u64, n: u64, action: Action) -> Option<[f64; 2]> {
        let single = [(0, self.single_density)];
        let template = self.template.as_deref().unwrap_or(&single);
        let space = &self.snapshot.geometry.space;
        let new_bin = space.nearest_index(self.pitch.exp2());
        let (old_bin, old) = match self.body {
            Body::Silent => (new_bin, None),
            Body::One { pitch, energy } => (space.nearest_index(pitch.exp2()), Some(energy)),
            Body::Unknown => return None,
        };
        let mut onset = self.onset?;
        onset.envelope.onset = at;
        if let Some(control) = &mut onset.control {
            control.issued_at = at;
            control.starts_at = Some(at);
            control.kick_at = Some(at);
        }
        let release = (action == Action::Release || (action == Action::Onset && self.replace))
            .then_some(ScheduledRelease {
                apply_at_sample: at,
                off_sample: at,
            });
        let reference = 0.5 * f64::from(self.nominal_amp).powi(2);
        if reference <= 0. || !reference.is_finite() {
            return None;
        }
        let mut result = [0.; 2];
        let mut previous: Option<([u32; 2], [f64; 3])> = None;
        for i in 0..n {
            let tick = at
                + ((u128::from(end - at) * (2 * u128::from(i) + 1)) / (2 * u128::from(n))) as u64;
            let old_energy = old.map_or(Some(0.), |mut tone| {
                if action == Action::Onset && self.replace {
                    tone.envelope = tone.envelope.with_replacement_release(at);
                    tone.at(tick, None)
                } else {
                    tone.at(tick, release)
                }
            })?;
            let new_energy = if action == Action::Onset {
                onset.at(tick, None)?
            } else {
                0.
            };
            let scale =
                |energy: f64| (energy / reference).powf(f64::from(self.loudness_exp)) as f32;
            let scales = [scale(old_energy), scale(new_energy)];
            if scales.iter().any(|s| !s.is_finite() || *s < 0.) {
                return None;
            }
            let key = scales.map(f32::to_bits);
            let features = match previous {
                Some((last, features)) if last == key => features,
                _ => self.snapshot.paired_template_features(
                    template,
                    [(old_bin, scales[0]), (new_bin, scales[1])],
                ),
            };
            previous = Some((key, features));
            result[0] += features[0] / n as f64;
            result[1] += features[2] / n as f64;
        }
        Some(result)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::{
        harmonic_support::Geometry,
        harmonicity_kernel::{HarmonicityKernel, HarmonicityParams},
        log2space::Log2Space,
        timebase::Timebase,
    };
    use crate::life::sound::{
        ToneAdsr,
        control_forecast::{AmplitudeModel, ControlForecast},
    };

    pub(crate) fn fixture() -> Context {
        let space = Log2Space::new(55., 1760., 24);
        let geometry = Geometry::new(
            &space,
            &HarmonicityKernel::new(&space, HarmonicityParams::default()),
        );
        let mut snapshot = Snapshot::new(geometry);
        let mut density = vec![0.; space.n_bins()];
        density[45] = 1.;
        snapshot.observe(&density, 1e-12, 512);
        let bin = (0..space.n_bins())
            .min_by(|&a, &b| {
                snapshot.template_features(a, &[(0, 1.)])[0]
                    .total_cmp(&snapshot.template_features(b, &[(0, 1.)])[0])
            })
            .unwrap();
        assert!(snapshot.template_features(bin, &[(0, 1.)])[0] < 0.);
        let onset = ToneEnergy {
            amplitude: 1.,
            envelope: crate::life::sound::tone::preview_envelope(
                Timebase {
                    fs: 48000.,
                    hop: 512,
                },
                512,
                u64::MAX,
                Some(ToneAdsr {
                    attack_sec: 0.002,
                    decay_sec: 0.004,
                    sustain_level: 0.4,
                    release_sec: 0.006,
                }),
            ),
            control: Some(ControlForecast {
                issued_at: 512,
                valid_until: None,
                amplitude_smoothing: None,
                amplitude_updates: None,
                sample_dt: 1. / 48000.,
                starts_at: Some(512),
                kick_at: None,
                model: AmplitudeModel::Unmodulated { gain: 1. },
            }),
            scheduled_release: None,
            sine: None,
            bank: None,
        };
        Context {
            source: None,
            snapshot: Arc::new(snapshot),
            template: Some(Arc::from([(0, 1.)])),
            single_density: 1.,
            weights: Weights([1., 0., 0.]),
            loudness_exp: 0.3,
            nominal_amp: 1.,
            pitch: space.freq_of_index(bin).log2(),
            onset: Some(onset),
            body: Body::Silent,
            replace: true,
            allow_release: true,
            retention_bonus: 0.,
        }
    }

    #[test]
    fn support_action_signed_focus_prefers_no_add_independently_of_temporal_controls() {
        let mut c = fixture();
        assert_eq!(c.choose(512, 1024, Action::Onset), Action::Continue);
        c.weights.0[0] = -1.;
        assert_eq!(c.choose(512, 1024, Action::Onset), Action::Onset);
        c.weights = Weights::default();
        c.body = Body::Unknown;
        for a in [Action::Onset, Action::Continue, Action::Release] {
            assert_eq!(c.choose(512, 1024, a), a);
        }
        c.weights.0[0] = 1.;
        assert_eq!(c.choose(512, 1024, Action::Onset), Action::Onset);
        c.body = Body::Silent;
        assert_eq!(c.choose(511, 1024, Action::Onset), Action::Onset);
    }

    #[test]
    fn support_action_keeps_release_tail_and_midpoint_rank_at_half_double_and_sample_resolution() {
        let mut c = fixture();
        c.body = Body::One {
            pitch: c.pitch,
            energy: c.onset.unwrap(),
        };
        let start = 1024;
        let end = 1536;
        let n = cells(end - start, c.onset.unwrap().envelope.attack_ticks);
        for count in [(n / 2).max(1), n, n * 2, end - start] {
            let held = c.mean(start, end, count, Action::Continue).unwrap();
            let released = c.mean(start, end, count, Action::Release).unwrap();
            assert!(released[0] > held[0]);
            assert!(released[0] < 0., "release is not silence");
            let added = c.mean(start, end, count, Action::Onset).unwrap();
            assert!(added[0] < released[0]);
            assert!(added[1].is_finite() && held[1].is_finite());
        }
        assert_eq!(c.choose(start, end, Action::Continue), Action::Release);
        c.weights.0[0] = -1.;
        assert_eq!(c.choose(start, end, Action::Continue), Action::Continue);
        c.onset
            .as_mut()
            .unwrap()
            .control
            .as_mut()
            .unwrap()
            .valid_until = Some(start + 1);
        c.body = Body::One {
            pitch: c.pitch,
            energy: c.onset.unwrap(),
        };
        assert_eq!(c.choose(start, end, Action::Continue), Action::Continue);
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn support_action_steady_evaluation_allocates_nothing() {
        let c = fixture();
        c.choose(512, 1024, Action::Onset);
        crate::runtime_profile::begin_allocations();
        for _ in 0..64 {
            c.choose(512, 1024, Action::Onset);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(counts.count, 0);
        assert_eq!(counts.bytes, 0);
    }
}
