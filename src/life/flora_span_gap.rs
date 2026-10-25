//! Instance-wide spans reuse the Voice hazard law and the shared acoustic history.
use super::*;

impl Flora {
    pub(super) fn span_gap_opportunity(&mut self, at: u64) -> bool {
        let Some(gap) = &mut self.span_gap else {
            return false;
        };
        let before = gap.phase;
        gap.opportunity(at, false);
        if gap.phase == SpanPhase::GapRequested {
            gap.off(at);
        }
        gap.phase != before
    }

    pub(super) fn observe_sounding_span_gap(&mut self, at: u64, amp: f32) {
        let Some(gap) = &mut self.span_gap else {
            return;
        };
        let mut mass = 0.0_f64;
        let mut heard = 0.0_f64;
        let mut pitch = 0.0_f64;
        for lane in &self.lanes {
            if amp > 0.0
                && lane.tonal
                && lane.mode != Mode::Free
                && lane.r > 0.0
                && lane.target > 0.0
            {
                let power = f64::from(lane.target).powi(2);
                mass += power;
                heard += power * f64::from(gap.heard_at_log2(lane.sounding_log2));
                pitch += power * f64::from(lane.sounding_log2.exp2());
            }
        }
        gap.heard_override = Some(if mass > 0.0 {
            (heard / mass) as f32
        } else {
            0.0
        });
        gap.pitch = if mass > 0.0 {
            (pitch / mass) as f32
        } else {
            0.0
        };
        gap.next_pitch = gap.pitch;
        if gap.phase == SpanPhase::GapTail && self.lanes.iter().all(|lane| lane.r == 0.0) {
            gap.quiet(at, Some(at));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::meter::{MeterBand, MeterState};
    use crate::life::phonation_engine::CandidatePoint;

    fn setup(weight: f32, residual: bool) -> (Landscape, Flora) {
        let mut l = Landscape::new(Log2Space::new(220., 880., 96));
        l.onset_habituation_state_scan = vec![0.8; l.space.n_bins()].into();
        l.onset_habituation_known = true;
        l.consonance_field_level.fill(0.7);
        l.harmonicity01.fill(0.4);
        let mut f = Flora::new(
            1,
            FloraSpec {
                rest_on_habituation: weight,
                residual,
                pace: 16.,
                fade_sec: 0.,
                pitch_apply_mode: PitchApplyMode::Rearticulate,
                ..Default::default()
            },
            (300., 400.),
            FloraSeeding::Uniform,
            21,
            &l,
            48000.,
            512,
        );
        f.external_observed_frame = Some(0);
        for (i, x) in f.biomass_mass_scan.iter_mut().enumerate() {
            *x = if f.range.contains(&i) {
                0.001
                    + 0.1
                        * (-0.5
                            * ((l.space.centers_hz[i].log2() - 330_f32.log2()) * 96. / 1.5).powi(2))
                        .exp()
            } else {
                0.
            };
        }
        if let Some(gap) = &mut f.span_gap {
            gap.trace_enabled = true;
        }
        (l, f)
    }

    fn render(f: &mut Flora, l: &Landscape, now: u64, active: bool, crossing: bool) -> Vec<f32> {
        f.observe_rhythm(
            TemporalObservation {
                observed_end: Some(now),
                known: active,
                flux: 0.5,
                state: MeterState {
                    beat: MeterBand {
                        freq_hz: 2.,
                        confidence: 1.,
                        amplitude: 1.,
                        phase: 0.,
                    },
                    ..Default::default()
                },
            },
            now,
            Default::default(),
        );
        if let Some(gap) = &mut f.span_gap {
            gap.observe(l, now);
        }
        let m = f.temporal.as_mut().unwrap();
        m.crossings.clear();
        if crossing {
            m.crossings.push(CandidatePoint {
                tick: now + 173,
                gate: now,
            });
        }
        let mut p = vec![0.; 512];
        f.render_add(&mut p, &mut vec![0.; 512]);
        p
    }

    #[test]
    fn acoustic_histories_change_flora_exit_and_zero_heard_ablates_it() {
        use crate::core::auditory_frontend::AuditoryFrontEnd;
        use crate::core::onset_habituation::OnsetHabituation;
        let (_, params) = crate::runtime::body_metabolism_test_core();
        let space = Log2Space::new(55., 8000., 96);
        let mut histories = Vec::new();
        for hz in [330.0, 2000.0] {
            let mut front = AuditoryFrontEnd::selected(space.clone(), 48000., 512, 16384);
            let mut observer = OnsetHabituation::new(
                &space,
                front.band_energy().0,
                Default::default(),
                0.5,
                48000.,
            );
            for hop in 0..500 {
                let pcm: Vec<_> = (0..512)
                    .map(|i| {
                        let t = (hop * 512 + i) as f64 / 48000.;
                        (0.1 * (-t.rem_euclid(0.5) / 0.025).exp()
                            * (std::f64::consts::TAU * hz * t).sin()) as f32
                    })
                    .collect();
                front.process(&pcm, &params);
                observer.process(front.band_energy().1, 512, (hop + 1) * 512);
            }
            let mut l = Landscape::new(space.clone());
            l.onset_habituation_state_scan = observer.state.into();
            l.onset_habituation_known = true;
            l.onset_habituation_end = 500 * 512;
            histories.push(l);
        }
        for ablate in [false, true] {
            let mut exits = Vec::new();
            for history in &histories {
                let (_, mut f) = setup(1., false);
                let mut history = history.clone();
                if ablate {
                    std::sync::Arc::make_mut(&mut history.onset_habituation_state_scan).fill(0.);
                }
                let start = 500 * 512;
                render(&mut f, &history, start, true, true);
                f.span_gap.as_mut().unwrap().target = 0.001;
                for hop in 1..12 {
                    render(&mut f, &history, start + hop * 512, true, true);
                }
                exits.push(
                    f.span_gap
                        .as_ref()
                        .unwrap()
                        .trace
                        .iter()
                        .find(|t| t.state == SpanPhase::GapTail)
                        .map(|t| t.execution_at),
                );
            }
            if ablate {
                assert_eq!(exits, [None, None]);
            } else {
                assert!(exits[0].is_some());
                assert_eq!(exits[1], None);
            }
        }
    }

    #[test]
    fn gap_suppresses_all_births_keeps_ecology_and_returns_at_crossing_after_tails() {
        for active in [false, true] {
            for residual in [false, true] {
                let (l, mut f) = setup(1., residual);
                render(&mut f, &l, 0, active, true);
                assert_eq!(f.span_gap.as_ref().unwrap().phase, SpanPhase::Span);
                f.span_gap.as_mut().unwrap().target = 0.;
                let births = f.births;
                render(&mut f, &l, 512, active, false);
                assert_eq!(f.span_gap.as_ref().unwrap().phase, SpanPhase::Span);
                render(&mut f, &l, 1024, active, true);
                assert_eq!(f.span_gap.as_ref().unwrap().phase, SpanPhase::GapTail);
                assert!(f.lanes.iter().all(|l| l.mode != Mode::Current));
                let biomass = f.biomass_mass_scan.clone();
                let resource = f.resource_state_scan.clone();
                let mut returned = false;
                let mut quiet = false;
                for hop in 3..1500 {
                    f.advance(&l);
                    if f.span_gap.as_ref().unwrap().phase == SpanPhase::GapActive {
                        quiet = true;
                        f.span_gap.as_mut().unwrap().target = 0.0001;
                    }
                    let crossing = hop % 4 == 0;
                    let p = render(&mut f, &l, hop * 512, active, crossing);
                    let gap = f.span_gap.as_ref().unwrap();
                    if gap.phase == SpanPhase::Span {
                        assert!(quiet && crossing);
                        assert!(f.births > births);
                        assert_eq!(gap.trace.last().unwrap().execution_at, hop * 512 + 173);
                        returned = true;
                        break;
                    }
                    assert_eq!(f.births, births);
                    assert!(f.lanes.iter().all(|lane| lane.mode != Mode::Current));
                    if quiet {
                        assert!(p.iter().all(|&x| x == 0.));
                    }
                }
                assert!(returned);
                assert_ne!(f.biomass_mass_scan, biomass);
                assert_ne!(f.resource_state_scan, resource);
                let trace = &f.span_gap.as_ref().unwrap().trace;
                let tail = trace
                    .iter()
                    .find(|t| t.state == SpanPhase::GapTail)
                    .unwrap();
                let gap = trace
                    .iter()
                    .find(|t| t.state == SpanPhase::GapActive)
                    .unwrap();
                assert_eq!(tail.execution_at, 1024 + 173);
                assert!(gap.execution_at > tail.execution_at);
                assert_eq!(gap.return_integral, 0.);
            }
        }
    }

    #[test]
    fn weighted_heard_uses_sounding_tonal_target_power_and_empty_is_zero() {
        let (mut l, mut f) = setup(1., true);
        let split = l.space.n_bins() / 2;
        std::sync::Arc::make_mut(&mut l.onset_habituation_state_scan)[..split].fill(0.);
        std::sync::Arc::make_mut(&mut l.onset_habituation_state_scan)[split..].fill(1.);
        let gap = f.span_gap.as_mut().unwrap();
        gap.observe(&l, 0);
        gap.advance(0, false);
        gap.on(0, false);
        for (i, hz, target) in [(0, 300_f32, 0.25), (1, 700_f32, 0.75)] {
            f.lanes[i].mode = Mode::Current;
            f.lanes[i].r = 0.1;
            f.lanes[i].target = target;
            f.lanes[i].sounding_log2 = hz.log2();
        }
        f.observe_sounding_span_gap(0, 1.0);
        assert!((f.span_gap.as_ref().unwrap().heard() - 0.9).abs() < 1e-6);
        f.span_gap.as_mut().unwrap().advance(48000, false);
        let expected = f64::from(0.9_f32) * std::f64::consts::LN_10 / 5.;
        assert!((f.span_gap.as_ref().unwrap().exit_integral - expected).abs() < 1e-12);
        f.observe_sounding_span_gap(48000, 0.0);
        assert_eq!(f.span_gap.as_ref().unwrap().heard(), 0.);
        for lane in &mut f.lanes {
            lane.r = 0.;
        }
        f.observe_sounding_span_gap(48000, 1.0);
        assert_eq!(f.span_gap.as_ref().unwrap().heard(), 0.);
    }

    #[test]
    fn release_overrides_gap_and_reporting_does_not_change_pcm_or_targets() {
        let (l, mut a) = setup(1., true);
        let mut b = a.clone();
        b.span_gap.as_mut().unwrap().trace_enabled = false;
        for hop in 0..160 {
            if hop == 1 {
                a.span_gap.as_mut().unwrap().target = 0.;
                b.span_gap.as_mut().unwrap().target = 0.;
            }
            if hop == 120 {
                a.release(0.);
                b.release(0.);
            }
            assert_eq!(
                render(&mut a, &l, hop * 512, true, hop % 4 == 0),
                render(&mut b, &l, hop * 512, true, hop % 4 == 0)
            );
            assert_eq!(
                a.span_gap.as_ref().unwrap().target,
                b.span_gap.as_ref().unwrap().target
            );
        }
        assert_eq!(a.span_gap.as_ref().unwrap().phase, SpanPhase::Finished);
        assert!(a.is_finished());
        assert!(b.span_gap.as_ref().unwrap().trace.is_empty());
        assert!(setup(0., true).1.span_gap.is_none());
    }
}
