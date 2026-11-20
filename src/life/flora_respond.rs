//! Author rule: new heard onsets outside a Flora's own sounding partials speed its
//! ecological clock. Reads only the shared habitat onset history.
use super::*;
use crate::core::habituation::HabituationParams;
use crate::core::onset_habituation::SUPPORT_ERB;

#[derive(Clone, Debug, Serialize)]
pub(super) struct Respond {
    weight: f32,
    /// Cumulative per-observation maximum rise of the listened history.
    pub(super) rise_sum: f64,
    /// Cumulative extra clock seconds spent, before the beat-relative pace.
    pub(super) extra_sec: f64,
    pub(super) pending: f64,
    pub(super) listening_bins: u32,
    #[serde(skip)]
    tau: f64,
    #[serde(skip)]
    prev_scan: Vec<f32>,
    #[serde(skip)]
    own: Vec<bool>,
    #[serde(skip)]
    prev_end: u64,
    #[serde(skip)]
    primed: bool,
    #[serde(skip)]
    pub(super) factor: f32,
}

fn satiation_tau(params: HabituationParams) -> f64 {
    f64::from(params.satiation_sec) / std::f64::consts::LN_10
}

impl Respond {
    pub(super) fn new(weight: f32) -> Self {
        Self {
            weight,
            rise_sum: 0.0,
            extra_sec: 0.0,
            pending: 0.0,
            listening_bins: 0,
            tau: satiation_tau(HabituationParams::default()),
            prev_scan: Vec::new(),
            own: Vec::new(),
            prev_end: 0,
            primed: false,
            factor: 1.0,
        }
    }

    /// A grid change discards the previous history; the next known observation re-primes.
    pub(super) fn reset_grid(&mut self, bins: usize) {
        self.prev_scan.clear();
        self.prev_scan.resize(bins, 0.0);
        self.own.clear();
        self.own.resize(bins, false);
        self.primed = false;
    }
}

impl Flora {
    #[cfg(test)]
    pub(crate) fn fixed_lane_response_probe(&self, hz: f32) -> Self {
        let mut probe = self.clone();
        for lane in &mut probe.lanes {
            lane.mode = Mode::Free;
        }
        let lane = &mut probe.lanes[0];
        lane.mode = Mode::Current;
        lane.r = 0.1;
        lane.sounding_log2 = hz.log2();
        probe
    }

    pub(crate) fn configure_respond(&mut self, params: HabituationParams) {
        if let Some(respond) = &mut self.respond {
            respond.tau = satiation_tau(params);
        }
    }

    pub(crate) fn needs_onset_history(&self) -> bool {
        self.respond.is_some() || self.span_gap.is_some()
    }

    pub(crate) fn needs_external_onsets(&self) -> bool {
        self.respond.is_some() && self.needs_hr_observer()
    }

    /// Reads a new onset-history observation and sets this hop's clock factor.
    /// A rise rho owes w*tau*rho extra seconds, spent at most w*hop_dt per hop.
    pub(crate) fn observe_onsets(&mut self, landscape: &Landscape, now: u64) {
        let excluded = landscape.site_end.is_some()
            && self.rendered_samples != 0
            && self.needs_external_onsets();
        let Some(respond) = &mut self.respond else {
            return;
        };
        let (scan, end, known) = if excluded {
            landscape
                .flora_observations
                .iter()
                .find(|s| s.id == self.id && s.generation == 0)
                .and_then(|s| s.perception.as_ref())
                .and_then(|s| {
                    Some((
                        s.onset_habituation_state_scan.as_deref()?,
                        s.onset_habituation_end,
                        s.onset_habituation_known,
                    ))
                })
                .unwrap_or((&[], 0, false))
        } else {
            (
                landscape.onset_habituation_state_scan.as_ref(),
                landscape.onset_habituation_end,
                landscape.onset_habituation_known,
            )
        };
        if !known || end > now {
            respond.primed = false;
        } else if !respond.primed || end > respond.prev_end {
            self.space
                .assert_scan_len_named(scan, "onset_habituation_state_scan");
            if respond.primed {
                respond.own.fill(false);
                for lane in &self.lanes {
                    if lane.mode == Mode::Free || !(lane.tonal || lane.r > 0.0) {
                        continue;
                    }
                    let erb = hz_to_erb(lane.sounding_log2.exp2());
                    let lo = self.erb_scan.partition_point(|&x| x < erb - SUPPORT_ERB);
                    let hi = self.erb_scan.partition_point(|&x| x <= erb + SUPPORT_ERB);
                    respond.own[lo..hi].fill(true);
                }
                let mut rise = 0.0_f32;
                let mut bins = 0;
                for i in self.range.clone() {
                    if !respond.own[i] {
                        bins += 1;
                        rise = rise.max(scan[i] - respond.prev_scan[i]);
                    }
                }
                respond.listening_bins = bins;
                respond.rise_sum += f64::from(rise);
                respond.pending += f64::from(respond.weight) * respond.tau * f64::from(rise);
            }
            respond.prev_scan.copy_from_slice(scan);
            respond.prev_end = end;
            respond.primed = true;
        }
        let hop_dt = self.hop as f64 / f64::from(self.fs);
        let spend = respond.pending.min(f64::from(respond.weight) * hop_dt);
        respond.pending -= spend;
        respond.extra_sec += spend;
        respond.factor = (1.0 + spend / hop_dt) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::meter::{MeterBand, MeterState};
    use std::sync::Arc;

    const HOP: usize = 512;
    const FS: f32 = 48_000.0;

    fn setup(weight: f32) -> (Landscape, Flora) {
        let mut l = Landscape::new(Log2Space::new(110.0, 1760.0, 96));
        for i in 0..l.space.n_bins() {
            l.consonance_field_level[i] = 0.6 + 0.3 * (i as f32 * 0.07).sin();
            l.harmonicity01[i] = 0.4 + 0.3 * (i as f32 * 0.05).cos();
        }
        l.onset_habituation_state_scan = vec![0.1; l.space.n_bins()].into();
        l.onset_habituation_known = true;
        let mut f = Flora::new(
            1,
            FloraSpec {
                respond_to_onsets: weight,
                fade_sec: 0.0,
                pitch_apply_mode: PitchApplyMode::Rearticulate,
                ..Default::default()
            },
            (220.0, 880.0),
            FloraSeeding::Uniform,
            21,
            &l,
            FS,
            HOP,
        );
        f.external_observed_frame = Some(0);
        (l, f)
    }

    /// One sounding tonal lane at `hz`.
    fn sound(f: &mut Flora, hz: f32) {
        let lane = &mut f.lanes[0];
        lane.mode = Mode::Current;
        lane.r = 0.1;
        lane.target = 0.1;
        lane.sounding_log2 = hz.log2();
    }

    fn observe(l: &mut Landscape, end: u64, f: impl Fn(f32, f32) -> f32) {
        let hz = l.space.centers_hz.clone();
        for (x, hz) in Arc::make_mut(&mut l.onset_habituation_state_scan)
            .iter_mut()
            .zip(hz)
        {
            *x = f(*x, hz);
        }
        l.onset_habituation_end = end;
    }

    fn rhythm(f: &mut Flora, now: u64) {
        f.observe_rhythm(
            TemporalObservation {
                observed_end: Some(now),
                known: true,
                flux: 0.5,
                state: MeterState {
                    beat: MeterBand {
                        freq_hz: 2.0,
                        confidence: 1.0,
                        amplitude: 1.0,
                        phase: 0.0,
                    },
                    ..Default::default()
                },
            },
            now,
            Default::default(),
        );
    }

    fn respond(f: &Flora) -> &Respond {
        f.respond.as_ref().unwrap()
    }

    fn tau() -> f64 {
        5.0 / std::f64::consts::LN_10
    }

    #[test]
    fn zero_weight_has_no_state_and_matches_the_unobserved_path() {
        let (mut l, mut observed) = setup(0.0);
        assert!(observed.respond.is_none());
        let mut plain = observed.clone();
        let (_, mut flat) = setup(4.0);
        for hop in 0..400_u64 {
            let now = hop * HOP as u64;
            observe(&mut l, now, |x, hz| {
                ((x + 0.37 * (hz * 0.01 + hop as f32).sin().abs()) * 1.7).fract()
            });
            l.onset_habituation_known = hop % 7 != 3;
            observed.observe_onsets(&l, now);
            let mut still = l.clone();
            still.onset_habituation_state_scan = vec![0.1; l.space.n_bins()].into();
            flat.observe_onsets(&still, now);
            let mut out = Vec::new();
            for f in [&mut observed, &mut plain, &mut flat] {
                rhythm(f, now);
                f.advance(&l);
                let (mut p, mut h) = (vec![0.0; HOP], vec![0.0; HOP]);
                f.render_add(&mut p, &mut h);
                let targets: Vec<_> = f.lanes.iter().map(|l| (l.target, l.position)).collect();
                out.push((
                    f.pace.to_bits(),
                    p,
                    h,
                    targets,
                    f.rng,
                    f.biomass_mass_scan.clone(),
                ));
            }
            assert_eq!(out[0], out[1], "hop {hop}");
            // A positive weight without rises leaves the multiplier at exactly one.
            assert_eq!(out[1], out[2], "hop {hop}");
        }
        assert!(observed.births > 0);
        assert_eq!(respond(&flat).extra_sec, 0.0);
    }

    #[test]
    fn rises_outside_own_lanes_speed_the_clock_and_owe_w_tau_rho() {
        let w = 3.0;
        let (mut l, mut f) = setup(w);
        sound(&mut f, 330.0);
        let mut base = f.clone();
        base.respond = None;
        f.observe_onsets(&l, 0);
        let mut rises = 0.0_f64;
        for hop in 1..=40_u64 {
            let now = hop * HOP as u64;
            let rise = if hop <= 10 { 0.002 * hop as f32 } else { 0.0 };
            observe(&mut l, now, |x, hz| {
                if (580.0..620.0).contains(&hz) {
                    x + rise
                } else {
                    x
                }
            });
            f.observe_onsets(&l, now);
            rhythm(&mut f, now);
            rhythm(&mut base, now);
            if hop <= 10 {
                rises += f64::from(rise);
                assert!(f.pace > base.pace, "hop {hop}");
                assert!(respond(&f).factor > 1.0);
            } else if respond(&f).factor == 1.0 {
                assert_eq!(f.pace.to_bits(), base.pace.to_bits());
            }
            assert!(respond(&f).factor <= 1.0 + w + 1e-6);
        }
        let r = respond(&f);
        assert!((r.rise_sum - rises).abs() < 1e-6);
        assert_eq!(r.pending, 0.0);
        let expected = f64::from(w) * tau() * r.rise_sum;
        assert!(
            (r.extra_sec - expected).abs() < 1e-12,
            "{} {expected}",
            r.extra_sec
        );
        assert!(r.listening_bins > 0);
    }

    #[test]
    fn own_bins_out_of_range_bins_and_falls_do_not_respond() {
        let cases: [(&str, fn(f32, f32) -> f32); 3] = [
            ("own", |x, hz| {
                if (hz - 330.0).abs() < 25.0 {
                    x + 0.003
                } else {
                    x
                }
            }),
            ("outside", |x, hz| {
                if hz > 1000.0 || hz < 200.0 {
                    x + 0.003
                } else {
                    x
                }
            }),
            ("falling", |x, _| x - 0.003),
        ];
        for (name, change) in cases {
            let (mut l, mut f) = setup(2.0);
            sound(&mut f, 330.0);
            for hop in 0..30_u64 {
                observe(&mut l, hop * HOP as u64, change);
                f.observe_onsets(&l, hop * HOP as u64);
                assert_eq!(respond(&f).factor, 1.0, "{name} hop {hop}");
            }
            assert_eq!(respond(&f).extra_sec, 0.0, "{name}");
            assert_eq!(respond(&f).rise_sum, 0.0, "{name}");
        }
        // Without a sounding lane the whole extent listens.
        let (mut l, mut f) = setup(2.0);
        f.observe_onsets(&l, 0);
        observe(&mut l, HOP as u64, |x, hz| {
            if (hz - 330.0).abs() < 25.0 {
                x + 0.003
            } else {
                x
            }
        });
        f.observe_onsets(&l, HOP as u64);
        assert!(respond(&f).factor > 1.0);
    }

    #[test]
    fn unknown_future_and_repeated_observations_contribute_nothing_then_reprime() {
        let (mut l, mut f) = setup(2.0);
        let up = |x: f32, _| x + 0.003;
        f.observe_onsets(&l, 0);
        let h = HOP as u64;
        // Repeat of the same end, even with changed values.
        observe(&mut l, 0, up);
        f.observe_onsets(&l, h);
        assert_eq!(respond(&f).rise_sum, 0.0);
        // Unknown.
        observe(&mut l, 2 * h, up);
        l.onset_habituation_known = false;
        f.observe_onsets(&l, 2 * h);
        assert_eq!(respond(&f).rise_sum, 0.0);
        // The next known observation only re-primes.
        l.onset_habituation_known = true;
        observe(&mut l, 3 * h, up);
        f.observe_onsets(&l, 3 * h);
        assert_eq!(respond(&f).rise_sum, 0.0);
        // Future.
        observe(&mut l, 5 * h, up);
        f.observe_onsets(&l, 4 * h);
        assert_eq!(respond(&f).rise_sum, 0.0);
        observe(&mut l, 5 * h, up);
        f.observe_onsets(&l, 5 * h);
        assert_eq!(respond(&f).rise_sum, 0.0);
        observe(&mut l, 6 * h, up);
        f.observe_onsets(&l, 6 * h);
        assert!((respond(&f).rise_sum - 0.003).abs() < 1e-6);
        // A grid change discards the history while keeping the debt.
        observe(&mut l, 7 * h, up);
        f.on_space_change(&l.space.clone());
        let pending = respond(&f).pending;
        f.observe_onsets(&l, 7 * h);
        assert!((respond(&f).rise_sum - 0.003).abs() < 1e-6);
        assert!(respond(&f).pending <= pending);
    }

    #[test]
    fn skipped_observations_keep_the_total_and_bound_the_factor() {
        let w = 2.0;
        let (mut every, mut a) = setup(w);
        let mut sparse = every.clone();
        let mut b = a.clone();
        a.observe_onsets(&every, 0);
        b.observe_onsets(&sparse, 0);
        let mut spread = false;
        for hop in 1..=200_u64 {
            let now = hop * HOP as u64;
            let rise = if hop <= 40 { 0.004 } else { 0.0 };
            observe(&mut every, now, |x, _| x + rise);
            if hop % 4 == 0 {
                sparse = every.clone();
            }
            a.observe_onsets(&every, now);
            b.observe_onsets(&sparse, now);
            for f in [&a, &b] {
                assert!(respond(f).factor <= 1.0 + w + 1e-6);
            }
            spread |= hop % 4 != 0 && respond(&b).factor > 1.0;
        }
        assert!(spread, "sparse debt must spill into later hops");
        let (ra, rb) = (respond(&a), respond(&b));
        // f32 history differences round separately; the totals agree within that rounding.
        assert!((ra.rise_sum - rb.rise_sum).abs() < 1e-6);
        assert!((ra.extra_sec - rb.extra_sec).abs() < 1e-5);
        assert_eq!((ra.pending, rb.pending), (0.0, 0.0));
        for r in [ra, rb] {
            assert!((r.extra_sec - f64::from(w) * tau() * r.rise_sum).abs() < 1e-12);
        }
    }

    #[test]
    fn heard_births_away_from_own_lane_respond_and_births_at_it_do_not() {
        // Flora-shaped births (80 ms slew, hold, short tail) at a moss-like level over a
        // steady line-like harmonic tone. Loud sharp bursts at the own pitch do leak past
        // the 1.5 ERB exclusion; this checks the Flora's own attack shape and level.
        use crate::core::auditory_frontend::AuditoryFrontEnd;
        use crate::core::onset_habituation::OnsetHabituation;
        let (_, params) = crate::runtime::body_metabolism_test_core();
        let space = Log2Space::new(55.0, 8000.0, 96);
        let mut extra = Vec::new();
        for births_hz in [None, Some(330.0), Some(600.0)] {
            let mut front = AuditoryFrontEnd::selected(space.clone(), FS, HOP, 16384);
            let mut observer =
                OnsetHabituation::new(&space, front.band_energy().0, Default::default(), 0.5, FS);
            let mut l = Landscape::new(space.clone());
            let mut f = Flora::new(
                1,
                FloraSpec {
                    respond_to_onsets: 2.0,
                    ..Default::default()
                },
                (220.0, 880.0),
                FloraSeeding::Uniform,
                21,
                &l,
                FS,
                HOP,
            );
            sound(&mut f, 330.0);
            let mut max_factor = 0.0_f32;
            for hop in 0..800_u64 {
                let pcm: Vec<_> = (0..HOP as u64)
                    .map(|i| {
                        let t = (hop * HOP as u64 + i) as f64 / f64::from(FS);
                        let mut x: f64 = (1..=8)
                            .map(|k| {
                                0.03 / k as f64
                                    * (std::f64::consts::TAU * 196.0 * k as f64 * t).sin()
                            })
                            .sum();
                        if let Some(hz) = births_hz {
                            let u = (t - 1.0).rem_euclid(2.0);
                            let env = if t < 1.0 {
                                0.0
                            } else if u < 0.08 {
                                u / 0.08
                            } else if u < 1.3 {
                                1.0
                            } else {
                                (-(u - 1.3) / 0.1).exp()
                            };
                            x += 0.005 * env * (std::f64::consts::TAU * hz * t).sin();
                        }
                        x as f32
                    })
                    .collect();
                front.process(&pcm, &params);
                let end = (hop + 1) * HOP as u64;
                observer.process(front.band_energy().1, HOP, end);
                l.onset_habituation_state_scan = observer.state.clone().into();
                l.onset_habituation_known = observer.known();
                l.onset_habituation_end = end;
                f.observe_onsets(&l, end);
                if hop * HOP as u64 > 48_000 {
                    max_factor = max_factor.max(respond(&f).factor);
                }
            }
            println!(
                "respond acoustic births_hz={births_hz:?} max_factor_after_1s={max_factor} extra_sec={}",
                respond(&f).extra_sec
            );
            extra.push((respond(&f).extra_sec, max_factor));
        }
        let (background, own, other) = (extra[0], extra[1], extra[2]);
        // The background's own start is heard by every case alike.
        assert_eq!(background.1, 1.0);
        assert!(other.1 > 1.5);
        assert!(other.0 - background.0 > 1.0);
        assert!(own.0 - background.0 <= 0.05 * (other.0 - background.0));
    }

    #[test]
    fn reports_carry_respond_only_for_positive_weights_and_do_not_change_pcm() {
        for weight in [0.0, 2.0] {
            let (mut l, mut a) = setup(weight);
            let mut b = a.clone();
            for hop in 0..120_u64 {
                let now = hop * HOP as u64;
                observe(&mut l, now, |x, hz| {
                    if hop % 10 == 0 && hz > 500.0 {
                        x + 0.004
                    } else {
                        x * 0.999
                    }
                });
                let mut out = Vec::new();
                for (f, reporting) in [(&mut a, true), (&mut b, false)] {
                    f.observe_onsets(&l, now);
                    rhythm(f, now);
                    f.advance(&l);
                    if reporting {
                        let value = serde_json::to_value(f.report(0.0, &l)).unwrap();
                        let respond = value.get("respond");
                        assert_eq!(respond.is_some(), weight > 0.0);
                        if let Some(respond) = respond {
                            let keys: Vec<_> = respond.as_object().unwrap().keys().collect();
                            assert_eq!(
                                keys,
                                [
                                    "extra_sec",
                                    "listening_bins",
                                    "pending",
                                    "rise_sum",
                                    "weight"
                                ]
                            );
                        }
                    }
                    let (mut p, mut h) = (vec![0.0; HOP], vec![0.0; HOP]);
                    f.render_add(&mut p, &mut h);
                    out.push((p, f.lanes.iter().map(|l| l.target).collect::<Vec<_>>()));
                }
                assert_eq!(out[0], out[1]);
            }
        }
    }
}
