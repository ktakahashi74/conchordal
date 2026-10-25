//! Audio-only timing observations and bounded per-Flora lane opportunities.
use super::*;
use crate::core::meter::{MeterShaping, MeterState, sane_basin};
use crate::core::modulation::NeuralRhythms;
use crate::life::phonation_engine::{CandidatePoint, CouplingClock, MAX_ONSET_RATE_HZ};

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct TemporalObservation {
    pub(crate) observed_end: Option<u64>,
    pub(crate) known: bool,
    pub(crate) flux: f32,
    pub(crate) state: MeterState,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Pending {
    bin: usize,
    since: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct LaneEvent {
    tick: u64,
    scheduled_tick: Option<u64>,
    slot: usize,
    generation: u64,
    action: &'static str,
    reason: &'static str,
    wait_samples: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct MetricState {
    pub(super) observation: TemporalObservation,
    pub(super) now: u64,
    pub(super) active: bool,
    pub(super) pace: f32,
    pending: Vec<Option<Pending>>,
    pub(super) events: Vec<LaneEvent>,
    pub(super) drops: u64,
    local_phase_cycles: f64,
    #[serde(skip)]
    clock: CouplingClock,
    #[serde(skip)]
    initialized: bool,
    #[serde(skip)]
    seed: u64,
    #[serde(skip)]
    pub(super) crossings: Vec<CandidatePoint>,
    #[serde(skip)]
    reserved: Vec<bool>,
    #[serde(skip)]
    before: Vec<(Mode, u64, Option<u64>)>,
}

impl MetricState {
    pub(super) fn new(seed: u64, fs: f32, hop: usize) -> Self {
        let capacity =
            (hop as f64 / fs as f64 * MAX_ONSET_RATE_HZ as f64 * 1.5).ceil() as usize + 1;
        Self {
            observation: TemporalObservation::default(),
            now: 0,
            active: false,
            pace: 1.0,
            pending: Vec::new(),
            events: Vec::new(),
            drops: 0,
            local_phase_cycles: 0.0,
            clock: CouplingClock::new(0.95, 2.0, 0.0, 0.0, seed),
            initialized: false,
            seed,
            crossings: Vec::with_capacity(capacity),
            reserved: Vec::new(),
            before: Vec::new(),
        }
    }

    pub(super) fn resize(&mut self, cells: usize, lanes: usize, bins: usize) {
        self.drops += self.pending.iter().filter(|p| p.is_some()).count() as u64;
        self.pending = vec![None; cells];
        self.reserved = vec![false; bins];
        self.before = Vec::with_capacity(lanes);
        self.events = Vec::with_capacity(2 * lanes * (self.crossings.capacity() + 2));
    }
}

impl Flora {
    pub(crate) fn observe_rhythm(
        &mut self,
        observation: TemporalObservation,
        now: u64,
        shaping: MeterShaping,
    ) {
        self.acoustic_observation = observation;
        let base = shaping.basin_hz.map_or(2.0, |(lo, hi)| {
            let (lo, hi) = sane_basin(lo, hi);
            0.5 * (lo + hi)
        });
        self.pace = (base / 2.0) * self.spec.pace;
        let m = self.temporal.as_mut().unwrap();
        if !m.initialized {
            m.clock = CouplingClock::new(0.95, base, 0.0, 0.0, m.seed);
            m.initialized = true;
        }
        m.now = now;
        m.observation = observation;
        m.events.clear();
        m.crossings.clear();
        // The observation describes the immediately preceding audio interval.
        m.active = observation.known
            && observation.observed_end == Some(now)
            && observation.state.beat.confidence > 0.0
            && observation.state.beat.freq_hz.is_finite()
            && observation.state.beat.freq_hz > 0.0;
        let mut rhythms = NeuralRhythms::from_meter_state(&observation.state);
        if !m.active {
            rhythms.delta.alpha = 0.0;
        }
        m.clock.gather_rhythm(
            self.fs,
            now,
            now + self.hop as u64,
            &rhythms,
            &mut m.crossings,
        );
        // Use the same confidence-weighted velocity as the opportunity clock;
        // phase correction changes opportunities, not the ecological time scale.
        m.pace = self.fs / m.clock.nominal_period_ticks(self.fs).unwrap() as f32 / 2.0;
        self.pace = m.pace * self.spec.pace;
        m.local_phase_cycles = m.clock.phase_cycles();
        if !m.active {
            m.drops += m.pending.iter().filter(|p| p.is_some()).count() as u64;
            m.pending.fill(None);
            for lane in &mut self.lanes {
                lane.pending_since = None;
            }
        }
    }

    pub(super) fn metric_allocate(&mut self, crossing: bool, offset: usize) {
        let mut m = self.temporal.take().expect("metric state");
        let tick = m.now + offset as u64;
        m.before.clear();
        m.before.extend(
            self.lanes
                .iter()
                .map(|l| (l.mode, l.generation, l.pending_since)),
        );
        self.decompose();
        if self.releasing {
            for lane in &mut self.lanes {
                if lane.tonal && lane.mode == Mode::Current {
                    lane.forced_reason = Some("release");
                    lane.retire();
                    self.retirements += 1;
                }
            }
            for pending in &mut m.pending {
                m.drops += u64::from(pending.take().is_some());
            }
        } else if crossing || !m.active {
            self.update_tones();
        } else {
            // Existing W priority determines which continuing lane owns a candidate.
            self.order.clear();
            self.order.extend(
                (0..self.lanes.len())
                    .filter(|&i| self.lanes[i].tonal && self.lanes[i].mode == Mode::Current),
            );
            self.order.sort_unstable_by(|&a, &b| {
                self.lanes[b]
                    .target
                    .total_cmp(&self.lanes[a].target)
                    .then(a.cmp(&b))
            });
            for &i in &self.order {
                let lane = &mut self.lanes[i];
                if lane.position.exp2() < self.extent_hz.0
                    || lane.position.exp2() > self.extent_hz.1
                {
                    lane.forced_reason = Some("out_of_range");
                    lane.retire();
                    self.retirements += 1;
                    continue;
                }
                let nearest = self
                    .cells
                    .iter()
                    .enumerate()
                    .filter_map(|(c, cell)| {
                        let peak = cell.peak.as_ref()?;
                        let distance = (peak.position - lane.position).abs();
                        (peak.stay && peak.claimed.is_none() && distance <= 50.0 / 1200.0)
                            .then_some((c, distance, peak.bin))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)));
                if let Some((c, _, _)) = nearest {
                    let peak = self.cells[c].peak.as_mut().unwrap();
                    peak.claimed = Some(i);
                    lane.cell = Some(c);
                    lane.target = peak.mass.sqrt();
                    lane.target_position = peak.position;
                    lane.sigma = peak.sigma;
                    lane.window = peak.window.clone();
                    lane.pending_since = None;
                } else {
                    lane.pending_since.get_or_insert(tick);
                    lane.target_position = lane.position;
                    let mass: f32 = lane
                        .cell
                        .filter(|&c| c < self.cells.len())
                        .map_or(0.0, |c| {
                            lane.window
                                .clone()
                                .filter(|&b| self.cells[c].bins.contains(&b))
                                .map(|b| (self.biomass_mass_scan[b] - self.cells[c].floor).max(0.0))
                                .sum()
                        });
                    lane.target = mass.sqrt();
                    if mass == 0.0
                        || !self.range.contains(&lane.window.start)
                        || lane.window.end > self.range.end
                    {
                        lane.forced_reason = Some(if mass == 0.0 {
                            "empty_window"
                        } else {
                            "out_of_range"
                        });
                        lane.retire();
                        self.retirements += 1;
                    }
                }
            }
            // Separation is a support constraint, including retained old windows.
            loop {
                self.order.retain(|&i| self.lanes[i].mode == Mode::Current);
                self.order.sort_unstable_by(|&a, &b| {
                    self.lanes[a]
                        .target_position
                        .total_cmp(&self.lanes[b].target_position)
                        .then(a.cmp(&b))
                });
                let conflict = self.order.windows(2).find_map(|pair| {
                    let (a, b) = (pair[0], pair[1]);
                    let (pa, pb) = (&self.lanes[a], &self.lanes[b]);
                    ((hz_to_erb(pb.target_position.exp2()) - hz_to_erb(pa.target_position.exp2()))
                        < 0.5)
                        .then_some(
                            if pa.target < pb.target || (pa.target == pb.target && a > b) {
                                a
                            } else {
                                b
                            },
                        )
                });
                let Some(i) = conflict else { break };
                if let Some(c) = self.lanes[i].cell
                    && let Some(p) = self.cells[c].peak.as_mut().filter(|p| p.claimed == Some(i))
                {
                    p.claimed = None;
                    p.suppressed = true;
                }
                self.lanes[i].forced_reason = Some("separation");
                self.lanes[i].retire();
                self.retirements += 1;
            }
        }
        // Reserve each bin once. A retained window never receives stale mass and
        // never shares its current excess with another tone or the residual.
        m.reserved.fill(false);
        for cell in &mut self.cells {
            cell.tonal_mass = 0.0;
        }
        for lane in &mut self.lanes {
            if !lane.tonal || lane.mode != Mode::Current {
                continue;
            }
            let Some(c) = lane.cell else { continue };
            let mut mass = 0.0;
            for b in lane
                .window
                .clone()
                .filter(|&b| self.cells[c].bins.contains(&b))
            {
                if !m.reserved[b] {
                    m.reserved[b] = true;
                    mass += (self.biomass_mass_scan[b] - self.cells[c].floor).max(0.0);
                }
            }
            self.cells[c].tonal_mass += mass;
            lane.target = mass.sqrt();
            if mass == 0.0 {
                lane.forced_reason = Some("empty_window");
                lane.retire();
                self.retirements += 1;
            }
        }
        for (c, cell) in self.cells.iter_mut().enumerate() {
            cell.residual_mass = (cell.excess - cell.tonal_mass).max(0.0);
            let lane = &mut self.lanes[self.tonal_len + c];
            lane.target = if self.spec.residual {
                (cell.residual_mass - DEAD_MASS).max(0.0).sqrt()
            } else {
                0.0
            };
            let mut moment = 0.0;
            let mut mass = 0.0;
            for b in cell.bins.clone().filter(|&b| !m.reserved[b]) {
                let x = self.biomass_mass_scan[b] - cell.floor;
                moment += x * self.space.centers_log2[b];
                mass += x;
            }
            if mass > 0.0 {
                lane.target_position = (moment / mass)
                    .clamp(erb_to_hz(cell.lo_erb).log2(), erb_to_hz(cell.hi_erb).log2());
            }
        }
        let scale = self
            .lanes
            .iter()
            .map(|l| l.target.powi(2))
            .sum::<f32>()
            .max(1.0)
            .sqrt()
            .recip();
        for lane in &mut self.lanes {
            lane.target *= scale;
        }
        for (slot, lane) in self.lanes.iter().enumerate().filter(|(_, l)| l.tonal) {
            let (mode, generation, since) = m.before[slot];
            let birth = lane.generation != generation;
            let retirement = mode == Mode::Current && lane.mode == Mode::Retiring;
            if birth || retirement {
                let since = if birth {
                    lane.cell
                        .and_then(|c| m.pending[c].as_ref().map(|p| p.since))
                } else {
                    since
                };
                let forced = retirement && (self.releasing || (m.active && !crossing));
                m.events.push(LaneEvent {
                    tick,
                    scheduled_tick: crossing.then_some(tick),
                    slot,
                    generation: lane.generation,
                    action: if birth { "born" } else { "retiring" },
                    reason: if self.releasing {
                        "release"
                    } else if forced {
                        lane.forced_reason.unwrap_or("support_lost")
                    } else if crossing {
                        "crossing"
                    } else {
                        "fallback"
                    },
                    wait_samples: tick.saturating_sub(since.unwrap_or(tick)),
                });
            }
        }
        for (c, cell) in self.cells.iter().enumerate() {
            let candidate = (m.active && !self.releasing)
                .then_some(())
                .and(cell.peak.as_ref())
                .filter(|p| p.enter && p.claimed.is_none() && !p.suppressed);
            match (m.pending[c].as_ref(), candidate) {
                (Some(old), Some(new)) if old.bin == new.bin => {}
                (old, new) => {
                    if old.is_some() && !crossing {
                        m.drops += 1;
                    }
                    m.pending[c] = new.map(|p| Pending {
                        bin: p.bin,
                        since: tick,
                    });
                }
            }
        }
        self.temporal = Some(m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::meter::MeterBand;

    fn setup(tone: FloraTone, residual: bool) -> Flora {
        let landscape = Landscape::new(Log2Space::new(220., 880., 96));
        Flora::new(
            1,
            FloraSpec {
                tone,
                residual,
                fade_sec: 0.,
                ..Default::default()
            },
            (220., 880.),
            FloraSeeding::Uniform,
            21,
            &landscape,
            48000.,
            512,
        )
    }

    fn observation(now: u64, known: bool) -> TemporalObservation {
        TemporalObservation {
            observed_end: Some(now),
            known,
            flux: 0.5,
            state: MeterState {
                beat: MeterBand {
                    phase: 0.,
                    freq_hz: 2.,
                    confidence: 0.9,
                    amplitude: 1.,
                },
                ..Default::default()
            },
        }
    }

    fn hill(f: &mut Flora, bin: usize, mass: f32) {
        for (i, x) in f.biomass_mass_scan.iter_mut().enumerate() {
            *x = 0.001 + mass * (-0.5 * ((i as f32 - bin as f32) / 1.5).powi(2)).exp();
        }
    }

    #[test]
    fn metric_pace_uses_basin_confidence_and_freshness() {
        let mut f = setup(FloraTone::Sine, false);
        let shaping = MeterShaping {
            basin_hz: Some((3.5, 4.5)),
            ..Default::default()
        };
        f.observe_rhythm(observation(0, false), 0, shaping);
        assert!((f.temporal.as_ref().unwrap().pace - 1.875).abs() < 1e-6);
        let mut heard = observation(512, true);
        heard.state.beat.freq_hz = 2.0;
        heard.state.beat.confidence = 1.0;
        f.observe_rhythm(heard, 512, shaping);
        assert!((f.temporal.as_ref().unwrap().pace - 1.04375).abs() < 1e-6);
        f.observe_rhythm(heard, 1024, shaping);
        assert!((f.temporal.as_ref().unwrap().pace - 1.875).abs() < 1e-6);
        heard.observed_end = Some(1536);
        heard.state.beat.confidence = 0.0;
        f.observe_rhythm(heard, 1536, shaping);
        assert!((f.temporal.as_ref().unwrap().pace - 1.875).abs() < 1e-6);
    }

    #[test]
    fn fallback_pace_uses_the_basin_and_preserves_dynamics_and_tail() {
        for basin in [None, Some((0.5, 1.5)), Some((2.0, 4.0))] {
            for tone in [FloraTone::Sine, FloraTone::Resonant] {
                let shaping = MeterShaping {
                    basin_hz: basin,
                    ..Default::default()
                };
                let mut unknown = setup(tone, false);
                let mut unconfirmed = unknown.clone();
                let mut obs = observation(0, true);
                obs.state.beat.confidence = 0.0;
                unconfirmed.observe_rhythm(obs, 0, shaping);
                unknown.observe_rhythm(observation(0, false), 0, shaping);
                assert!(
                    (unconfirmed.pace - basin.map_or(1.0, |(lo, hi)| (lo + hi) / 4.0)).abs() < 1e-6
                );
                assert_eq!(unconfirmed.pace, unknown.pace);
                let mut landscape = Landscape::new(unconfirmed.space.clone());
                landscape.consonance_field_level.fill(0.7);
                landscape.harmonicity01.fill(0.4);
                unconfirmed.external_observed_frame = Some(0);
                unknown.external_observed_frame = Some(0);
                unconfirmed.advance(&landscape);
                unknown.advance(&landscape);
                assert_eq!(unconfirmed.biomass_mass_scan, unknown.biomass_mass_scan);
                assert_eq!(unconfirmed.resource_state_scan, unknown.resource_state_scan);
                let mut pcm = vec![0.0; 512];
                let mut habitat = pcm.clone();
                hill(&mut unconfirmed, 60, 1.0);
                unconfirmed.render_add(&mut pcm, &mut habitat);
                let slot = unconfirmed
                    .lanes
                    .iter()
                    .position(|l| l.tonal && l.mode == Mode::Current)
                    .unwrap();
                unconfirmed.release(2.0);
                unconfirmed.render_add(&mut pcm, &mut habitat);
                assert_eq!(unconfirmed.lanes[slot].mode, Mode::Retiring);
                assert!(unconfirmed.lanes[slot].retirement_decay > 0.0);
                let mut hops = 1;
                while !unconfirmed.is_finished() && hops <= 200 {
                    unconfirmed.render_add(&mut pcm, &mut habitat);
                    hops += 1;
                }
                assert!(unconfirmed.is_finished());
                assert!(hops as f32 * 512.0 / 48000.0 <= 2.0 + 512.0 / 48000.0);
            }
        }
    }

    #[test]
    fn flora_basin_uses_the_same_supported_band_as_metric_voices() {
        for (basin, bounds, expected_pace) in [
            ((4.0, 1.0), (1.0, 4.0), 1.25),
            ((-1.0, 20.0), (0.5, 4.0), 1.125),
            ((f32::NAN, f32::INFINITY), (0.5, 4.0), 1.125),
            ((1.0, 1.0), (0.95, 1.05), 0.5),
            ((0.5, 0.5), (0.5, 0.55), 0.2625),
            ((4.0, 4.0), (3.95, 4.0), 1.9875),
        ] {
            assert_eq!(sane_basin(basin.0, basin.1), bounds);
            {
                let mut f = setup(FloraTone::Sine, false);
                f.observe_rhythm(
                    TemporalObservation::default(),
                    0,
                    MeterShaping {
                        basin_hz: Some(basin),
                        ..Default::default()
                    },
                );
                // One rounding step through the clock's period representation is allowed.
                assert!((f.pace - expected_pace).abs() <= f32::EPSILON * expected_pace);
            }
        }
    }

    #[test]
    fn metabolic_retirement_is_continuous_finite_and_preserves_power() {
        for noisy in [false, true] {
            let mut lane = Lane::new(noisy, 8.0);
            lane.mode = Mode::Current;
            lane.r = 0.2;
            lane.q = lane.r * lane.r;
            lane.retirement_decay = (-0.5_f32 * 8.0 * METABOLISM / 48000.0).exp();
            lane.retire();
            let mut previous = lane.r;
            let mut samples = 0;
            while lane.mode != Mode::Free && samples < 48000 * 3 {
                lane.slew(48000.0);
                assert!(lane.r >= 0.0 && lane.r <= previous);
                assert!((previous - lane.r) < 1.0 / (SLEW_SEC * 48000.0));
                assert_eq!(lane.q, lane.r * lane.r);
                previous = lane.r;
                samples += 1;
            }
            assert_eq!(lane.mode, Mode::Free);
            assert!(samples > 48000 / 2);
            let expected = ((0.2_f64 + (DEAD_MASS as f64).sqrt()) / (DEAD_MASS as f64).sqrt()).ln()
                / -(lane.retirement_decay as f64).ln();
            // The finite-zero deadline must remain within one existing 512-sample hop.
            assert!((samples as f64 - expected).abs() < 512.0);
        }
    }

    #[test]
    fn flora_metric_birth_is_inside_hop_and_dead_candidates_never_sound() {
        let mut f = setup(FloraTone::Sine, false);
        let bin = (f.cells[3].bins.start + f.cells[3].bins.end) / 2;
        hill(&mut f, bin, 0.1);
        f.observe_rhythm(observation(0, true), 0, Default::default());
        f.temporal.as_mut().unwrap().crossings.clear();
        let mut p = vec![0.; 512];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        assert_eq!(f.births, 0);
        assert!(
            f.temporal
                .as_ref()
                .unwrap()
                .pending
                .iter()
                .any(Option::is_some)
        );
        f.observe_rhythm(observation(512, true), 512, Default::default());
        f.temporal.as_mut().unwrap().crossings.clear();
        f.temporal.as_mut().unwrap().crossings.push(CandidatePoint {
            tick: 512 + 173,
            gate: 0,
        });
        f.render_add(&mut p, &mut h);
        assert_eq!(f.births, 1);
        assert!(p[..173].iter().all(|&x| x == 0.));
        assert!(p[173..].iter().any(|&x| x != 0.));
        let event = &f.temporal.as_ref().unwrap().events[0];
        assert_eq!((event.tick, event.wait_samples), (685, 685));
        assert_eq!(event.scheduled_tick, Some(685));
        assert_eq!(
            f.lanes
                .iter()
                .find(|l| l.mode == Mode::Current && l.tonal)
                .unwrap()
                .age_samples,
            339
        );
        let mut dead = setup(FloraTone::Sine, false);
        hill(&mut dead, bin, 0.1);
        dead.observe_rhythm(observation(0, true), 0, Default::default());
        dead.temporal.as_mut().unwrap().crossings.clear();
        dead.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
        dead.biomass_mass_scan.fill(0.001);
        dead.observe_rhythm(observation(512, true), 512, Default::default());
        dead.temporal
            .as_mut()
            .unwrap()
            .crossings
            .push(CandidatePoint { tick: 700, gate: 0 });
        dead.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
        assert_eq!(dead.births, 0);
        assert!(dead.temporal.as_ref().unwrap().drops > 0);
    }

    #[test]
    fn flora_metric_retained_window_reserves_current_mass_and_forces_empty_retirement() {
        let mut f = setup(FloraTone::Resonant, true);
        let bin = (f.cells[3].bins.start + f.cells[3].bins.end) / 2;
        hill(&mut f, bin, 0.1);
        f.observe_rhythm(observation(0, false), 0, Default::default());
        f.metric_allocate(false, 0);
        let slot = f
            .lanes
            .iter()
            .position(|l| l.tonal && l.mode == Mode::Current)
            .unwrap();
        let window = f.lanes[slot].window.clone();
        let c = f.lanes[slot].cell.unwrap();
        // A taller distant candidate replaces the winner; the old window still has excess.
        let far = (f.cells[c].bins.end - 1).max(bin + 5).min(f.range.end - 1);
        f.biomass_mass_scan[far] = 0.5;
        f.observe_rhythm(observation(512, true), 512, Default::default());
        f.temporal.as_mut().unwrap().crossings.clear();
        f.metric_allocate(false, 0);
        assert_eq!(f.lanes[slot].mode, Mode::Current);
        assert_eq!(f.lanes[slot].window, window);
        let reserved = f
            .temporal
            .as_ref()
            .unwrap()
            .reserved
            .iter()
            .filter(|&&b| b)
            .count();
        assert!(reserved > 0);
        for cell in &f.cells {
            assert!((cell.excess - cell.tonal_mass - cell.residual_mass).abs() < 1e-6);
        }
        f.biomass_mass_scan.fill(0.001);
        f.metric_allocate(false, 0);
        assert_eq!(f.lanes[slot].mode, Mode::Retiring);
        assert_eq!(f.lanes[slot].target, 0.);
        assert_eq!(f.lanes[slot].forced_reason, Some("empty_window"));
    }

    #[test]
    fn flora_metric_unknown_fallback_revalidates_and_keeps_private_clock() {
        let mut f = setup(FloraTone::Sine, false);
        let bin = f.cells[3].bins.start + 1;
        hill(&mut f, bin, 0.1);
        f.observe_rhythm(observation(0, true), 0, Default::default());
        f.temporal.as_mut().unwrap().crossings.clear();
        f.metric_allocate(false, 0);
        f.biomass_mass_scan.fill(0.001);
        f.observe_rhythm(observation(512, false), 512, Default::default());
        let phase = f.temporal.as_ref().unwrap().clock.phase_cycles();
        f.metric_allocate(false, 0);
        assert_eq!(f.births, 0);
        f.observe_rhythm(observation(1024, true), 1024, Default::default());
        assert!(f.temporal.as_ref().unwrap().clock.phase_cycles() > phase);
        f.release(0.01);
        f.metric_allocate(false, 0);
        assert!(
            f.temporal
                .as_ref()
                .unwrap()
                .pending
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn supported_observation_transitions_preserve_phase_and_revalidate_pending() {
        for unavailable in ["unknown", "stale", "unconfirmed"] {
            let mut f = setup(FloraTone::Sine, true);
            let bin = (f.cells[3].bins.start + f.cells[3].bins.end) / 2;
            hill(&mut f, bin, 0.1);
            f.observe_rhythm(observation(0, true), 0, Default::default());
            f.temporal.as_mut().unwrap().crossings.clear();
            f.metric_allocate(false, 0);
            assert!(
                f.temporal
                    .as_ref()
                    .unwrap()
                    .pending
                    .iter()
                    .any(Option::is_some)
            );
            assert_eq!(f.births, 0);
            let mut expected_clock = f.temporal.as_ref().unwrap().clock.clone();
            let mut expected_crossings = Vec::new();
            let mut obs = observation(512, true);
            match unavailable {
                "unknown" => obs.known = false,
                "stale" => obs.observed_end = Some(0),
                _ => obs.state.beat.confidence = 0.0,
            }
            // Keep the hill and acoustic support; only observation availability changes.
            let mut rhythms = NeuralRhythms::from_meter_state(&obs.state);
            rhythms.delta.alpha = 0.0;
            expected_clock.gather_rhythm(f.fs, 512, 1024, &rhythms, &mut expected_crossings);
            hill(&mut f, bin, 0.04);
            f.observe_rhythm(obs, 512, Default::default());
            let m = f.temporal.as_ref().unwrap();
            assert!(!m.active);
            assert_eq!(m.clock.phase_cycles(), expected_clock.phase_cycles());
            assert!(m.pending.iter().all(Option::is_none));
            assert!(m.drops > 0);
            f.metric_allocate(false, 0);
            assert_eq!(f.births, 1, "{unavailable}");
            let event = &f.temporal.as_ref().unwrap().events[0];
            assert_eq!(
                (event.tick, event.wait_samples, event.reason),
                (512, 0, "fallback")
            );
            let slot = f
                .lanes
                .iter()
                .position(|l| l.tonal && l.mode == Mode::Current)
                .unwrap();
            let mass: f32 = f.lanes[slot]
                .window
                .clone()
                .map(|b| {
                    (f.biomass_mass_scan[b] - f.cells[f.lanes[slot].cell.unwrap()].floor).max(0.0)
                })
                .sum();
            assert!((f.lanes[slot].target.powi(2) - mass).abs() < 1e-6);
            let obs = observation(1024, true);
            expected_clock.gather_rhythm(
                f.fs,
                1024,
                1536,
                &NeuralRhythms::from_meter_state(&obs.state),
                &mut expected_crossings,
            );
            f.observe_rhythm(obs, 1024, Default::default());
            let m = f.temporal.as_ref().unwrap();
            assert!(m.active);
            assert_eq!(m.clock.phase_cycles(), expected_clock.phase_cycles());
            f.metric_allocate(true, 173);
            assert_eq!(f.births, 1, "stale pending sounded after {unavailable}");
            assert!(
                f.temporal
                    .as_ref()
                    .unwrap()
                    .pending
                    .iter()
                    .all(Option::is_none)
            );
            for cell in &f.cells {
                assert!((cell.excess - cell.tonal_mass - cell.residual_mass).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn flora_metric_release_discards_pending_on_active_beats() {
        let mut violations = Vec::new();
        for fade_sec in [0.05, 0.0] {
            let mut f = setup(FloraTone::Sine, false);
            let bin = (f.cells[3].bins.start + f.cells[3].bins.end) / 2;
            hill(&mut f, bin, 0.1);
            f.observe_rhythm(observation(0, true), 0, Default::default());
            f.temporal.as_mut().unwrap().crossings.clear();
            f.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
            let m = f.temporal.as_ref().unwrap();
            let pending = m.pending.iter().filter(|p| p.is_some()).count() as u64;
            let expected_drops = m.drops + pending;
            assert!(m.active && pending > 0);
            assert!(
                f.cells
                    .iter()
                    .any(|c| c.peak.as_ref().is_some_and(|p| p.enter))
            );
            assert_eq!(f.births, 0);
            f.release(fade_sec);
            for hop in 1..=3 {
                let now = hop * 512;
                f.observe_rhythm(observation(now, true), now, Default::default());
                let m = f.temporal.as_mut().unwrap();
                m.crossings.clear();
                m.crossings.push(CandidatePoint {
                    tick: now + 173,
                    gate: 0,
                });
                f.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
                let m = f.temporal.as_ref().unwrap();
                let remaining = m.pending.iter().filter(|p| p.is_some()).count();
                if remaining != 0 || m.drops != expected_drops {
                    violations.push((fade_sec, hop, remaining, m.drops, expected_drops));
                }
                assert_eq!(f.births, 0);
                if hop == 1 && fade_sec > 0.0 {
                    assert!(!f.is_finished());
                }
            }
        }
        assert!(
            violations.is_empty(),
            "release must cancel pending once and never recreate it: {violations:?}"
        );
    }

    #[test]
    fn flora_metric_zero_fade_release_records_immediate_retirement() {
        let mut f = setup(FloraTone::Sine, false);
        hill(&mut f, 35, 0.1);
        f.observe_rhythm(observation(0, false), 0, Default::default());
        f.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
        assert!(f.births > 0);
        f.release(0.0);
        assert!(!f.is_finished());
        f.observe_rhythm(observation(512, true), 512, Default::default());
        let mut pcm = vec![0.; 512];
        f.render_add(&mut pcm, &mut vec![0.; 512]);
        assert!(f.is_finished());
        assert!(pcm.iter().all(|&x| x == 0.));
        assert!(
            f.temporal
                .as_ref()
                .unwrap()
                .events
                .iter()
                .any(|e| e.reason == "release" && e.tick == 512 && e.scheduled_tick.is_none())
        );
    }

    #[test]
    fn flora_metric_saved_candidate_and_rhythm_replay() {
        let rows: Vec<(usize, f32, f32, f32)> = serde_json::from_str(include_str!(
            "../../tests/fixtures/flora_metric_replay.json"
        ))
        .unwrap();
        let mut coupled = setup(FloraTone::Sine, true);
        let mut fallback = coupled.clone();
        let mut pcm0 = vec![0.; 512];
        let mut pcm1 = pcm0.clone();
        let mut habitat = pcm0.clone();
        let mut records = Vec::new();
        let mut waits = 0;
        let mut forced = 0;
        let mut different = 0;
        for (frame, (bin, height, phase, confidence)) in rows.into_iter().enumerate() {
            let now = frame as u64 * 512;
            hill(&mut fallback, bin, height);
            hill(&mut coupled, bin, height);
            let mut obs = observation(now, true);
            obs.state.beat.phase = phase;
            obs.state.beat.confidence = confidence;
            coupled.observe_rhythm(obs, now, Default::default());
            obs.state.beat.confidence = 0.0;
            fallback.observe_rhythm(obs, now, Default::default());
            pcm0.fill(0.);
            pcm1.fill(0.);
            habitat.fill(0.);
            fallback.render_add(&mut pcm0, &mut habitat);
            habitat.fill(0.);
            coupled.render_add(&mut pcm1, &mut habitat);
            assert_eq!(fallback.biomass_mass_scan, coupled.biomass_mass_scan);
            assert_eq!(fallback.resource_state_scan, coupled.resource_state_scan);
            let m = coupled.temporal.as_ref().unwrap();
            for e in &m.events {
                if m.active && e.action == "born" {
                    assert_eq!(e.scheduled_tick, Some(e.tick));
                }
                waits += usize::from(e.wait_samples > 0);
                forced += usize::from(e.action == "retiring" && e.scheduled_tick.is_none());
            }
            different += usize::from(pcm0 != pcm1);
            for cell in &coupled.cells {
                assert!((cell.excess - cell.tonal_mass - cell.residual_mass).abs() < 1e-6);
            }
            records.push(
                serde_json::json!({"frame":frame,"fallback_births":fallback.births,"coupled_births":coupled.births,
                "events":m.events,"drops":m.drops,"active":m.active,
                "fallback_power":pcm0.iter().map(|x|(*x as f64).powi(2)).sum::<f64>()/512.,
                "coupled_power":pcm1.iter().map(|x|(*x as f64).powi(2)).sum::<f64>()/512.}),
            );
        }
        assert!(
            waits > 0 && forced > 0 && different > 0,
            "waits={waits} forced={forced} different={different}"
        );
        println!(
            "F11_REPLAY hops=400 waits={waits} forced={forced} different_pcm_hops={different} drops={}",
            coupled.temporal.as_ref().unwrap().drops
        );
        if let Ok(path) = std::env::var("F11_REPLAY_OUTPUT") {
            std::fs::write(path, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
        }
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn flora_metric_turnover_with_crossings_has_zero_allocations() {
        use crate::runtime_profile::{begin_allocations, finish_allocations};
        for tone in [FloraTone::Sine, FloraTone::Resonant] {
            let mut f = setup(tone, true);
            let bin = f.cells[3].bins.start + 1;
            let mut p = vec![0.; 512];
            let mut h = p.clone();
            begin_allocations();
            for frame in 0..300 {
                let now = frame * 512;
                hill(
                    &mut f,
                    bin + ((frame / 20) % 3) as usize * 8,
                    if frame % 20 < 15 { 0.1 } else { 0. },
                );
                f.observe_rhythm(observation(now, true), now, Default::default());
                let m = f.temporal.as_mut().unwrap();
                m.crossings.clear();
                if frame % 5 == 0 {
                    m.crossings.push(CandidatePoint {
                        tick: now + 173,
                        gate: frame,
                    });
                }
                f.render_add(&mut p, &mut h);
            }
            let allocations = finish_allocations().unwrap();
            assert_eq!(allocations.count, 0, "{allocations:?}");
            assert_eq!(allocations.bytes, 0, "{allocations:?}");
            assert!(f.births > 0 && f.retirements > 0);
            println!(
                "F11_METRIC_ALLOC tone={tone:?} {allocations:?} births={} retirements={}",
                f.births, f.retirements
            );
        }
    }
}
