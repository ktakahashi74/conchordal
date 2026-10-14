//! Continuous native T2 recurrence and future confirmation.
use super::*;
use serde::{Deserialize, Serialize};

const CHANNELS: usize = 30;
type Values = [[f64; 4]; CHANNELS];
type Mask = [[bool; 4]; CHANNELS];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Packet {
    pub bus: u64,
    pub config_epoch: u64,
    pub feature_units: u64,
    pub fs: u32,
    pub hop: u64,
    pub sequence: u64,
    pub start: u64,
    pub end: u64,
    pub reset_origin: u64,
    pub available: u64,
    pub received: u64,
    pub means: Values,
    pub known: Mask,
    pub dropped: Option<[u64; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Identity {
    bus: u64,
    epoch: u64,
    units: u64,
}

#[derive(Clone, Copy, Debug)]
struct Row {
    end: Option<u64>,
    origin: u64,
    available: u64,
    y: Values,
    y_known: Mask,
    x: Values,
    known: Mask,
}

impl Default for Row {
    fn default() -> Self {
        Self {
            end: None,
            origin: 0,
            available: 0,
            y: [[0.0; 4]; CHANNELS],
            y_known: [[false; 4]; CHANNELS],
            x: [[0.0; 4]; CHANNELS],
            known: [[false; 4]; CHANNELS],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Sum {
    value: f64,
    error: f64,
}

impl Sum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.error;
        let next = self.value + corrected;
        self.error = (next - self.value) - corrected;
        self.value = next;
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Mass {
    i: f64,
    u: f64,
    pairs: u64,
    i_positive: u64,
    u_positive: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Lag {
    i: Sum,
    u: Sum,
    pairs: u64,
    i_positive: u64,
    u_positive: u64,
}

impl Lag {
    fn update(&mut self, mass: Mass, dt: f64, adding: bool, work: &mut Work) {
        let sign = if adding { 1.0 } else { -1.0 };
        self.i.add(sign * dt * mass.i);
        self.u.add(sign * dt * mass.u);
        work.mass_arithmetic += 12;
        if adding {
            self.pairs += mass.pairs;
            self.i_positive += mass.i_positive;
            self.u_positive += mass.u_positive;
        } else {
            self.pairs -= mass.pairs;
            self.i_positive -= mass.i_positive;
            self.u_positive -= mass.u_positive;
        }
        if self.i_positive == 0 {
            self.i = Sum::default();
        }
        if self.u_positive == 0 {
            self.u = Sum::default();
        }
    }

    fn score(&self, work: &mut Work) -> Option<f64> {
        work.score_reads += 1;
        if self.u_positive == 0 {
            return None;
        }
        assert!(
            self.u.value > 0.0 && self.u.value.is_finite(),
            "nonpositive known union"
        );
        Some((self.i.value / self.u.value).clamp(0.0, 1.0))
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Work {
    pub pair_dimensions: u64,
    pub interpolation_dimensions: u64,
    pub forecast_dimensions: u64,
    pub score_reads: u64,
    pub mass_arithmetic: u64,
    pub quadratic_vertices: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Proposal {
    pub lag_hops: f64,
    pub lag_sec: f64,
    pub score: f64,
    pub pairs: u64,
    pub source_cut: u64,
    pub available: u64,
    pub received: u64,
    pub use_cut: u64,
}

#[derive(Clone, Copy, Debug)]
struct Forecast {
    target: Option<u64>,
    issued: u64,
    source_origin: Option<u64>,
    x: Values,
    known: Mask,
}

impl Default for Forecast {
    fn default() -> Self {
        Self {
            target: None,
            issued: 0,
            source_origin: None,
            x: [[0.0; 4]; CHANNELS],
            known: [[false; 4]; CHANNELS],
        }
    }
}

#[derive(Debug)]
pub(crate) struct State {
    fs: u32,
    hop: u64,
    key: Option<Identity>,
    rows: Vec<Row>,
    forecasts: Vec<Forecast>,
    lags: Vec<Lag>,
    k_min: usize,
    oldest: Option<u64>,
    newest: Option<u64>,
    pub fast_start: u64,
    pub fast_end: u64,
    pub pending: Option<Proposal>,
    pub seed_cut: Option<u64>,
    retained: Option<f64>,
    ratio_support: f32,
    matched: f64,
    predicted: f64,
    observed: f64,
    last_packet: Option<Packet>,
    pub suppressed: bool,
    source_known: bool,
    known_coordinates: usize,
    pub work: Work,
    resets: u64,
    duplicates: u64,
    raw_gaps: u64,
    expired_rows: u64,
    expired_pending: u64,
    expired_forecasts: u64,
    issued: u64,
    scored: u64,
    unknown_targets: u64,
    last_issued_target: Option<u64>,
}

impl State {
    pub(super) fn sample_rate(&self) -> u32 {
        self.fs
    }
    pub(super) fn accept(&mut self, ratio: f32) {
        self.retained = self.pending.map(|p| p.lag_hops);
        self.ratio_support = ratio;
    }
    pub(super) fn seed(&mut self) {
        self.seed_cut = Some(self.fast_end);
        self.reset_evidence();
    }
    pub(super) fn confidence(&self, beat_plv: f32, presence: f32) -> f32 {
        let support = self
            .support()
            .map_or(beat_plv, |support| beat_plv.max(support));
        (support * presence).clamp(0.0, 1.0)
    }
    fn new(fs: u32, hop: u64, suppressed: bool) -> Self {
        assert!(fs > 0 && hop > 0);
        let dt = hop as f64 / fs as f64;
        let capacity = (RECURRENCE_HISTORY_SECONDS / dt).ceil() as usize + 4 + 2;
        let k_min = ((1.0 / (4.0 * F_BEAT_MAX as f64)) / dt).ceil() as usize;
        let k_max = ((1.0 / F_BEAT_MIN as f64) / dt).floor() as usize;
        assert!(k_min > 0 && k_max >= k_min && k_max < capacity);
        Self {
            fs,
            hop,
            key: None,
            rows: vec![Row::default(); capacity],
            forecasts: vec![Forecast::default(); capacity],
            lags: vec![Lag::default(); k_max - k_min + 1],
            k_min,
            oldest: None,
            newest: None,
            fast_start: 0,
            fast_end: 0,
            pending: None,
            seed_cut: None,
            retained: None,
            ratio_support: 0.0,
            matched: 0.0,
            predicted: 0.0,
            observed: 0.0,
            last_packet: None,
            suppressed,
            source_known: false,
            known_coordinates: 0,
            work: Work::default(),
            resets: 0,
            duplicates: 0,
            raw_gaps: 0,
            expired_rows: 0,
            expired_pending: 0,
            expired_forecasts: 0,
            issued: 0,
            scored: 0,
            unknown_targets: 0,
            last_issued_target: None,
        }
    }

    fn index(&self, end: u64) -> usize {
        ((end / self.hop) % self.rows.len() as u64) as usize
    }

    fn row(&self, end: u64) -> Option<&Row> {
        let row = &self.rows[self.index(end)];
        (row.end == Some(end)).then_some(row)
    }

    fn reset_coordinate(&mut self) {
        self.rows.fill(Row::default());
        self.forecasts.fill(Forecast::default());
        self.lags.fill(Lag::default());
        self.oldest = None;
        self.newest = None;
        self.pending = None;
        self.seed_cut = None;
        self.retained = None;
        self.ratio_support = 0.0;
        self.reset_evidence();
        self.last_packet = None;
        self.source_known = false;
        self.known_coordinates = 0;
        self.last_issued_target = None;
        self.resets += 1;
    }

    fn reset_evidence(&mut self) {
        self.matched = 0.0;
        self.predicted = 0.0;
        self.observed = 0.0;
        self.forecasts.fill(Forecast::default());
    }

    fn pair(a: &Row, b: &Row, work: &mut Work) -> Mass {
        if a.origin != b.origin {
            return Mass::default();
        }
        let mut i = Sum::default();
        let mut u = Sum::default();
        let mut mass = Mass::default();
        for ch in 0..CHANNELS {
            for d in 0..4 {
                work.pair_dimensions += 1;
                if !a.known[ch][d] || !b.known[ch][d] {
                    continue;
                }
                let low = a.x[ch][d].min(b.x[ch][d]);
                let high = a.x[ch][d].max(b.x[ch][d]);
                i.add(low);
                u.add(high);
                work.mass_arithmetic += 8;
                mass.pairs += 1;
                mass.i_positive += u64::from(low > 0.0);
                mass.u_positive += u64::from(high > 0.0);
            }
        }
        mass.i = i.value;
        mass.u = u.value;
        mass
    }

    fn expire(&mut self, cut: u64) {
        let horizon = RECURRENCE_HISTORY_SECONDS * self.fs as f64;
        let mut work = Work::default();
        while let Some(end) = self.oldest {
            if cut.saturating_sub(end) as f64 <= horizon {
                break;
            }
            if let Some(row) = self.row(end).copied() {
                for index in 0..self.lags.len() {
                    let right_end = end + (self.k_min + index) as u64 * self.hop;
                    if let Some(right) = self.row(right_end) {
                        let mass = Self::pair(&row, right, &mut work);
                        self.lags[index].update(
                            mass,
                            self.hop as f64 / self.fs as f64,
                            false,
                            &mut work,
                        );
                    }
                }
                let index = self.index(end);
                self.rows[index] = Row::default();
                self.expired_rows += 1;
            }
            self.oldest = if self.newest.is_some_and(|n| end < n) {
                Some(end + self.hop)
            } else {
                None
            };
        }
        if self
            .pending
            .is_some_and(|p| cut.saturating_sub(p.source_cut) as f64 > horizon)
        {
            self.pending = None;
            self.expired_pending += 1;
        }
        for forecast in &mut self.forecasts {
            if forecast
                .target
                .is_some_and(|t| cut.saturating_sub(t) as f64 > horizon)
            {
                *forecast = Forecast::default();
                self.expired_forecasts += 1;
            }
        }
        self.add_work(work);
    }

    fn add_work(&mut self, w: Work) {
        self.work.pair_dimensions += w.pair_dimensions;
        self.work.interpolation_dimensions += w.interpolation_dimensions;
        self.work.forecast_dimensions += w.forecast_dimensions;
        self.work.score_reads += w.score_reads;
        self.work.mass_arithmetic += w.mass_arithmetic;
        self.work.quadratic_vertices += w.quadratic_vertices;
    }

    fn sample(&self, target_hops: f64, use_cut: u64, origin: Option<u64>, work: &mut Work) -> Row {
        let mut out = Row::default();
        if target_hops < 1.0 || !target_hops.is_finite() {
            return out;
        }
        let left_hops = target_hops.floor() as u64;
        let fraction = target_hops - left_hops as f64;
        let left_end = left_hops * self.hop;
        let Some(left) = self.row(left_end) else {
            return out;
        };
        if left.available > use_cut || origin.is_some_and(|o| o != left.origin) {
            return out;
        }
        out.origin = left.origin;
        if fraction == 0.0 {
            out.x = left.x;
            out.known = left.known;
            return out;
        }
        let Some(right) = self.row(left_end + self.hop) else {
            return out;
        };
        if right.origin != left.origin || right.available > use_cut {
            return out;
        }
        for ch in 0..CHANNELS {
            for d in 0..4 {
                work.interpolation_dimensions += 1;
                if left.known[ch][d] && right.known[ch][d] {
                    out.x[ch][d] = left.x[ch][d] + (right.x[ch][d] - left.x[ch][d]) * fraction;
                    out.known[ch][d] = true;
                    work.mass_arithmetic += 3;
                }
            }
        }
        out
    }

    fn score_at(
        &self,
        lag_hops: f64,
        use_cut: u64,
        latest: u64,
        work: &mut Work,
    ) -> (Option<f64>, Mass, bool) {
        let mut total = Mass::default();
        let mut intersection = Sum::default();
        let mut union = Sum::default();
        let mut fresh = false;
        for right in &self.rows {
            let Some(end) = right.end else {
                continue;
            };
            let target = end as f64 / self.hop as f64 - lag_hops;
            if target * (self.hop as f64)
                < use_cut as f64 - RECURRENCE_HISTORY_SECONDS * self.fs as f64
            {
                continue;
            }
            let left = self.sample(target, use_cut, Some(right.origin), work);
            let mass = Self::pair(right, &left, work);
            intersection.add(mass.i);
            union.add(mass.u);
            work.mass_arithmetic += 8;
            total.pairs += mass.pairs;
            total.i_positive += mass.i_positive;
            total.u_positive += mass.u_positive;
            fresh |= end == latest && mass.i_positive > 0;
        }
        total.i = intersection.value;
        total.u = union.value;
        let score = (total.u_positive > 0).then(|| (total.i / total.u).clamp(0.0, 1.0));
        (score, total, fresh)
    }

    // Preserve the fixed prototype's evaluation and branching order.
    #[allow(clippy::collapsible_if)]
    fn select(&mut self, packet: &Packet) {
        let mut work = Work::default();
        let mut best = None::<(usize, f64)>;
        let mut low = None::<f64>;
        let mut high = None::<f64>;
        for (index, lag) in self.lags.iter().enumerate() {
            let Some(score) = lag.score(&mut work) else {
                continue;
            };
            low = Some(low.map_or(score, |v| v.min(score)));
            high = Some(high.map_or(score, |v| v.max(score)));
            if lag.i_positive > 0 && best.is_none_or(|(_, v)| score > v) {
                best = Some((index, score));
            }
        }
        if low == high {
            self.add_work(work);
            return;
        }
        let Some((index, mut score)) = best else {
            self.add_work(work);
            return;
        };
        let mut hops = (self.k_min + index) as f64;
        let lag = &self.lags[index];
        let mut pairs = lag.pairs;
        let fresh_mass = packet
            .end
            .checked_sub(hops as u64 * self.hop)
            .and_then(|left| self.row(left).zip(self.row(packet.end)))
            .map(|(a, b)| Self::pair(a, b, &mut work))
            .unwrap_or_default();
        let mut fresh = fresh_mass.i_positive > 0;
        if index > 0 && index + 1 < self.lags.len() {
            if let (Some(a), Some(b)) = (
                self.lags[index - 1].score(&mut work),
                self.lags[index + 1].score(&mut work),
            ) {
                let curvature = a - 2.0 * score + b;
                if score > a && score > b && curvature < 0.0 {
                    let delta = (a - b) / (2.0 * curvature);
                    let candidate = hops + delta;
                    work.quadratic_vertices += 1;
                    work.mass_arithmetic += 7;
                    let p = candidate * self.hop as f64 / self.fs as f64;
                    if p.is_finite()
                        && (1.0 / (4.0 * F_BEAT_MAX as f64)..=1.0 / F_BEAT_MIN as f64).contains(&p)
                    {
                        let (actual, mass, current) =
                            self.score_at(candidate, self.fast_end, packet.end, &mut work);
                        if let Some(value) = actual {
                            if mass.i_positive > 0 && value > score {
                                hops = candidate;
                                score = value;
                                pairs = mass.pairs;
                                fresh = current;
                            }
                        }
                    }
                }
            }
        }
        if let Some(retained) = self.retained {
            let (actual, mass, current) =
                self.score_at(retained, self.fast_end, packet.end, &mut work);
            if let Some(value) = actual {
                if mass.i_positive > 0 && value >= score {
                    hops = retained;
                    score = value;
                    pairs = mass.pairs;
                    fresh = current;
                }
            }
        }
        if fresh && !self.suppressed {
            self.pending = Some(Proposal {
                lag_hops: hops,
                lag_sec: hops * self.hop as f64 / self.fs as f64,
                score,
                pairs,
                source_cut: packet.end,
                available: packet.available,
                received: packet.received,
                use_cut: self.fast_end,
            });
        }
        self.add_work(work);
    }

    // Keep the original packet validation and coordinate updates together.
    #[allow(clippy::collapsible_if, clippy::manual_is_multiple_of)]
    fn ingest(&mut self, p: Packet) -> Result<(), &'static str> {
        if p.fs != self.fs || p.hop != self.hop {
            return Err("operator geometry requires reconstruction before the hop");
        }
        if p.start.checked_add(p.hop) != Some(p.end)
            || p.start % p.hop != 0
            || p.sequence != p.start / p.hop
            || p.end > p.available
            || p.available > p.received
            || p.received > self.fast_end
            || p.reset_origin > p.start
        {
            return Err("invalid packet clock");
        }
        if p.means.iter().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite native means");
        }
        if self.last_packet == Some(p) {
            self.duplicates += 1;
            return Ok(());
        }
        if self
            .last_packet
            .is_some_and(|last| p.sequence <= last.sequence || p.received < last.received)
        {
            return Err("nonmonotone source packet");
        }
        if self.fast_end.saturating_sub(p.end) as f64 > RECURRENCE_HISTORY_SECONDS * self.fs as f64
        {
            self.expired_rows += 1;
            self.last_packet = Some(p);
            return Ok(());
        }
        let mut row = Row {
            end: Some(p.end),
            origin: p.reset_origin,
            available: p.available,
            y: p.means,
            y_known: p.known,
            ..Default::default()
        };
        if let Some(previous) = p
            .end
            .checked_sub(self.hop)
            .and_then(|end| self.row(end))
            .copied()
        {
            if previous.origin == p.reset_origin {
                for ch in 0..CHANNELS {
                    for d in 0..4 {
                        if previous.y_known[ch][d] && p.known[ch][d] {
                            row.x[ch][d] = (p.means[ch][d] - previous.y[ch][d]).max(0.0);
                            if !row.x[ch][d].is_finite() {
                                return Err("nonfinite positive rise");
                            }
                            row.known[ch][d] = true;
                            self.work.mass_arithmetic += 1;
                        }
                    }
                }
            }
        }
        self.raw_gaps += u64::from(
            p.dropped.is_some() || self.last_packet.is_some_and(|last| p.start != last.end),
        );
        let index = self.index(p.end);
        assert!(self.rows[index].end.is_none(), "bounded history overwrite");
        self.rows[index] = row;
        self.oldest = self.oldest.or(Some(p.end));
        self.newest = Some(p.end);
        let mut work = Work::default();
        for index in 0..self.lags.len() {
            let Some(left_end) = p.end.checked_sub((self.k_min + index) as u64 * self.hop) else {
                continue;
            };
            if let Some(left) = self.row(left_end) {
                let mass = Self::pair(&row, left, &mut work);
                self.lags[index].update(mass, self.hop as f64 / self.fs as f64, true, &mut work);
            }
        }
        self.add_work(work);
        self.known_coordinates = row.known.iter().flatten().filter(|v| **v).count();
        self.source_known = self.known_coordinates > 0;
        let index = self.index(p.end);
        let forecast = self.forecasts[index];
        if forecast.target == Some(p.end) {
            let mut m = Sum::default();
            let mut q = Sum::default();
            let mut o = Sum::default();
            let mut compared = 0;
            if self.seed_cut.is_some_and(|s| p.start >= s)
                && forecast.issued <= p.start
                && forecast.source_origin == Some(p.reset_origin)
            {
                for ch in 0..CHANNELS {
                    for d in 0..4 {
                        self.work.forecast_dimensions += 1;
                        if forecast.known[ch][d] && row.known[ch][d] {
                            m.add(forecast.x[ch][d].min(row.x[ch][d]));
                            q.add(forecast.x[ch][d]);
                            o.add(row.x[ch][d]);
                            compared += 1;
                            self.work.mass_arithmetic += 12;
                        }
                    }
                }
            }
            if compared > 0 {
                let weight = (self.hop as f64 / self.fs as f64)
                    * (-((self.fast_end - p.end) as f64) / self.fs as f64 / PERSIST_TAU as f64)
                        .exp();
                self.matched += weight * m.value;
                self.predicted += weight * q.value;
                self.observed += weight * o.value;
                self.scored += 1;
            } else {
                self.unknown_targets += 1;
            }
            self.forecasts[index] = Forecast::default();
        }
        self.last_packet = Some(p);
        self.select(&p);
        Ok(())
    }

    fn issue(&mut self) {
        let Some(lag) = self.retained else {
            return;
        };
        if self.seed_cut.is_none() {
            return;
        }
        let start = self.fast_end.div_ceil(self.hop) * self.hop;
        let target = start + self.hop;
        if self.last_issued_target.is_some_and(|t| t >= target) {
            return;
        }
        let mut work = Work::default();
        let template = self.sample(
            target as f64 / self.hop as f64 - lag,
            self.fast_end,
            None,
            &mut work,
        );
        let index = self.index(target);
        assert!(
            self.forecasts[index].target.is_none(),
            "bounded forecast overwrite"
        );
        self.forecasts[index] = Forecast {
            target: Some(target),
            issued: self.fast_end,
            source_origin: template
                .known
                .iter()
                .flatten()
                .any(|v| *v)
                .then_some(template.origin),
            x: template.x,
            known: template.known,
        };
        self.last_issued_target = Some(target);
        self.issued += 1;
        self.add_work(work);
    }

    fn support(&self) -> Option<f32> {
        if self.suppressed {
            return None;
        }
        let union = self.predicted + self.observed - self.matched;
        (union > 0.0).then(|| (self.matched / union).clamp(0.0, 1.0) as f32 * self.ratio_support)
    }
}

impl MeterNetwork {
    #[cfg(test)]
    pub(crate) fn enable_continuous(&mut self, fs: u32, hop: u64, suppressed: bool) {
        if self.continuous().is_some() {
            let shaping = self.shaping;
            *self = Self::new();
            self.set_shaping(shaping);
        }
        self.recurrence = RecurrenceInput::Continuous(Box::new(State::new(fs, hop, suppressed)));
    }

    pub(crate) fn continuous_fast(
        &mut self,
        start: u64,
        end: u64,
        dt: f32,
        drive: Option<f32>,
    ) -> MeterState {
        let state = self.continuous_mut().expect("continuous mode");
        assert!(start == state.fast_end && end > start);
        state.work = Work::default();
        state.fast_start = start;
        state.fast_end = end;
        state.source_known = false;
        state.known_coordinates = 0;
        let decay = (-((end - start) as f64) / state.fs as f64 / PERSIST_TAU as f64).exp();
        state.matched *= decay;
        state.predicted *= decay;
        state.observed *= decay;
        state.expire(end);
        self.process_evidence(dt, drive)
    }

    pub(crate) fn continuous_ingest(&mut self, p: Packet) -> Result<(), &'static str> {
        let RecurrenceInput::Continuous(state) = &mut self.recurrence else {
            return Err("continuous mode absent");
        };
        let key = Identity {
            bus: p.bus,
            epoch: p.config_epoch,
            units: p.feature_units,
        };
        if state.key.is_some_and(|old| old != key) {
            state.reset_coordinate();
            self.beat_phi = 0.0;
            self.beat_omega = TAU * F_BEAT_INIT;
            self.beat_r = 0.1;
            self.onset_detector = OnsetDetector::default();
            self.onset_age = None;
            self.pending_period = None;
            self.seeded = false;
            self.previous_phase_error = None;
            self.plv_count = 0.0;
            self.beat_re = 0.0;
            self.beat_im = 0.0;
            self.sub_re = [0.0; 3];
            self.sub_im = [0.0; 3];
            self.beat_cycles = 0.0;
            self.strength_baseline = 0.0;
            self.meas_re = [0.0; 3];
            self.meas_im = [0.0; 3];
            self.meas_norm = 0.0;
            self.attention_ema = 0.0;
            self.last = MeterState::default();
            if let Some((lo, hi)) = self.shaping.basin_hz {
                let (lo, hi) = sane_basin(lo, hi);
                self.beat_omega = TAU * 0.5 * (lo + hi);
            }
        }
        state.key = Some(key);
        state.ingest(p)
    }

    pub(crate) fn continuous_finish(&mut self) -> MeterState {
        self.last = self.build_state();
        self.continuous_mut().expect("continuous mode").issue();
        self.last
    }

    pub(crate) fn continuous_diagnostics(&self) -> Option<Diagnostics> {
        let s = self.continuous()?;
        Some(Diagnostics {
            source_known: s.source_known,
            known_coordinates: s.known_coordinates,
            seeded: self.seeded,
            seed_cut: s.seed_cut,
            pending: s.pending,
            matched: s.matched,
            predicted: s.predicted,
            observed: s.observed,
            recurrence_support: s.support(),
            issued: s.issued,
            scored: s.scored,
            unknown_targets: s.unknown_targets,
            duplicates: s.duplicates,
            raw_gaps: s.raw_gaps,
            resets: s.resets,
            expired_rows: s.expired_rows,
            expired_pending: s.expired_pending,
            expired_forecasts: s.expired_forecasts,
            work: s.work,
        })
    }
    pub(crate) fn new_continuous(fs: u32, hop: u64) -> Self {
        Self::with_input(RecurrenceInput::Continuous(Box::new(State::new(
            fs, hop, false,
        ))))
    }
    pub(crate) fn continuous_geometry(fs: u32, hop: u64) -> Result<(), &'static str> {
        if fs == 0 || hop == 0 {
            return Err("invalid continuous meter geometry");
        }
        let dt = hop as f64 / fs as f64;
        let k_min = ((1.0 / (4.0 * F_BEAT_MAX as f64)) / dt).ceil() as usize;
        let k_max = ((1.0 / F_BEAT_MIN as f64) / dt).floor() as usize;
        if k_min == 0 || k_max < k_min {
            return Err("source hop cannot represent the native lag band");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet(j: u64, value: f64) -> Packet {
        Packet {
            bus: 0,
            config_epoch: 0,
            feature_units: 0,
            fs: 48000,
            hop: 512,
            sequence: j,
            start: j * 512,
            end: (j + 1) * 512,
            reset_origin: 0,
            available: (j + 1) * 512,
            received: (j + 1) * 512,
            means: [[value; 4]; CHANNELS],
            known: [[true; 4]; CHANNELS],
            dropped: None,
        }
    }
    fn step(m: &mut MeterNetwork, j: u64, value: f64) {
        m.continuous_fast(j * 512, (j + 1) * 512, 512.0 / 48000.0, Some(0.0));
        m.continuous_ingest(packet(j, value)).unwrap();
        m.continuous_finish();
    }
    #[test]
    fn constant_known_input_does_not_induce_a_beat() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        for j in 0..512 {
            step(&mut m, j, 1.0);
        }
        assert!(!m.seeded);
        assert!(m.continuous().unwrap().pending.is_none());
    }
    #[test]
    fn unknown_is_not_a_zero_or_a_negative_example() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        step(&mut m, 0, 0.0);
        step(&mut m, 1, 1.0);
        let s = m.continuous_mut().unwrap();
        s.seed_cut = Some(0);
        s.retained = Some(1.0);
        s.ratio_support = 1.0;
        s.issue();
        m.continuous_fast(1024, 1536, 512.0 / 48000.0, None);
        let mut p = packet(2, 0.0);
        p.known = [[false; 4]; CHANNELS];
        m.continuous_ingest(p).unwrap();
        m.continuous_finish();
        let s = m.continuous().unwrap();
        assert_eq!((s.matched, s.predicted, s.observed), (0.0, 0.0, 0.0));
        assert_eq!(s.unknown_targets, 1);
        assert_eq!(s.scored, 0);
    }
    #[test]
    fn partial_target_uses_the_same_known_coordinates_for_all_masses() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        step(&mut m, 0, 0.0);
        step(&mut m, 1, 1.0);
        let s = m.continuous_mut().unwrap();
        s.seed_cut = Some(0);
        s.retained = Some(1.0);
        s.ratio_support = 1.0;
        s.issue();
        m.continuous_fast(1024, 1536, 512.0 / 48000.0, None);
        let mut p = packet(2, 2.0);
        p.known[0] = [false; 4];
        m.continuous_ingest(p).unwrap();
        let s = m.continuous().unwrap();
        let mass = 116.0 * (512.0 / 48000.0);
        assert_eq!((s.matched, s.predicted, s.observed), (mass, mass, mass));
        assert_eq!(s.known_coordinates, 116);
        assert_eq!(s.scored, 1);
    }
    #[test]
    fn gap_preserves_history_pending_and_coordinate() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        step(&mut m, 0, 0.0);
        step(&mut m, 1, 1.0);
        m.seeded = true;
        m.continuous_mut().unwrap().pending = Some(Proposal {
            lag_hops: 47.0,
            lag_sec: 47.0 * 512.0 / 48000.0,
            score: 1.0,
            pairs: 120,
            source_cut: 1024,
            available: 1024,
            received: 1024,
            use_cut: 1024,
        });
        m.continuous_fast(1024, 1536, 512.0 / 48000.0, None);
        m.continuous_finish();
        assert!(m.seeded);
        assert!(m.continuous().unwrap().row(1024).is_some());
        assert!(m.continuous().unwrap().pending.is_some());
        m.continuous_fast(1536, 2048, 512.0 / 48000.0, None);
        let mut p = packet(3, 1.0);
        p.reset_origin = p.start;
        p.dropped = Some([1024, 1536]);
        m.continuous_ingest(p).unwrap();
        m.continuous_finish();
        assert!(m.seeded);
        assert_eq!(m.continuous().unwrap().resets, 0);
    }
    #[test]
    fn target_is_issued_and_scored_once() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        step(&mut m, 0, 0.0);
        step(&mut m, 1, 1.0);
        let s = m.continuous_mut().unwrap();
        s.seed_cut = Some(0);
        s.retained = Some(1.0);
        s.ratio_support = 1.0;
        s.issue();
        s.issue();
        assert_eq!(s.issued, 1);
        m.continuous_fast(1024, 1536, 512.0 / 48000.0, None);
        let p = packet(2, 2.0);
        m.continuous_ingest(p).unwrap();
        m.continuous_ingest(p).unwrap();
        assert_eq!(m.continuous().unwrap().scored, 1);
        assert_eq!(m.continuous().unwrap().duplicates, 1);
    }
    #[test]
    fn past_targets_are_never_issued_retroactively() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        step(&mut m, 0, 0.0);
        step(&mut m, 1, 1.0);
        let s = m.continuous_mut().unwrap();
        s.seed_cut = Some(0);
        s.retained = Some(1.0);
        s.issue();
        assert!(
            s.forecasts
                .iter()
                .all(|f| f.target.is_none_or(|t| t > s.fast_end))
        );
    }
    #[test]
    fn first_seed_cannot_certify_itself() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        let dt = 512.0 / 48000.0;
        for j in 0..512 {
            m.continuous_fast(
                j * 512,
                (j + 1) * 512,
                dt,
                Some(if j % 47 == 0 { 1.0 } else { 0.0 }),
            );
            let value = if j % 47 == 0 { 1.0 } else { 0.0 };
            m.continuous_ingest(packet(j, value)).unwrap();
            let state = m.continuous_finish();
            if m.continuous().unwrap().seed_cut == Some((j + 1) * 512) {
                assert_eq!(state.beat.confidence, 0.0);
                return;
            }
        }
        panic!("deterministic recurrence did not seed");
    }
    #[test]
    fn configuration_epoch_initializes_an_unknown_coordinate() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        step(&mut m, 0, 0.0);
        step(&mut m, 1, 1.0);
        m.seeded = true;
        m.continuous_fast(1024, 1536, 512.0 / 48000.0, None);
        let mut p = packet(2, 1.0);
        p.config_epoch = 1;
        m.continuous_ingest(p).unwrap();
        assert!(!m.seeded);
        assert_eq!(m.continuous().unwrap().resets, 1);
        assert!(m.continuous().unwrap().row(1024).is_none());
    }
    #[test]
    fn operator_geometry_reconstruction_does_not_keep_an_old_beat() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        step(&mut m, 0, 0.0);
        step(&mut m, 1, 1.0);
        m.seeded = true;
        m.enable_continuous(2 * 48000, 2 * 512, false);
        assert!(!m.seeded);
        assert!(m.continuous().unwrap().oldest.is_none());
    }
    #[test]
    fn rolling_lags_match_direct_overlap_with_missing_rows() {
        let mut m = MeterNetwork::new();
        m.enable_continuous(48000, 512, false);
        for j in 0..512 {
            m.continuous_fast(j * 512, (j + 1) * 512, 512.0 / 48000.0, None);
            let mut p = packet(j, if j % 47 == 0 { 1.0 } else { 0.0 });
            if j % 47 == 1 {
                p.known = [[false; 4]; CHANNELS];
            }
            m.continuous_ingest(p).unwrap();
        }
        let s = m.continuous().unwrap();
        for (index, lag) in s.lags.iter().enumerate() {
            let (_, direct, _) = s.score_at(
                (s.k_min + index) as f64,
                s.fast_end,
                s.fast_end,
                &mut Work::default(),
            );
            assert_eq!(lag.pairs, direct.pairs);
            assert_eq!(lag.i_positive, direct.i_positive);
            assert_eq!(lag.u_positive, direct.u_positive);
            let dt = s.hop as f64 / s.fs as f64;
            let n = (s.rows.len() * CHANNELS * 4) as f64;
            let gamma = (n * f64::EPSILON) / (1.0 - n * f64::EPSILON);
            assert!(
                (lag.i.value - dt * direct.i).abs()
                    <= gamma * (lag.i.value.abs() + dt * direct.i.abs())
            );
            assert!(
                (lag.u.value - dt * direct.u).abs()
                    <= gamma * (lag.u.value.abs() + dt * direct.u.abs())
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Diagnostics {
    pub source_known: bool,
    pub known_coordinates: usize,
    pub seeded: bool,
    pub seed_cut: Option<u64>,
    pub pending: Option<Proposal>,
    pub matched: f64,
    pub predicted: f64,
    pub observed: f64,
    pub recurrence_support: Option<f32>,
    pub issued: u64,
    pub scored: u64,
    pub unknown_targets: u64,
    pub duplicates: u64,
    pub raw_gaps: u64,
    pub resets: u64,
    pub expired_rows: u64,
    pub expired_pending: u64,
    pub expired_forecasts: u64,
    pub work: Work,
}
