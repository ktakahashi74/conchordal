//! The same fixed tone-energy prior for issued commands and conditional actions.

use super::ScheduledRelease;
use crate::life::sound::{control_forecast::ControlForecast, envelope::Envelope};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ToneEnergy {
    pub amplitude: f32,
    pub envelope: Envelope,
    pub control: Option<ControlForecast>,
    pub scheduled_release: Option<ScheduledRelease>,
    pub sine: Option<crate::life::sound::sine_forecast::SineForecast>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bank: Option<crate::life::sound::bank_forecast::BankForecast>,
}

impl ToneEnergy {
    pub(crate) fn control_for(
        self,
        intervention: Option<ScheduledRelease>,
    ) -> Option<ControlForecast> {
        let mut control = self.control?;
        // Due updates run before the queue is cleared, even on a late release.
        let mut clear_at = self.envelope.hold_end.max(control.issued_at);
        for release in [self.scheduled_release, intervention].into_iter().flatten() {
            clear_at = clear_at.min(
                release
                    .apply_at_sample
                    .max(release.off_sample)
                    .max(control.issued_at),
            );
        }
        control.valid_until = control.valid_until.filter(|at| *at <= clear_at);
        if let Some(updates) = control.amplitude_updates.as_mut() {
            updates.len = updates.events[..updates.len]
                .iter()
                .take_while(|e| e.at_sample.max(control.issued_at) <= clear_at)
                .count();
        }
        Some(control)
    }

    fn envelope_at(self, tick: u64, intervention: Option<ScheduledRelease>) -> Envelope {
        let mut envelope = self.envelope;
        for release in [self.scheduled_release, intervention].into_iter().flatten() {
            if tick >= release.apply_at_sample {
                envelope = envelope.with_release(release.off_sample);
            }
        }
        envelope
    }

    pub(crate) fn renderer_end_after(
        self,
        after: u64,
        intervention: Option<ScheduledRelease>,
    ) -> Option<u64> {
        let mut edges = [0, u64::MAX, 0, 0];
        for (edge, release) in edges[2..]
            .iter_mut()
            .zip([self.scheduled_release, intervention])
        {
            *edge = release.map_or(0, |r| r.apply_at_sample);
        }
        edges.sort_unstable();
        for pair in edges.windows(2) {
            let end = self
                .envelope_at(pair[0], intervention)
                .release_end
                .max(pair[0]);
            if end < pair[1] {
                return (end >= after).then_some(end);
            }
        }
        None
    }

    pub(crate) fn support_after(
        self,
        after: u64,
        intervention: Option<ScheduledRelease>,
    ) -> Option<[u64; 2]> {
        let mut edges = [after, u64::MAX, after, after];
        for (edge, release) in edges[2..]
            .iter_mut()
            .zip([self.scheduled_release, intervention])
        {
            *edge = release.map_or(after, |r| r.apply_at_sample.max(after));
        }
        edges.sort_unstable();
        let mut support: Option<[u64; 2]> = None;
        for pair in edges.windows(2) {
            let envelope = self.envelope_at(pair[0], intervention);
            let start = pair[0].max(envelope.onset);
            let end = pair[1].min(envelope.release_end);
            if start < end {
                support = Some(support.map_or([start, end], |[a, b]| [a.min(start), b.max(end)]));
            }
        }
        support
    }

    #[cfg(test)]
    pub(crate) fn sine_point(
        self,
        tick: u64,
        intervention: Option<ScheduledRelease>,
    ) -> Option<([f64; 4], u64, u64)> {
        self.sine_point_from_energy(tick, intervention, self.at(tick, intervention))
    }

    /// Reuse this tone's energy evaluated at the same tick and intervention.
    pub(crate) fn sine_point_from_energy(
        self,
        tick: u64,
        intervention: Option<ScheduledRelease>,
        energy: Option<f64>,
    ) -> Option<([f64; 4], u64, u64)> {
        let energy = energy?;
        if energy == 0. {
            return Some(([0.; 4], 0, u64::MAX));
        }
        let sine = self.sine?;
        // Boost decay assumes continuous activity up to this nonzero target.
        if self.control_for(intervention)?.gain_at(sine.first_sample)? <= 0. {
            return None;
        }
        let [x, y, omega] = sine.at(tick)?;
        let amplitude = (2. * energy).sqrt();
        let envelope = self.envelope_at(tick, intervention);
        Some((
            [x * amplitude, y * amplitude, omega, 0.],
            sine.first_sample.max(envelope.onset),
            envelope.release_end,
        ))
    }

    /// Ticks where this tone's envelope or control changes slope; zero marks an unused
    /// slot. Spans cut here keep each frozen-envelope rule on one smooth piece.
    pub(crate) fn breakpoints(self, intervention: Option<ScheduledRelease>) -> [u64; 24] {
        let mut out = [0; 24];
        let mut n = 0;
        let mut push = |at: u64| {
            if n < out.len() {
                out[n] = at;
                n += 1;
            }
        };
        let mut envelope = self.envelope;
        for release in [None, self.scheduled_release, intervention] {
            if let Some(release) = release {
                // A release that leaves the envelope unchanged must not move any span.
                if release.off_sample >= envelope.hold_end {
                    continue;
                }
                push(release.apply_at_sample);
                envelope = envelope.with_release(release.off_sample);
            }
            let attack = envelope
                .attack_ticks
                .min(envelope.hold_end.saturating_sub(envelope.onset).max(1));
            for at in [
                envelope.onset,
                envelope.onset.saturating_add(attack),
                envelope.onset.saturating_add(attack + envelope.decay_ticks),
                envelope.hold_end,
                envelope.release_end,
            ] {
                push(at);
            }
        }
        for at in [
            self.sine.map(|s| s.first_sample),
            self.bank.map(|b| b.first_sample),
            self.control.and_then(|c| c.starts_at),
            self.control.and_then(|c| c.kick_at),
            self.control.and_then(|c| c.attack_span()).map(|s| s[1]),
        ]
        .into_iter()
        .flatten()
        {
            push(at);
        }
        out
    }

    /// Span bound when `[from, to)` lies in a fast piece of this tone. The exponential
    /// envelope decay keeps `2 lambda h <= 0.1`; an envelope or modulator attack, whose two
    /// ramps make the energy a quartic, uses 32 spans. Partial beats inside a tone ride on
    /// these pieces, so they need more than the smooth Simpson bound alone would ask for.
    pub(crate) fn fast_span(
        self,
        [from, to]: [u64; 2],
        intervention: Option<ScheduledRelease>,
    ) -> Option<u64> {
        let envelope = self.envelope_at(from, intervention);
        let attack = envelope
            .attack_ticks
            .min(envelope.hold_end.saturating_sub(envelope.onset).max(1));
        let begin = envelope.onset.saturating_add(attack);
        let end = begin.saturating_add(envelope.decay_ticks);
        let lambda = f64::from(envelope.decay_lambda);
        let inside = |[a, b]: [u64; 2]| from < b && to > a;
        [
            (inside([begin, end]) && lambda > 0.).then(|| (0.1 / (2. * lambda)) as u64),
            inside([envelope.onset, begin]).then_some(attack / 32),
            self.control
                .and_then(|c| c.attack_span())
                .filter(|s| inside(*s))
                .map(|[a, b]| (b - a) / 32),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// Add this tone's carriers for one span. Over the supported part of the span the
    /// envelope energy, times a bank lane's own squared gain, is read as a local exponential
    /// whose mean is Simpson's rule; its rate joins the carrier's log decay, so the kernel
    /// integrates envelope and carrier together instead of multiplying two span means.
    pub(crate) fn add_span(
        self,
        window: &mut CoherentWindow,
        span: [u64; 2],
        tick: u64,
        intervention: Option<ScheduledRelease>,
    ) {
        let envelope = self.envelope_at(tick, intervention);
        let first = match (self.sine, self.bank) {
            (Some(sine), _) => sine.first_sample,
            (None, Some(bank)) => bank.first_sample,
            (None, None) => return window.add(None),
        };
        let lo = span[0].max(first).max(envelope.onset);
        let hi = span[1].min(envelope.release_end);
        if lo >= hi || (self.sine.is_some() && tick < lo) {
            let energy = self.at(tick, intervention);
            return self.add_carriers(window, tick, intervention, energy);
        }
        let nodes = [lo, lo + (hi - lo) / 2, hi - 1];
        let energies = nodes.map(|at| self.at(at, intervention));
        if energies.iter().all(|e| *e == Some(0.)) {
            return;
        }
        // Energy at `tick` and the amplitude log rate of the local exponential.
        let trend = |[a, b, c]: [f64; 3]| {
            let mean = (a + 4. * b + c) / 6.;
            let n = (hi - lo) as f64;
            let mut rate = if a > 0. && c > 0. && hi - lo > 1 {
                (c / a).ln() / (2. * (n - 1.))
            } else {
                0.
            };
            if !rate.is_finite() || (rate * n).abs() > 4. {
                rate = 0.;
            }
            let shape = if rate == 0. {
                1.
            } else {
                (rate * n).sinh() / (rate * n)
            };
            let center = lo as f64 + (n - 1.) / 2.;
            (
                mean / shape * (2. * rate * (tick as f64 - center)).exp(),
                rate,
            )
        };
        let support = (first.max(envelope.onset), envelope.release_end);
        let Some(bank) = self.bank.filter(|_| self.sine.is_none()) else {
            let point = energies[0]
                .zip(energies[1])
                .zip(energies[2])
                .and_then(|((a, b), c)| {
                    let (energy, rate) = trend([a, b, c]);
                    let (mut point, start, end) =
                        self.sine_point_from_energy(tick, intervention, Some(energy))?;
                    point[3] = rate;
                    Some((point, start, end))
                });
            return window.add(point);
        };
        for lane in 0..bank.len {
            let mut scaled = [Some(0.); 3];
            for ((slot, at), energy) in scaled.iter_mut().zip(nodes).zip(energies) {
                *slot = energy
                    .zip(bank.gain_ratio(lane, at, tick))
                    .map(|(e, g)| e * g * g);
            }
            let point = scaled[0]
                .zip(scaled[1])
                .zip(scaled[2])
                .zip(bank.lane_at(lane, tick))
                .map(|(((a, b), c), [x, y, omega, decay])| {
                    let (energy, rate) = trend([a, b, c]);
                    let amplitude = (2. * energy).sqrt();
                    (
                        [x * amplitude, y * amplitude, omega, decay + rate],
                        support.0,
                        support.1,
                    )
                });
            window.add(point);
        }
    }

    /// Add this tone's carriers, reusing its energy at the same tick and intervention:
    /// the sine, or every lane of a harmonic or modal bank.
    pub(crate) fn add_carriers(
        self,
        window: &mut CoherentWindow,
        tick: u64,
        intervention: Option<ScheduledRelease>,
        energy: Option<f64>,
    ) {
        let Some(bank) = self.bank.filter(|_| self.sine.is_none()) else {
            return window.add(self.sine_point_from_energy(tick, intervention, energy));
        };
        let Some(energy) = energy else {
            return window.add(None);
        };
        if energy == 0. {
            return;
        }
        let amplitude = (2. * energy).sqrt();
        let envelope = self.envelope_at(tick, intervention);
        for lane in 0..bank.len {
            window.add(bank.lane_at(lane, tick).map(|[x, y, omega, decay]| {
                (
                    [x * amplitude, y * amplitude, omega, decay],
                    bank.first_sample.max(envelope.onset),
                    envelope.release_end,
                )
            }));
        }
    }

    pub(crate) fn at(self, tick: u64, intervention: Option<ScheduledRelease>) -> Option<f64> {
        let envelope = self.envelope_at(tick, intervention);
        let outer = f64::from(envelope.gain_at(tick));
        if outer == 0. {
            return Some(0.);
        }
        let control = self.control_for(intervention)?;
        let gain = control.gain_at(tick)?;
        let amplitude = control.amplitude_at(tick, self.amplitude)?;
        Some(amplitude.powi(2) * outer.powi(2) * gain.powi(2) / 2.)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CoherentWindow {
    points: [([f64; 4], u64, u64); 64],
    len: usize,
    complete: bool,
}

impl CoherentWindow {
    pub(crate) fn new() -> Self {
        Self {
            points: [([0.; 4], 0, 0); 64],
            len: 0,
            complete: true,
        }
    }

    pub(crate) fn add(&mut self, point: Option<([f64; 4], u64, u64)>) {
        let Some(point) = point else {
            self.complete = false;
            return;
        };
        if point.0[..2] == [0., 0.] {
            return;
        }
        if self.len == self.points.len() {
            self.complete = false;
            return;
        }
        self.points[self.len] = point;
        self.len += 1;
    }

    pub(crate) fn mean(&self, left: u64, right: u64, midpoint: u64) -> Option<f64> {
        if !self.complete || left >= right || midpoint < left || midpoint >= right {
            return None;
        }
        if self.points[..self.len].iter().any(|p| p.0[3] != 0.) {
            return self.decaying_mean(left, right, midpoint);
        }
        let width = (right - left) as f64;
        let cosine = |real: f64, imag: f64, omega: f64, begin: u64, end: u64| {
            if end <= begin {
                return 0.;
            }
            // Integer samples alias 2*pi to zero; reduce before the small-angle ratio.
            let omega = if omega.abs() > std::f64::consts::PI {
                omega - std::f64::consts::TAU * (omega / std::f64::consts::TAU).round()
            } else {
                omega
            };
            let n = (end - begin) as f64;
            if omega == 0. {
                return real * n / width;
            }
            let center = (begin as i128 - midpoint as i128) as f64 + (n - 1.) / 2.;
            let half = omega / 2.;
            let phase = omega * center;
            if (n * half).abs().max(phase.abs()) <= 0.01 {
                // Degree-six remainders here are below 3e-21, beneath f64 roundoff.
                let a = (n * half).powi(2);
                let b = half * half;
                let p = phase * phase;
                let numerator = 1. + a * (-1. / 6. + a * (1. / 120. - a / 5040.));
                let denominator = 1. + b * (-1. / 6. + b * (1. / 120. - b / 5040.));
                let s = phase * (1. + p * (-1. / 6. + p * (1. / 120. - p / 5040.)));
                let c = 1. + p * (-0.5 + p * (1. / 24. - p / 720.));
                return (real * c - imag * s) * (numerator / denominator) * n / width;
            }
            let scale = if half == 0. {
                1.
            } else {
                (n * half).sin() / (n * half.sin())
            };
            let (s, c) = (omega * center).sin_cos();
            (real * c - imag * s) * scale * n / width
        };
        let rotate = |[x, y, omega, _]: [f64; 4], offset: f64| {
            let (s, c) = (omega * offset).sin_cos();
            [x * c - y * s, x * s + y * c]
        };
        #[derive(Clone, Copy, Default)]
        struct Edges {
            begin: [f64; 2],
            end: [f64; 2],
            half_rotation: [f64; 2],
            begin_offset: f64,
            end_offset: f64,
        }
        let mut edges = [Edges::default(); 64];
        for (edge, &(point, start, end)) in edges.iter_mut().zip(&self.points[..self.len]) {
            let (begin, end) = (left.max(start), right.min(end));
            if begin < end {
                let (s, c) = (point[2] * 0.5).sin_cos();
                let begin_offset = (begin as i128 - midpoint as i128) as f64 - 0.5;
                let end_offset = (end as i128 - midpoint as i128) as f64 - 0.5;
                *edge = Edges {
                    begin: rotate(point, begin_offset),
                    end: rotate(point, end_offset),
                    half_rotation: [s, c],
                    begin_offset,
                    end_offset,
                };
            }
        }
        let mut sum = 0.;
        for i in 0..self.len {
            let ([x, y, omega, _], start, end) = self.points[i];
            let (begin, end) = (left.max(start), right.min(end));
            if begin >= end {
                continue;
            }
            let edge = edges[i];
            let sin_omega = 2. * edge.half_rotation[0] * edge.half_rotation[1];
            // The centered form avoids cancellation near a singular geometric sum.
            let doubled = if sin_omega.abs() < 1e-4 {
                cosine(x * x - y * y, 2. * x * y, 2. * omega, begin, end)
            } else {
                (edge.end[0] * edge.end[1] - edge.begin[0] * edge.begin[1]) / (sin_omega * width)
            };
            sum += 0.5 * ((x * x + y * y) * (end - begin) as f64 / width - doubled);
            for j in 0..i {
                let ([u, v, other, _], a, b) = self.points[j];
                let (a, b) = (left.max(a), right.min(b));
                let (lo, hi) = (begin.max(a), end.min(b));
                if lo >= hi {
                    continue;
                }
                // Equal clipped edges share rotations across every carrier pair.
                let p = if lo == begin {
                    edge.begin
                } else {
                    rotate(self.points[i].0, edges[j].begin_offset)
                };
                let q = if hi == end {
                    edge.end
                } else {
                    rotate(self.points[i].0, edges[j].end_offset)
                };
                let r = if lo == a {
                    edges[j].begin
                } else {
                    rotate(self.points[j].0, edge.begin_offset)
                };
                let s = if hi == b {
                    edges[j].end
                } else {
                    rotate(self.points[j].0, edge.end_offset)
                };
                let [si, ci] = edge.half_rotation;
                let [sj, cj] = edges[j].half_rotation;
                let difference = si * cj - ci * sj;
                let combined = si * cj + ci * sj;
                if difference.abs() >= 1e-4 && combined.abs() >= 1e-4 {
                    let a = q[1] * s[0] - p[1] * r[0];
                    let b = q[0] * s[1] - p[0] * r[1];
                    sum += (ci * sj * a - si * cj * b) / (difference * combined * width);
                    continue;
                }
                let minus = if difference.abs() < 1e-4 {
                    cosine(x * u + y * v, y * u - x * v, omega - other, lo, hi)
                } else {
                    (q[1] * s[0] - q[0] * s[1] - (p[1] * r[0] - p[0] * r[1]))
                        / (2. * difference * width)
                };
                let plus = if combined.abs() < 1e-4 {
                    cosine(x * u - y * v, x * v + y * u, omega + other, lo, hi)
                } else {
                    (q[1] * s[0] + q[0] * s[1] - (p[1] * r[0] + p[0] * r[1]))
                        / (2. * combined * width)
                };
                sum += minus - plus;
            }
        }
        sum.is_finite().then_some(sum.max(0.))
    }
}

impl CoherentWindow {
    /// The same integer-sample mean for carriers `Re(a exp(u k))` with a per-sample log
    /// decay in `u`, as complex geometric sums about the midpoint.
    fn decaying_mean(&self, left: u64, right: u64, midpoint: u64) -> Option<f64> {
        type Complex = [f64; 2];
        let mul = |[a, b]: Complex, [c, d]: Complex| [a * c - b * d, a * d + b * c];
        let exp = |[a, b]: Complex| {
            let (s, c) = b.sin_cos();
            [a.exp() * c, a.exp() * s]
        };
        // Real part of `amplitude * sum_{k=first}^{first+n-1} exp(u k)`.
        let series = |amplitude: Complex, [decay, omega]: Complex, first: f64, n: f64| {
            let omega = omega - std::f64::consts::TAU * (omega / std::f64::consts::TAU).round();
            let u = [decay, omega];
            let sum = if decay == 0. && omega == 0. {
                [n, 0.]
            } else if decay.hypot(omega) < 1e-5 {
                // Centered ratio `sinh(un/2) / sinh(u/2)` avoids cancellation near `u = 0`.
                let sinh = |[a, b]: Complex| [a.sinh() * b.cos(), a.cosh() * b.sin()];
                let [c, d] = sinh([decay / 2., omega / 2.]);
                let norm = c * c + d * d;
                let ratio = mul(
                    sinh([decay * n / 2., omega * n / 2.]),
                    [c / norm, -d / norm],
                );
                mul(exp([decay * (n - 1.) / 2., omega * (n - 1.) / 2.]), ratio)
            } else {
                let [p, q] = exp([decay * n, omega * n]);
                let [c, d] = exp(u);
                let norm = (1. - c).powi(2) + d * d;
                mul([1. - p, -q], [(1. - c) / norm, d / norm])
            };
            mul(mul(amplitude, exp([decay * first, omega * first])), sum)[0]
        };
        let width = (right - left) as f64;
        // Per-carrier powers over its own clipped support: pairs that share those edges,
        // the common case, then need no transcendental call.
        let mut powers = [([0.; 2], [0.; 2], [0.; 2]); 64];
        for (power, &([_, _, omega, decay], start, end)) in
            powers.iter_mut().zip(&self.points[..self.len])
        {
            let (begin, end) = (left.max(start), right.min(end));
            if begin < end {
                let first = (begin as i128 - midpoint as i128) as f64;
                let n = (end - begin) as f64;
                *power = (
                    exp([decay, omega]),
                    exp([decay * first, omega * first]),
                    exp([decay * n, omega * n]),
                );
            }
        }
        let conj = |[a, b]: Complex| [a, -b];
        let mut sum = 0.;
        for i in 0..self.len {
            let ([x, y, omega, decay], start, end) = self.points[i];
            for j in 0..=i {
                let ([u, v, other, other_decay], a, b) = self.points[j];
                let (lo, hi) = (left.max(start).max(a), right.min(end).min(b));
                if lo >= hi {
                    continue;
                }
                let first = (lo as i128 - midpoint as i128) as f64;
                let n = (hi - lo) as f64;
                // `y cos + x sin = Re((y - ix) exp(i omega k))`.
                let (p, q) = ([y, -x], [v, -u]);
                let rate = decay + other_decay;
                let shared = (lo, hi) == (left.max(start), right.min(end))
                    && (lo, hi) == (left.max(a), right.min(b));
                let pair = [(conj(q), omega - other, true), (q, omega + other, false)]
                    .into_iter()
                    .map(|(q, angle, conjugate)| {
                        let wrapped =
                            angle - std::f64::consts::TAU * (angle / std::f64::consts::TAU).round();
                        if !shared || rate.hypot(wrapped) < 1e-5 {
                            return series(mul(p, q), [rate, angle], first, n);
                        }
                        let (pi, pj) = (powers[i], powers[j]);
                        let turn = |k: Complex| if conjugate { conj(k) } else { k };
                        let [c, d] = mul(pi.0, turn(pj.0));
                        let [g, h] = mul(pi.2, turn(pj.2));
                        let norm = (1. - c).powi(2) + d * d;
                        let ratio = mul([1. - g, -h], [(1. - c) / norm, d / norm]);
                        mul(mul(mul(p, q), mul(pi.1, turn(pj.1))), ratio)[0]
                    })
                    .sum::<f64>();
                sum += if i == j { 0.5 } else { 1. } * pair / width;
            }
        }
        sum.is_finite().then_some(sum.max(0.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::control_forecast::AmplitudeModel;

    #[test]
    fn conditional_release_preserves_old_plan_and_causal_attack_clipping() {
        let tone = ToneEnergy {
            amplitude: 0.2,
            sine: None,
            bank: None,
            envelope: Envelope {
                onset: 100,
                hold_end: 1000,
                release_end: 1040,
                attack_ticks: 80,
                decay_ticks: 0,
                sustain_level: 1.,
                decay_lambda: 0.,
                release_ticks: 40,
            },
            control: Some(ControlForecast {
                issued_at: 96,
                valid_until: None,
                amplitude_smoothing: None,
                amplitude_updates: None,
                sample_dt: 1. / 8000.,
                starts_at: Some(100),
                kick_at: None,
                model: AmplitudeModel::Unmodulated { gain: 1. },
            }),
            scheduled_release: Some(ScheduledRelease {
                apply_at_sample: 128,
                off_sample: 144,
            }),
        };
        let added = ScheduledRelease {
            apply_at_sample: 192,
            off_sample: 200,
        };
        for tick in 96..256 {
            assert_eq!(tone.at(tick, Some(added)), tone.at(tick, None));
        }
        let early = ScheduledRelease {
            apply_at_sample: 112,
            off_sample: 120,
        };
        for tick in 100..112 {
            assert_eq!(tone.at(tick, Some(early)), tone.at(tick, None));
        }
        assert_ne!(tone.at(112, Some(early)), tone.at(112, None));
        assert!(tone.at(159, Some(early)).unwrap() > 0.);
        assert_eq!(tone.at(160, Some(early)), Some(0.));
        let swapped = ToneEnergy {
            scheduled_release: Some(early),
            ..tone
        };
        for tick in 96..256 {
            assert_eq!(
                tone.at(tick, Some(early)),
                swapped.at(tick, tone.scheduled_release)
            );
        }
        let unknown = ToneEnergy {
            control: None,
            ..tone
        };
        assert_eq!(unknown.at(100, None), None);
        assert_eq!(unknown.at(184, None), Some(0.));
    }
}

#[cfg(test)]
mod coherent_tests {
    use super::*;

    #[test]
    #[ignore = "explicit kernel cost probe, not complete hop-resource acceptance"]
    fn coherent_window_cost_probe() {
        for support in [
            "aligned",
            "staggered",
            "near_unison",
            "unison",
            "staggered_near_unison",
            "staggered_unison",
        ] {
            for count in [1, 4, 16, 64] {
                let mut nanos = Vec::new();
                for repetition in 0..257 {
                    let start = std::time::Instant::now();
                    for k in 0..32 {
                        let mut window = CoherentWindow::new();
                        for i in 0..count {
                            let angle = (i + k + repetition) as f64 * 0.11;
                            let omega = match support {
                                "near_unison" | "staggered_near_unison" => 0.02 + i as f64 * 1e-7,
                                "unison" | "staggered_unison" => 0.02,
                                _ => 0.02 + i as f64 * 0.003,
                            };
                            let (begin, end) = if support.starts_with("staggered") {
                                (1000 + i as u64 * 5, 2000 - i as u64 * 3)
                            } else {
                                (0, 100000)
                            };
                            window.add(Some((
                                [0.2 * angle.cos(), 0.2 * angle.sin(), omega, 0.],
                                begin,
                                end,
                            )));
                        }
                        std::hint::black_box(window.mean(1000, 2000, 1500));
                    }
                    nanos.push(start.elapsed().as_nanos());
                }
                nanos.sort_unstable();
                eprintln!(
                    "COHERENT_COST {{\"support\":\"{support}\",\"tones\":{count},\"windows\":32,\"samples\":257,\"median_ns\":{},\"p99_ns\":{},\"scratch_bytes\":{}}}",
                    nanos[128],
                    nanos[254],
                    32 * std::mem::size_of::<CoherentWindow>()
                        + 64 * 8 * std::mem::size_of::<f64>()
                );
            }
        }
    }

    #[test]
    fn coherent_edges_match_direct_samples_at_capacity_and_near_singular_frequencies() {
        let mut maximum = 0.0f64;
        let mut cases = 0;
        for width in [1, 7, 4096] {
            for count in [1, 4, 16, 64] {
                for mode in 0..7 {
                    let left = 1_000_000_000_000_000;
                    let right = left + width;
                    let mid = left + width / 2;
                    let mut window = CoherentWindow::new();
                    let mut direct = vec![0.; width as usize];
                    for i in 0..count {
                        let angle = i as f64 * 1.39;
                        let omega = match mode {
                            0 => 0.02 + i as f64 * 0.003,
                            1 => 0.2,
                            2 => 0.2 + i as f64 * 1e-12,
                            3 => i as f64 * 1e-12,
                            4 => std::f64::consts::PI - i as f64 * 1e-12,
                            6 => 0.2 + i as f64 * 1e-7,
                            _ => 0.2 + i as f64 * 0.000099,
                        };
                        let (s, c) = angle.sin_cos();
                        let (x, y) = (0.02 * c, 0.02 * s);
                        let start = left + i as u64 * (width / 128);
                        let end = right - (count - 1 - i) as u64 * (width / 128);
                        window.add(Some(([x, y, omega, 0.], start, end)));
                        for tick in start..end {
                            let (s, c) = (omega * (tick as i128 - mid as i128) as f64).sin_cos();
                            direct[(tick - left) as usize] += x * s + y * c;
                        }
                    }
                    let direct = direct.iter().map(|x| x * x).sum::<f64>() / width as f64;
                    let error = (window.mean(left, right, mid).unwrap() - direct).abs();
                    maximum = maximum.max(error);
                    cases += 1;
                    assert!(
                        error < 1e-12,
                        "width={width} count={count} mode={mode} error={error}"
                    );
                }
            }
        }
        eprintln!("COHERENT_EDGES cases={cases} max_error={maximum}");
    }

    #[test]
    fn coherent_window_matches_direct_wave_sums_with_phase_cancellation_and_partial_overlap() {
        for left in [0, 1_000_000_000_000_000] {
            for width in [1, 7, 60, 750, 12000] {
                let right = left + width;
                let mid = left + width / 2;
                for (frequencies, decays) in [
                    ([0.04, 0.04], [0., 0.]),
                    ([0.04, 0.04001], [0., 0.]),
                    ([0.04, 0.12], [0., 0.]),
                    ([0.04, 0.04], [-3e-4, -3e-4]),
                    ([0.04, 0.04000001], [-1e-9, 0.]),
                    ([0.04, 0.12], [-2e-3, -1e-7]),
                ] {
                    for shift in [0., 1., std::f64::consts::PI] {
                        let mut window = CoherentWindow::new();
                        let points = [
                            ([0.2, 0., frequencies[0], decays[0]], left, right),
                            (
                                [
                                    0.2 * shift.cos(),
                                    0.2 * shift.sin(),
                                    frequencies[1],
                                    decays[1],
                                ],
                                left + width / 3,
                                right,
                            ),
                        ];
                        for point in points {
                            window.add(Some(point));
                        }
                        let direct: f64 = (left..right)
                            .map(|tick| {
                                points
                                    .iter()
                                    .map(|&([x, y, omega, decay], start, end)| {
                                        if tick < start || tick >= end {
                                            return 0.;
                                        }
                                        let k = (tick as i128 - mid as i128) as f64;
                                        let (s, c) = (omega * k).sin_cos();
                                        (x * s + y * c) * (decay * k).exp()
                                    })
                                    .sum::<f64>()
                                    .powi(2)
                            })
                            .sum::<f64>()
                            / width as f64;
                        let error = (window.mean(left, right, mid).unwrap() - direct).abs();
                        assert!(
                            error < 1e-12 * direct.max(1.),
                            "width={width} decays={decays:?} error={error} direct={direct}"
                        );
                    }
                }
            }
        }
        let mut unsupported = CoherentWindow::new();
        unsupported.add(None);
        assert_eq!(unsupported.mean(0, 10, 5), None);
        let mut full = CoherentWindow::new();
        for _ in 0..64 {
            full.add(Some(([1., 0., 0.2, 0.], 0, 100)));
        }
        assert!(full.mean(0, 10, 5).is_some());
        full.add(Some(([1., 0., 0.2, 0.], 0, 100)));
        assert_eq!(full.mean(0, 10, 5), None);
    }
}
