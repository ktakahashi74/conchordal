use super::*;

impl ToneEnergy {
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
    pub(crate) fn breakpoints(self, intervention: Option<ScheduledRelease>) -> [u64; 32] {
        let mut out = [0; 32];
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
            let attack = envelope.attack_len();
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
        .chain(self.control.map_or([None; 5], |c| c.steps()))
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
        let attack = envelope.attack_len();
        let begin = envelope.onset.saturating_add(attack);
        let end = begin.saturating_add(envelope.decay_ticks);
        let lambda = f64::from(envelope.decay_lambda);
        let inside = |[a, b]: [u64; 2]| from < b && to > a;
        [
            (inside([begin, end]) && lambda > 0.).then(|| (0.1 / (2. * lambda)) as u64),
            inside([envelope.onset, begin]).then_some(attack / 32),
            // Cross terms with a free burst also resolve the carrier's linear release.
            (self.bank.is_some_and(|b| b.burst.is_some())
                && inside([envelope.hold_end, envelope.release_end]))
            .then_some(envelope.release_ticks / 32),
            self.control
                .and_then(|c| c.attack_span())
                .filter(|s| inside(*s))
                .map(|[a, b]| (b - a) / 32),
            self.control.and_then(|c| c.smoothing_span([from, to])),
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
        if let Some(bank) = self.bank
            && let Some(burst) = bank.burst
        {
            let end = burst.end();
            // Off can cancel a future strike, but cannot shorten a burst already emitted.
            let emitted = burst.onset < bank.first_sample;
            if (emitted || burst.onset < self.envelope_at(burst.onset, intervention).release_end)
                && span[0] < end
                && burst.onset < span[1]
            {
                for lane in 0..crate::life::sound::attack_noise::NOISE_LANES {
                    window.add(Some((burst.point(lane, tick), burst.onset, end)));
                }
            }
        }
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
            let n = (hi - lo) as f64;
            // One or two samples are their own mean; Simpson's weights need three nodes.
            let mean = if hi - lo <= 2 {
                (a + c) / 2.
            } else {
                (a + 4. * b + c) / 6.
            };
            let mut rate = if a > 0. && c > 0. && hi - lo > 1 {
                (c / a).ln() / (2. * (n - 1.))
            } else {
                0.
            };
            if !rate.is_finite() || (rate * n).abs() > 4. {
                rate = 0.;
            }
            // Mean of `exp(2 rate (k - center))` over the span's integer samples.
            let shape = if rate == 0. {
                1.
            } else {
                (rate * n).sinh() / (n * rate.sinh())
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
