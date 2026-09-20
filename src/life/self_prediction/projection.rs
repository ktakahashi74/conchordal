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

    pub(crate) fn sine_point(
        self,
        tick: u64,
        intervention: Option<ScheduledRelease>,
    ) -> Option<([f64; 3], u64, u64)> {
        self.sine_point_from_energy(tick, intervention, self.at(tick, intervention))
    }

    /// Reuse this tone's energy evaluated at the same tick and intervention.
    pub(crate) fn sine_point_from_energy(
        self,
        tick: u64,
        intervention: Option<ScheduledRelease>,
        energy: Option<f64>,
    ) -> Option<([f64; 3], u64, u64)> {
        let energy = energy?;
        if energy == 0. {
            return Some(([0.; 3], 0, u64::MAX));
        }
        let sine = self.sine?;
        // Boost decay assumes continuous activity up to this nonzero target.
        if self.control_for(intervention)?.gain_at(sine.first_sample)? <= 0. {
            return None;
        }
        let mut point = sine.at(tick)?;
        let amplitude = (2. * energy).sqrt();
        point[0] *= amplitude;
        point[1] *= amplitude;
        let envelope = self.envelope_at(tick, intervention);
        Some((
            point,
            sine.first_sample.max(envelope.onset),
            envelope.release_end,
        ))
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
    points: [([f64; 3], u64, u64); 64],
    len: usize,
    complete: bool,
}

impl CoherentWindow {
    pub(crate) fn new() -> Self {
        Self {
            points: [([0.; 3], 0, 0); 64],
            len: 0,
            complete: true,
        }
    }

    pub(crate) fn add(&mut self, point: Option<([f64; 3], u64, u64)>) {
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
        let rotate = |[x, y, omega]: [f64; 3], offset: f64| {
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
            let ([x, y, omega], start, end) = self.points[i];
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
                let ([u, v, other], a, b) = self.points[j];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::sound::control_forecast::AmplitudeModel;

    #[test]
    fn conditional_release_preserves_old_plan_and_causal_attack_clipping() {
        let tone = ToneEnergy {
            amplitude: 0.2,
            sine: None,
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
                                [0.2 * angle.cos(), 0.2 * angle.sin(), omega],
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
                        window.add(Some(([x, y, omega], start, end)));
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
                for frequencies in [[0.04, 0.04], [0.04, 0.04001], [0.04, 0.12]] {
                    for shift in [0., 1., std::f64::consts::PI] {
                        let mut window = CoherentWindow::new();
                        let points = [
                            ([0.2, 0., frequencies[0]], left, right),
                            (
                                [0.2 * shift.cos(), 0.2 * shift.sin(), frequencies[1]],
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
                                    .map(|&([x, y, omega], start, end)| {
                                        if tick < start || tick >= end {
                                            return 0.;
                                        }
                                        let (s, c) = (omega
                                            * ((tick as i128 - mid as i128) as f64))
                                            .sin_cos();
                                        x * s + y * c
                                    })
                                    .sum::<f64>()
                                    .powi(2)
                            })
                            .sum::<f64>()
                            / width as f64;
                        assert!((window.mean(left, right, mid).unwrap() - direct).abs() < 1e-12);
                    }
                }
            }
        }
        let mut unsupported = CoherentWindow::new();
        unsupported.add(None);
        assert_eq!(unsupported.mean(0, 10, 5), None);
        let mut full = CoherentWindow::new();
        for _ in 0..64 {
            full.add(Some(([1., 0., 0.2], 0, 100)));
        }
        assert!(full.mean(0, 10, 5).is_some());
        full.add(Some(([1., 0., 0.2], 0, 100)));
        assert_eq!(full.mean(0, 10, 5), None);
    }
}
