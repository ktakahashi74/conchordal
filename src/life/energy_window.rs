//! Frozen body energy over the representative 16-bin window.

use super::tone_energy::{CoherentWindow, ScheduledRelease, ToneEnergy};
use serde::Serialize;

const CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Window {
    pub start: u64,
    pub end: u64,
    pub points: [u64; 16],
    pub energies: [Option<f64>; 16],
    pub incoherent_energies: [Option<f64>; 16],
    pub coherent_energies: [Option<f64>; 16],
    pub mean: Option<f64>,
    pub default_mean: Option<f64>,
    pub difference: Option<f64>,
}

pub(crate) struct Body<'a> {
    pub(crate) retained: &'a [(u64, [bool; 2], ToneEnergy)],
    pub(crate) added: Option<([bool; 2], ToneEnergy)>,
    pub(crate) at: u64,
    pub(crate) intervention: (Option<ScheduledRelease>, Option<u64>),
}

impl Body<'_> {
    pub(crate) fn tones(
        &self,
        bus: usize,
    ) -> impl Iterator<Item = (Option<u64>, ToneEnergy, Option<ScheduledRelease>)> + '_ {
        self.retained
            .iter()
            .filter_map(move |(id, routed, tone)| {
                if !routed[bus]
                    || self.intervention.1.is_some_and(|until| {
                        self.at <= tone.envelope.onset && tone.envelope.onset < until
                    })
                {
                    return None;
                }
                let envelope = tone
                    .scheduled_release
                    .map_or(tone.envelope, |p| tone.envelope.with_release(p.off_sample));
                let release = (envelope.onset <= self.at && self.at < envelope.release_end)
                    .then_some(self.intervention.0)
                    .flatten();
                Some((Some(*id), *tone, release))
            })
            .chain(
                self.added
                    .into_iter()
                    .filter_map(move |(routed, tone)| routed[bus].then_some((None, tone, None))),
            )
    }

    pub(crate) fn point(
        &self,
        tick: u64,
        left: u64,
        right: u64,
        bus: usize,
    ) -> (Option<f64>, Option<f64>) {
        let mut energy = Some(0.);
        for (_, tone, release) in self.tones(bus) {
            energy = energy.zip(tone.at(tick, release)).map(|(a, b)| a + b);
        }
        // Envelopes are frozen per span: cut the bin at every envelope breakpoint, then into
        // spans short enough for the rule to hold on each smooth piece.
        let mut edges = [left; 130];
        let mut count = 1;
        let mut complete = true;
        for (_, tone, release) in self.tones(bus) {
            for at in tone.breakpoints(release) {
                if at <= left || at >= right || edges[..count].contains(&at) {
                    continue;
                }
                if count == edges.len() - 1 {
                    complete = false;
                    break;
                }
                edges[count] = at;
                count += 1;
            }
        }
        edges[count] = right;
        edges[..=count].sort_unstable();
        // A breakpoint that does not fit would leave a step inside a span.
        let mut coherent = complete.then_some(0.);
        for piece in edges[..=count].windows(2) {
            let [from, to] = [piece[0], piece[1]];
            let limit = self
                .tones(bus)
                .filter_map(|(_, tone, release)| tone.fast_span([from, to], release))
                .fold(COHERENT_SPAN, u64::min)
                .max(1);
            let spans = (to - from).div_ceil(limit);
            for k in 0..spans {
                let [a, b] = [k, k + 1].map(|edge| from + (to - from) * edge / spans);
                let mut cursor = a;
                while cursor < b {
                    // Held partial gains change at the bank's actual refresh boundary.
                    let next = self
                        .tones(bus)
                        .filter_map(|(_, tone, _)| {
                            let crate::life::sound::bank_forecast::Response::Oscillator {
                                refresh_at,
                                refresh_period,
                                ..
                            } = tone.bank?.response
                            else {
                                return None;
                            };
                            if cursor < refresh_at {
                                Some(refresh_at)
                            } else {
                                let period = refresh_period.max(1);
                                cursor.checked_add(period - (cursor - refresh_at) % period)
                            }
                        })
                        .min()
                        .unwrap_or(b)
                        .min(b);
                    let middle = cursor + (next - cursor) / 2;
                    let mut window = CoherentWindow::new();
                    for (_, tone, release) in self.tones(bus) {
                        tone.add_span(&mut window, [cursor, next], middle, release);
                    }
                    coherent =
                        coherent
                            .zip(window.mean(cursor, next, middle))
                            .map(|(sum, mean)| {
                                sum + mean * (next - cursor) as f64 / (right - left) as f64
                            });
                    cursor = next;
                }
            }
        }
        (energy, coherent)
    }
}

/// Longest span over which a candidate's envelope and control gains are frozen.
const COHERENT_SPAN: u64 = 750;

pub(crate) fn project_window(
    retained: &[(u64, [bool; 2], ToneEnergy)],
    added: Option<([bool; 2], ToneEnergy)>,
    action_at: u64,
    intervention: (Option<ScheduledRelease>, Option<u64>),
    bus: usize,
    interval: [u64; 2],
    use_coherent: bool,
) -> Option<Window> {
    let [start, end] = interval;
    let width = end.checked_sub(start)?;
    if bus > 1 || width < 16 || retained.len() > CAPACITY {
        return None;
    }
    let mut window = Window {
        start,
        end,
        points: [0; 16],
        energies: [None; 16],
        incoherent_energies: [None; 16],
        coherent_energies: [None; 16],
        mean: None,
        default_mean: None,
        difference: None,
    };
    let mut total = Some(0.);
    for k in 0..16 {
        let [left, right] =
            [k, k + 1].map(|edge| start + (u128::from(width) * edge as u128).div_ceil(16) as u64);
        let tick = left + (right - left) / 2;
        window.points[k] = tick;
        let body = Body {
            retained,
            added,
            at: action_at,
            intervention,
        };
        let (mut energy, coherent) = body.point(tick, left, right, bus);
        window.incoherent_energies[k] = energy;
        window.coherent_energies[k] = coherent;
        if use_coherent {
            energy = window.coherent_energies[k].or(energy);
        }
        window.energies[k] = energy;
        total = total
            .zip(energy)
            .map(|(sum, e)| sum + e * (right - left) as f64 / width as f64);
    }
    window.mean = total;
    Some(window)
}
