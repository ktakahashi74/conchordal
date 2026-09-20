//! Bounded native private participation filter; runtime inventory wiring is separate.

pub(crate) use super::reference_inventory::{Family, Key};
use super::ridge::Handle;

pub(crate) const REFERENCES: usize = 16;
pub(crate) const BINS: usize = 32;
const COORDINATES: usize = BINS + 1;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Anchor {
    pub group: Handle,
    pub interval: Option<[f64; 2]>,
    pub period: f64,
    pub weight: f64,
}

impl Default for Anchor {
    fn default() -> Self {
        Self {
            group: Handle {
                bus: 0,
                epoch: 0,
                generation: 0,
            },
            interval: None,
            period: 1.,
            weight: 0.,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Reference {
    pub key: Key,
    pub weight: f64,
    pub anchors: [Anchor; 7],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Head {
    Onset,
    Release,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Parameters {
    pub tau: f64,
    pub kappa: f64,
    pub strength_max: f64,
    pub interference: bool,
}

#[derive(Clone, Copy)]
struct Trace {
    key: Key,
    mass: [[f64; COORDINATES]; 2],
    end: f64,
}

pub(crate) struct Bank {
    epoch: u64,
    parameters: Parameters,
    capacity: usize,
    traces: Vec<Trace>,
    last: Option<(f64, Head, u64)>,
}

#[derive(Debug)]
pub(crate) struct Receipt {
    pub applied: bool,
    pub credits: [f64; REFERENCES],
    pub unassigned: f64,
    pub removed: [Option<Key>; REFERENCES],
    pub evicted: [Option<Key>; REFERENCES],
}

pub(crate) struct Outcome<'a> {
    pub id: u64,
    pub head: Head,
    pub interval: Option<[f64; 2]>,
    pub references: &'a [Reference],
    pub observed_fraction: f64,
    pub retained: &'a [Key],
    pub confirmed: bool,
}

/// Both values use the same supported anchor mass, without redistributing its complement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PairedLookup {
    pub candidate: f64,
    pub body_default: f64,
    pub support: f64,
}

fn lookup_mass(family: Family, masses: &[f64; COORDINATES], coordinate: f64) -> Option<f64> {
    if !coordinate.is_finite() {
        return None;
    }
    let (left, right, fraction) = if family == Family::Periodic {
        let at = coordinate.rem_euclid(1.) * BINS as f64 - 0.5;
        let left = at.floor() as i64;
        (
            left.rem_euclid(BINS as i64) as usize,
            (left + 1).rem_euclid(BINS as i64) as usize,
            at - at.floor(),
        )
    } else {
        if !(0. ..=4.).contains(&coordinate) {
            return None;
        }
        let at = (coordinate * BINS as f64 / 4. - 0.5).clamp(0., (BINS - 1) as f64);
        (
            at.floor() as usize,
            (at.floor() as usize + 1).min(BINS - 1),
            at - at.floor(),
        )
    };
    Some(masses[left] * (1. - fraction) + masses[right] * fraction)
}

fn mean_lookup(family: Family, masses: &[f64; COORDINATES], lo: f64, width: f64) -> Option<f64> {
    if !lo.is_finite() || !width.is_finite() || width < 0. {
        return None;
    }
    if width == 0. {
        return lookup_mass(family, masses, lo);
    }
    let periodic = family == Family::Periodic;
    let (start, mut remaining, complete) = if periodic {
        (
            lo.rem_euclid(1.),
            width.fract(),
            width.floor() / width * masses[..BINS].iter().sum::<f64>() / BINS as f64,
        )
    } else {
        if lo < 0. || lo + width > 4. {
            return None;
        }
        (lo, width, 0.)
    };
    let mut result = complete;
    let mut left = start;
    let mut left_value = lookup_mass(family, masses, left)?;
    // Keep interval width separate: adding a tiny width to a phase can round it away.
    for k in 0..=2 * BINS {
        let knot = (k as f64 + 0.5) * if periodic { 1. } else { 4. } / BINS as f64;
        let distance = (knot - left).max(0.).min(remaining);
        if distance > 0. {
            let right = left + distance;
            let right_value = lookup_mass(family, masses, right)?;
            result += distance / width * (left_value + right_value) / 2.;
            remaining -= distance;
            left = right;
            left_value = right_value;
        }
        if remaining == 0. {
            break;
        }
    }
    Some(result)
}

pub(crate) fn paired_lookup(
    reference: &Reference,
    masses: &[f64; COORDINATES],
    candidate: f64,
    body_default: f64,
) -> PairedLookup {
    let mut result = PairedLookup::default();
    if !candidate.is_finite() || !body_default.is_finite() {
        return result;
    }
    for anchor in &reference.anchors {
        let Some([a, b]) = anchor.interval else {
            continue;
        };
        if anchor.weight <= 0. {
            continue;
        }
        let (lo, hi) = if reference.key.family == Family::Periodic {
            (a, b)
        } else {
            (
                a.max(candidate - 4. * anchor.period)
                    .max(body_default - 4. * anchor.period),
                b.min(candidate).min(body_default),
            )
        };
        if hi < lo || (hi == lo && b > a) {
            continue;
        }
        let mean = |event: f64| {
            let start = if reference.key.family == Family::Periodic {
                // Reduce before subtracting, preserving phase far from the time origin.
                (event.rem_euclid(anchor.period) - hi.rem_euclid(anchor.period)) / anchor.period
            } else {
                (event - hi) / anchor.period
            };
            mean_lookup(
                reference.key.family,
                masses,
                start,
                (hi - lo) / anchor.period,
            )
        };
        let Some(c) = mean(candidate) else {
            continue;
        };
        let Some(d) = mean(body_default) else {
            continue;
        };
        let weight =
            reference.weight * anchor.weight * if b == a { 1. } else { (hi - lo) / (b - a) };
        result.candidate += weight * c;
        result.body_default += weight * d;
        result.support += weight;
    }
    result
}

/// CDF of two independent uniforms, reflecting the upper half for stability.
pub(super) fn difference_cdf(value: f64, [a, b]: [f64; 2], [c, d]: [f64; 2]) -> f64 {
    let lower = a - d;
    let upper = b - c;
    if value < lower {
        return 0.;
    }
    if value >= upper {
        return 1.;
    }
    let x = b - a;
    let y = d - c;
    let position = value - lower;
    if x == 0. || y == 0. {
        return position / x.max(y);
    }
    let reflected = position > (x + y) / 2.;
    let z = if reflected {
        x + y - position
    } else {
        position
    };
    let small = x.min(y);
    let cumulative = if z <= small {
        z * z / (2. * x * y)
    } else {
        (z - small / 2.) / x.max(y)
    };
    if reflected {
        1. - cumulative
    } else {
        cumulative
    }
}

pub(crate) fn timing(
    interval: [f64; 2],
    reference: &Reference,
    tau: f64,
) -> ([f64; COORDINATES], [f64; COORDINATES], f64) {
    let mut support = [0.; COORDINATES];
    let mut retained = [0.; COORDINATES];
    let mut coverage = 0.;
    let periodic = reference.key.family == Family::Periodic;
    let [origin, end] = interval;
    let width = end - origin;
    for anchor in &reference.anchors {
        let Some([a, b]) = anchor.interval else {
            continue;
        };
        if anchor.weight == 0. {
            continue;
        }
        coverage += anchor.weight;
        let low = origin - b;
        let high = end - a;
        if low == high {
            let u = low / anchor.period;
            let bin = if periodic {
                ((u.rem_euclid(1.) * BINS as f64).floor() as usize).min(BINS - 1)
            } else if (0. ..=4.).contains(&u) {
                ((u * BINS as f64 / 4.).floor() as usize).min(BINS - 1)
            } else {
                BINS
            };
            support[bin] += anchor.weight;
            retained[bin] += anchor.weight;
            continue;
        }
        let mut inside = [0.; BINS];
        let mut kept = [0.; BINS];
        let cycles = if periodic {
            (low / anchor.period).floor() as i64..=(high / anchor.period).floor() as i64
        } else {
            0..=0
        };
        for cycle in cycles {
            for bin in 0..BINS {
                let left = if periodic {
                    (cycle as f64 + bin as f64 / BINS as f64) * anchor.period
                } else {
                    4. * anchor.period * bin as f64 / BINS as f64
                };
                let right = left + anchor.period / BINS as f64 * if periodic { 1. } else { 4. };
                inside[bin] += (difference_cdf(right, interval, [a, b])
                    - difference_cdf(left, interval, [a, b]))
                .max(0.);
                if width == 0. {
                    continue;
                }
                let c = a - origin;
                let d = b - origin;
                let mut points = [
                    0.,
                    width,
                    (c + left).clamp(0., width),
                    (c + right).clamp(0., width),
                    (d + left).clamp(0., width),
                    (d + right).clamp(0., width),
                ];
                points.sort_unstable_by(f64::total_cmp);
                for pair in points.windows(2) {
                    let [lo, hi] = [pair[0], pair[1]];
                    if hi <= lo {
                        continue;
                    }
                    let (p_lo, p_hi) = if c == d {
                        if lo >= c + left && hi <= c + right {
                            (1., 1.)
                        } else {
                            continue;
                        }
                    } else {
                        (
                            (d.min(lo - left) - c.max(lo - right)).max(0.) / (d - c),
                            (d.min(hi - left) - c.max(hi - right)).max(0.) / (d - c),
                        )
                    };
                    let q = (hi - lo) / tau;
                    let (moment0, moment1) = if q < 1e-3 {
                        let (mut m0, mut m1, mut term) = (0., 0., 1.);
                        for k in 0..8 {
                            m0 += term / f64::from(k + 1);
                            m1 += term / f64::from((k + 1) * (k + 2));
                            term *= -q / f64::from(k + 1);
                        }
                        (m0, m1)
                    } else {
                        let m0 = -(-q).exp_m1() / q;
                        (m0, (1. - m0) / q)
                    };
                    kept[bin] += ((hi - lo) / width
                        * ((hi - width) / tau).exp()
                        * (p_lo * moment0 + (p_hi - p_lo) * moment1))
                        .max(0.);
                }
            }
        }
        for bin in 0..BINS {
            support[bin] += anchor.weight * inside[bin];
            retained[bin] += anchor.weight * if width == 0. { inside[bin] } else { kept[bin] };
        }
        if !periodic {
            support[BINS] += anchor.weight * (1. - inside.iter().sum::<f64>()).max(0.);
            let total = if width == 0. {
                1.
            } else {
                -(-width / tau).exp_m1() * tau / width
            };
            retained[BINS] += anchor.weight
                * (total
                    - if width == 0. {
                        inside.iter().sum::<f64>()
                    } else {
                        kept.iter().sum::<f64>()
                    })
                .max(0.);
        }
    }
    (support, retained, coverage)
}

impl Bank {
    pub(crate) fn reset(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.traces.clear();
        self.last = None;
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.traces.iter().map(|t| t.key)
    }

    pub(crate) fn retain(&mut self, keep: impl Fn(Key) -> bool) -> usize {
        let before = self.traces.len();
        self.traces.retain(|t| keep(t.key));
        before - self.traces.len()
    }
    pub(crate) fn new(
        epoch: u64,
        parameters: Parameters,
        capacity: usize,
    ) -> Result<Self, &'static str> {
        if !(1..=REFERENCES).contains(&capacity)
            || [parameters.tau, parameters.kappa, parameters.strength_max]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err("positive private trace parameters and capacity in 1..=16 required");
        }
        Ok(Self {
            epoch,
            parameters,
            capacity,
            traces: Vec::with_capacity(capacity + REFERENCES),
            last: None,
        })
    }

    pub(crate) fn probabilities(&self, key: Key, head: Head) -> Option<[f64; COORDINATES]> {
        let trace = self.traces.iter().find(|t| t.key == key)?;
        let values = trace.mass[head as usize];
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if maximum == f64::NEG_INFINITY {
            return None;
        }
        let mut masses = values.map(|v| (v - maximum).exp());
        let total: f64 = masses.iter().sum();
        for value in &mut masses {
            *value /= total;
        }
        Some(masses)
    }

    #[cfg(test)]
    pub(crate) fn lookup(&self, key: Key, head: Head, coordinate: f64) -> Option<f64> {
        let masses = self.probabilities(key, head)?;
        lookup_mass(key.family, &masses, coordinate)
    }

    pub(crate) fn observe(&mut self, outcome: Outcome<'_>) -> Result<Receipt, &'static str> {
        let Outcome {
            id,
            head,
            interval,
            references,
            observed_fraction,
            retained,
            confirmed,
        } = outcome;
        if references.len() > REFERENCES
            || !observed_fraction.is_finite()
            || !(0. ..=1.).contains(&observed_fraction)
            || references.iter().enumerate().any(|(i, r)| {
                !r.weight.is_finite()
                    || !(0. ..=1.).contains(&r.weight)
                    || references[..i].iter().any(|other| other.key == r.key)
                    || r.anchors.iter().any(|a| {
                        !a.weight.is_finite()
                            || !(0. ..=1.).contains(&a.weight)
                            || a.interval.is_some_and(|v| {
                                v.iter().any(|v| !v.is_finite())
                                    || v[0] > v[1]
                                    || !a.period.is_finite()
                                    || a.period <= 0.
                            })
                    })
                    || r.anchors.iter().map(|a| a.weight).sum::<f64>() > 1. + 1e-12
            })
            || references.iter().map(|r| r.weight).sum::<f64>() > 1. + 1e-12
        {
            return Err(
                "unique bounded references, normalized frozen weights and valid timing support required",
            );
        }
        let mut receipt = Receipt {
            applied: false,
            credits: [0.; REFERENCES],
            unassigned: 1.,
            removed: [None; REFERENCES],
            evicted: [None; REFERENCES],
        };
        let Some([start, end]) = interval.filter(|_| confirmed) else {
            return Ok(receipt);
        };
        if !start.is_finite() || !end.is_finite() || start > end {
            return Err("finite ordered observed outcome interval required");
        }
        let order = (end, head, id);
        if self.last.is_some_and(|last| order <= last) {
            return Err("duplicate or out-of-order sealed private outcome");
        }
        let mut additions = [[0.; COORDINATES]; REFERENCES];
        for (i, reference) in references.iter().enumerate() {
            if reference.key.epoch != self.epoch || !retained.contains(&reference.key) {
                continue;
            }
            let (_, values, coverage) = timing([start, end], reference, self.parameters.tau);
            receipt.credits[i] = observed_fraction * reference.weight * coverage;
            if coverage > 0. {
                additions[i] = values.map(|v| receipt.credits[i] * v / coverage);
            }
        }
        let total: f64 = receipt.credits.iter().sum();
        receipt.unassigned = (1. - total).max(0.);
        let mut removed = 0;
        self.traces.retain(|t| {
            let keep = t.key.epoch == self.epoch && retained.contains(&t.key);
            if !keep {
                receipt.removed[removed] = Some(t.key);
                removed += 1;
            }
            keep
        });
        for (i, reference) in references.iter().enumerate() {
            if receipt.credits[i] > 0. && !self.traces.iter().any(|t| t.key == reference.key) {
                self.traces.push(Trace {
                    key: reference.key,
                    mass: [[f64::NEG_INFINITY; COORDINATES]; 2],
                    end,
                });
            }
        }
        for trace in &mut self.traces {
            let index = references.iter().position(|r| r.key == trace.key);
            let own = index.map_or(0., |i| receipt.credits[i]);
            let interference = if self.parameters.interference {
                (total - own).max(0.) / self.parameters.kappa
            } else {
                0.
            };
            for (h, values) in trace.mass.iter_mut().enumerate() {
                for (bin, value) in values.iter_mut().enumerate() {
                    *value -= (end - trace.end) / self.parameters.tau;
                    let added = if h == head as usize {
                        index.map_or(0., |i| additions[i][bin])
                    } else {
                        0.
                    };
                    if added > 0. {
                        let new = added.ln();
                        *value = (value.max(new) + (value.min(new) - value.max(new)).exp().ln_1p())
                            .min(self.parameters.strength_max.ln());
                    }
                    *value -= interference;
                }
            }
            trace.end = end;
        }
        let mut evicted = 0;
        while self.traces.len() > self.capacity {
            let victim = self
                .traces
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let mass = |t: &Trace| {
                        let maximum = t
                            .mass
                            .iter()
                            .flatten()
                            .copied()
                            .fold(f64::NEG_INFINITY, f64::max);
                        if maximum == f64::NEG_INFINITY {
                            return maximum;
                        }
                        maximum
                            + t.mass
                                .iter()
                                .flatten()
                                .map(|v| (v - maximum).exp())
                                .sum::<f64>()
                                .ln()
                    };
                    mass(a).total_cmp(&mass(b)).then(a.key.cmp(&b.key))
                })
                .unwrap()
                .0;
            receipt.evicted[evicted] = Some(self.traces.remove(victim).key);
            evicted += 1;
        }
        self.last = Some(order);
        receipt.applied = true;
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests;
