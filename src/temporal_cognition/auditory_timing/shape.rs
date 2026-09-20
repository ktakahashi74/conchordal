//! Exact plateau modes and continuous uncertainty-aware residual distances.

use super::{BINS, Family, History, Record, Summary};

#[derive(Clone, Copy)]
pub(super) struct Shape {
    bins: [u8; BINS / 2],
    count: usize,
    dispersion: Option<f64>,
}

impl Shape {
    fn modes(masses: &[f64; BINS], periodic: bool) -> Self {
        let mut shape = Self {
            bins: [0; BINS / 2],
            count: 0,
            dispersion: None,
        };
        if masses.iter().all(|v| *v == masses[0]) {
            return shape;
        }
        for bin in 0..BINS {
            let mass = masses[bin];
            if mass <= 0. || ((periodic || bin > 0) && masses[(bin + BINS - 1) % BINS] >= mass) {
                continue;
            }
            let mut next = bin + 1;
            while (periodic || next < BINS) && masses[next % BINS] == mass {
                next += 1;
            }
            if (!periodic && next == BINS) || masses[next % BINS] < mass {
                shape.bins[shape.count] = bin as u8;
                shape.count += 1;
            }
        }
        shape.bins[..shape.count].sort_unstable_by(|a, b| {
            masses[*b as usize]
                .total_cmp(&masses[*a as usize])
                .then(a.cmp(b))
        });
        shape
    }
}

impl Record {
    fn residual(self, bins: &[u8], periodic: bool) -> f64 {
        let [a, b] = self.target;
        let [c, d] = self.anchor;
        let low = a - d;
        let high = b - c;
        let span = if periodic { 1. } else { 4. };
        let scale = if periodic { 0.5 } else { 4. };
        let position = |bin: u8| (f64::from(bin) + 0.5) * span / BINS as f64;
        if high == low {
            return bins
                .iter()
                .map(|&bin| {
                    let mut delta = (low - position(bin)).abs();
                    if periodic {
                        delta = ((delta + 0.5).rem_euclid(1.) - 0.5).abs();
                    }
                    (delta / scale).powi(2)
                })
                .fold(f64::INFINITY, f64::min);
        }
        let x = b - a;
        let y = d - c;
        let mut knots = [low, low + x.min(y), low + x.max(y), high];
        // Clipping removes only roundoff outside the original support.
        for knot in &mut knots {
            *knot = knot.clamp(low, high);
        }
        let density = |value: f64| {
            if x == 0. || y == 0. {
                return 1. / (high - low);
            }
            let z = value - low;
            z.min(x).min(y).min(high - value).max(0.) / (x * y)
        };
        let mut ordered = [0; BINS / 2];
        ordered[..bins.len()].copy_from_slice(bins);
        ordered[..bins.len()].sort_unstable();
        let ordered = &ordered[..bins.len()];
        let (first, last) = if periodic {
            (low.floor() as i64 - 1, high.floor() as i64 + 1)
        } else {
            (0, 0)
        };
        let mut integral = 0.;
        for cycle in first..=last {
            for (i, &bin) in ordered.iter().enumerate() {
                let center = position(bin) + cycle as f64;
                let before = if i > 0 {
                    position(ordered[i - 1]) + cycle as f64
                } else if periodic {
                    position(*ordered.last().unwrap()) + cycle as f64 - 1.
                } else {
                    f64::NEG_INFINITY
                };
                let after = if i + 1 < ordered.len() {
                    position(ordered[i + 1]) + cycle as f64
                } else if periodic {
                    position(ordered[0]) + cycle as f64 + 1.
                } else {
                    f64::INFINITY
                };
                let left = ((before + center) / 2.).max(low);
                let right = ((after + center) / 2.).min(high);
                if left >= right {
                    continue;
                }
                for pair in knots.windows(2) {
                    let l = left.max(pair[0]);
                    let r = right.min(pair[1]);
                    if l >= r {
                        continue;
                    }
                    let m = l + (r - l) / 2.;
                    let value = |v: f64| density(v) * ((v - center) / scale).powi(2);
                    // On this cell the PDF is linear and squared distance is quadratic.
                    integral += (r - l) / 6. * (value(l) + 4. * value(m) + value(r));
                }
            }
        }
        integral
    }
}

impl History {
    pub(super) fn describe(&self, summary: &mut Summary) {
        if !summary.supported {
            return;
        }
        let periodic = summary.family == Family::Periodic;
        let shape = self
            .shape
            .get()
            .filter(|(family, _)| *family == periodic)
            .map(|(_, s)| s)
            .unwrap_or_else(|| {
                let mut shape = Shape::modes(&summary.bins, periodic);
                if shape.count > 0 {
                    let residual = self
                        .records
                        .iter()
                        .map(|r| r.weight * r.residual(&shape.bins[..shape.count], periodic))
                        .sum::<f64>()
                        / summary.retained_weight;
                    shape.dispersion = residual.is_finite().then_some(residual);
                }
                self.shape.set(Some((periodic, shape)));
                shape
            });
        summary.mode_bins = std::array::from_fn(|i| (i < shape.count).then_some(shape.bins[i]));
        summary.mode_count = Some(shape.count);
        summary.residual_dispersion = shape.dispersion;
    }
}

impl Summary {
    pub(super) fn features(self, selected_weight: f64) -> [Option<f64>; 14] {
        let mut values = [None; 14];
        if !self.supported {
            return values;
        }
        let periodic = self.family == Family::Periodic;
        for (i, bin) in self.mode_bins.into_iter().enumerate() {
            let Some(bin) = bin else { continue };
            let coordinate = (f64::from(bin) + 0.5) / BINS as f64;
            let (sin, cos) = (std::f64::consts::TAU * coordinate).sin_cos();
            values[i * 4..i * 4 + 4].copy_from_slice(&[
                Some(if periodic { cos } else { 0. }),
                Some(if periodic { sin } else { 0. }),
                Some(if periodic { 0. } else { coordinate }),
                Some(self.bins[bin as usize]),
            ]);
        }
        values[8] = Some(f64::from(periodic));
        values[9] = Some(selected_weight);
        values[10] = self.residual_dispersion;
        values[11] = self
            .mode_count
            .map(|n| (n as f64).ln_1p() / (BINS as f64 / 2.).ln_1p());
        values[12] = Some(self.overflow);
        values[13] = Some(self.coverage);
        values
    }
}

#[cfg(test)]
mod tests;
