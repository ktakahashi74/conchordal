//! Projective density update; C/D retain the heard first-order gradients.
use super::Snapshot;

/// Conservative representation scales, not perceptual tolerances.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Errors {
    pub mass: f64,
    pub focus: f64,
    pub conflict: f64,
    pub shape: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Observer {
    Mixed,
    Excluded { kind: u8, id: u64, generation: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Baseline {
    pub observer: Observer,
    pub end_sample: u64,
    pub coordinate_version: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Preview {
    pub baseline: Baseline,
    pub mass: f64,
    pub focus: f64,
    pub conflict: f64,
    pub novelty: f64,
    pub p: Vec<f64>,
    pub shape_valid: bool,
    pub partials_outside: bool,
    pub errors: Errors,
}

impl Preview {
    pub(crate) fn new(snapshot: &Snapshot) -> Self {
        Self {
            baseline: snapshot.baseline(Observer::Mixed),
            mass: 0.,
            focus: 0.,
            conflict: 0.,
            novelty: 0.,
            p: vec![0.; snapshot.p.len()],
            shape_valid: false,
            partials_outside: false,
            errors: Errors::default(),
        }
    }
}

impl Snapshot {
    pub(crate) fn numerical_errors(&self) -> Errors {
        self.errors
    }
    pub(super) fn representation_errors(&self) -> Errors {
        if self.mass == 0. {
            return Errors::default();
        }
        let g = &self.geometry;
        // Standard gamma_n accumulation bound; coefficients are the producer's
        // declared floating representation. Count multiply/add contributions,
        // normalization, and a conservative radix-2 complex FFT operation bound.
        let projection_ops =
            2 * g.columns.iter().map(Vec::len).sum::<usize>() + g.space.n_bins() + self.p.len();
        let unit = f64::from(f32::EPSILON) / 2.;
        let gamma = projection_ops as f64 * unit / (1. - projection_ops as f64 * unit);
        let shape_l1 = (2. * gamma / (1. - gamma)).min(2.);
        let fft_ops = 8 * self.frequency.len() * self.frequency.len().ilog2() as usize;
        let fft_unit = f64::EPSILON / 2.;
        let fft_error = fft_ops as f64 * fft_unit / (1. - fft_ops as f64 * fft_unit);
        let quadratic = (2. * shape_l1 + shape_l1 * shape_l1 + fft_error).min(1.);
        Errors {
            mass: self.mass * gamma,
            focus: quadratic,
            conflict: quadratic,
            shape: (shape_l1 / 2.).sqrt(),
        }
    }
    pub(crate) fn baseline(&self, observer: Observer) -> Baseline {
        // Coordinate identity is stable across observers and allocation addresses.
        let mut version = 0xcbf2_9ce4_8422_2325u64;
        for word in [
            u64::from(self.geometry.root_start_log2.to_bits()),
            u64::from(self.geometry.space.step().to_bits()),
            self.p.len() as u64,
        ] {
            for byte in word.to_le_bytes() {
                version = (version ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
        }
        Baseline {
            observer,
            end_sample: self.end_sample,
            coordinate_version: version,
        }
    }

    /// Caller owns preallocated output; no graph, FFT, rendering or random draw.
    pub(crate) fn preview(
        &self,
        observer: Observer,
        log2: f32,
        template: &[(isize, f32)],
        out: &mut Preview,
    ) {
        assert_eq!(out.p.len(), self.p.len());
        out.baseline = self.baseline(observer);
        out.partials_outside = false;
        let g = &self.geometry;
        let step = f64::from(g.space.step());
        let amount = self.mass / step;
        for (r, p) in out.p.iter_mut().zip(&self.p) {
            *r = p * amount;
        }
        let position = (log2 - g.space.centers_log2[0]) / g.space.step();
        for &(offset, density) in template {
            assert!(density.is_finite() && density >= 0.);
            if density == 0. {
                continue;
            }
            let at = position + offset as f32;
            if !at.is_finite() || at < 0. || at > (g.space.n_bins() - 1) as f32 {
                // No projection column exists here. Do not invent its mass.
                out.partials_outside = true;
                continue;
            }
            let lo = at.floor() as usize;
            let hi = (lo + 1).min(g.space.n_bins() - 1);
            let fraction = f64::from(at - lo as f32);
            for (column, weight) in [(lo, 1. - fraction), (hi, fraction)] {
                for &(root, a) in &g.columns[column] {
                    out.p[root] += f64::from(density) * weight * a;
                }
            }
        }
        let total = out.p.iter().sum::<f64>();
        out.mass = total * step;
        out.shape_valid = total > 0. && !out.partials_outside;
        if total > 0. {
            for p in &mut out.p {
                *p /= total;
            }
        }
        let [dc, dd, novelty] = self.features_at(log2, template);
        out.focus = self.focus + dc;
        out.conflict = self.conflict + dd;
        out.novelty = novelty;
        out.errors = self.numerical_errors();
        let operations = template
            .len()
            .saturating_mul(self.p.len())
            .saturating_mul(2)
            + self.p.len();
        let unit = f64::EPSILON / 2.;
        let rounding = operations as f64 * unit / (1. - operations as f64 * unit);
        out.errors.mass += out.mass * rounding;
        out.errors.focus += (self.focus.abs() + dc.abs()) * rounding;
        out.errors.conflict += (self.conflict.abs() + dd.abs()) * rounding;
        out.errors.shape += rounding.sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        harmonic_support::Geometry, harmonicity_kernel::HarmonicityKernel, log2space::Log2Space,
    };

    #[test]
    fn projective_preview_agrees_with_density_projection_and_tangent() {
        let space = Log2Space::new(80., 2000., 48);
        let geometry = Geometry::new(&space, &HarmonicityKernel::new(&space, Default::default()));
        let mut base = Snapshot::new(geometry.clone());
        let mut density = vec![0.; space.n_bins()];
        let x = space.n_bins() / 3;
        let y = space.n_bins() * 2 / 3;
        density[x] = 0.25;
        base.observe_projection(&density, None, 1e-6, 512);
        let mut out = Preview::new(&base);
        let observer = Observer::Excluded {
            kind: 0,
            id: 9,
            generation: 3,
        };
        base.preview(observer, space.centers_log2[y], &[(0, 2.)], &mut out);
        let mut exact = Snapshot::new(geometry.clone());
        density[y] = 2.;
        exact.observe_projection(&density, None, 1e-6, 1024);
        assert!(out.shape_valid);
        assert!(out.p.iter().all(|p| *p >= 0.));
        let bound = out.p.len() as f64 * f64::from(f32::EPSILON);
        assert!((out.p.iter().sum::<f64>() - 1.).abs() <= bound);
        assert!(
            out.p
                .iter()
                .zip(&exact.p)
                .all(|(p, q)| (p - q).abs() <= bound)
        );
        assert!((out.mass - exact.mass).abs() <= bound * exact.mass);
        assert_eq!(out.baseline, base.baseline(observer));
        assert_eq!(out.baseline.end_sample, 512);
        let r = base.mass / f64::from(space.step());
        let mut previous = f64::INFINITY;
        for q in [0.001f32, 0.0005, 0.00025] {
            base.preview(observer, space.centers_log2[y], &[(0, q)], &mut out);
            let mut derivative = base
                .p
                .iter()
                .map(|p| -p * geometry.column_sum[y] / r)
                .collect::<Vec<_>>();
            for &(i, a) in &geometry.columns[y] {
                derivative[i] += a / r;
            }
            let error = out
                .p
                .iter()
                .zip(&base.p)
                .zip(&derivative)
                .map(|((p, b), d)| ((p - b) / f64::from(q) - d).abs())
                .sum::<f64>();
            assert!(error < previous);
            previous = error;
        }
    }

    #[test]
    fn silent_baseline_uses_candidate_support_and_empty_candidate_is_unknown() {
        let space = Log2Space::new(80., 2000., 48);
        let geometry = Geometry::new(&space, &HarmonicityKernel::new(&space, Default::default()));
        let mut base = Snapshot::new(geometry);
        base.observe_projection(&vec![0.; space.n_bins()], None, 1e-6, 512);
        let mut out = Preview::new(&base);
        base.preview(
            Observer::Mixed,
            space.centers_log2[20],
            &[(0, 1.)],
            &mut out,
        );
        assert!(out.shape_valid && out.mass > 0.);
        assert_eq!(out.focus, 0.);
        assert_eq!(out.conflict, 0.);
        base.preview(Observer::Mixed, space.centers_log2[20], &[], &mut out);
        assert!(!out.shape_valid);
        assert_eq!(out.mass, 0.);
        base.preview(
            Observer::Mixed,
            space.centers_log2[0],
            &[(-1, 1.)],
            &mut out,
        );
        assert!(out.partials_outside && !out.shape_valid);
    }
}
