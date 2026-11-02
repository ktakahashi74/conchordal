//! First-order support fields read from the heard sound.
use super::harmonicity_kernel::HarmonicityKernel;
use super::log2space::Log2Space;
use super::roughness_kernel::erb_grid;
use rustfft::{Fft, FftPlanner, num_complex::Complex64};
use std::sync::Arc;
pub(crate) mod body_density;
pub(crate) mod preview;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Weights(pub [f64; 3]);

impl Weights {
    pub(crate) fn active(self) -> bool {
        self.0.iter().any(|&w| w != 0.)
    }

    pub(crate) fn set(&mut self, axis: usize, value: f64) -> bool {
        if !value.is_finite() || (axis == 2 && value < 0.) {
            return false;
        }
        let changed = self.0[axis] != value;
        self.0[axis] = value;
        changed
    }

    pub(crate) fn value(self, features: [f64; 3]) -> f64 {
        let scale = self.0.iter().map(|w| w.abs()).fold(0., f64::max);
        if scale == 0. {
            return 0.;
        }
        let value = self
            .0
            .iter()
            .zip(features)
            .map(|(w, x)| (w / scale) * x)
            .sum::<f64>();
        (value * scale).clamp(-f64::MAX, f64::MAX)
    }
}

/// Shared projection columns and zero-padded root-space filters.
pub(crate) struct Geometry {
    pub(crate) space: Log2Space,
    pub(crate) root_start_log2: f32,
    roots: usize,
    columns: Vec<Vec<(usize, f64)>>,
    column_sum: Vec<f64>,
    column_norm: Vec<f64>,
    du: Vec<f32>,
    kernels: [Vec<Complex64>; 2],
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}
impl std::fmt::Debug for Geometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupportGeometry")
            .field("space", &self.space)
            .finish_non_exhaustive()
    }
}
impl Geometry {
    pub(crate) fn new(space: &Log2Space, h: &HarmonicityKernel) -> Arc<Self> {
        let n = space.n_bins();
        let roots = h.root_buffer_len(space);
        let padding = (roots - n) / 2;
        let mut linear = h.clone();
        linear.params.gamma_root = 1.;
        let mut input = vec![0.; n];
        let mut smeared = vec![0.; n];
        let mut projected = vec![0.; roots];
        let mut output = vec![0.; n];
        let mut columns = Vec::with_capacity(n);
        for i in 0..n {
            input[i] = 1.;
            linear.potential_h_into(&input, space, &mut smeared, &mut projected, &mut output);
            columns.push(
                projected
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &v)| (v > 0.).then_some((j, f64::from(v))))
                    .collect::<Vec<_>>(),
            );
            input[i] = 0.;
        }
        let smooth = h.support_smoothing();
        let harmonics = h.params.num_harmonics.max(1);
        let extent = ((harmonics as f32).log2() * space.bins_per_oct as f32).ceil() as usize;
        let mut template = vec![0.; extent + smooth.len() + 1];
        for harmonic in 1..=harmonics {
            let position = (harmonic as f32).log2() * space.bins_per_oct as f32;
            let lo = position.floor() as usize;
            let fraction = position - lo as f32;
            let weight = (harmonic as f32).powf(-h.params.rho_common_root);
            for (j, &v) in smooth.iter().enumerate() {
                template[lo + j] += f64::from(v * weight * (1. - fraction));
                template[lo + j + 1] += f64::from(v * weight * fraction);
            }
        }
        let len = (roots + 2 * template.len()).next_power_of_two();
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(len);
        let inverse = planner.plan_fft_inverse(len);
        let patterns = [
            smooth.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            template,
        ];
        let kernels = patterns.map(|pattern| {
            let norm: f64 = pattern.iter().map(|v| v * v).sum();
            let mut values = vec![Complex64::default(); len];
            for (out, v) in values.iter_mut().zip(pattern) {
                out.re = v;
            }
            forward.process(&mut values);
            for v in &mut values {
                *v = Complex64::new(v.norm_sqr() / norm, 0.);
            }
            values
        });
        Arc::new(Self {
            space: space.clone(),
            root_start_log2: space.centers_log2[0] - padding as f32 * space.step(),
            roots,
            column_sum: columns
                .iter()
                .map(|c| c.iter().map(|(_, v)| v).sum())
                .collect(),
            column_norm: columns
                .iter()
                .map(|c| c.iter().map(|(_, v)| v * v).sum::<f64>().sqrt())
                .collect(),
            columns,
            du: erb_grid(space).1,
            kernels,
            forward,
            inverse,
        })
    }
}

/// One observed reference, with gC, gD and k on its Log2Space.
#[derive(Clone, Debug)]
pub(crate) struct Snapshot {
    errors: preview::Errors,
    pub(crate) geometry: Arc<Geometry>,
    pub(crate) end_sample: u64,
    pub(crate) p: Vec<f64>,
    pub(crate) mass: f64,
    pub(crate) focus: f64,
    pub(crate) conflict: f64,
    pub(crate) g_c: Vec<f64>,
    pub(crate) g_d: Vec<f64>,
    pub(crate) k: Vec<f64>,
    density: Vec<f32>,
    sum: f64,
    root_work: Vec<f64>,
    frequency: Vec<Complex64>,
    work: Vec<Complex64>,
    fft_scratch: Vec<Complex64>,
}
impl Snapshot {
    pub(crate) fn new(geometry: Arc<Geometry>) -> Self {
        let n = geometry.space.n_bins();
        let len = geometry.kernels[0].len();
        Self {
            errors: preview::Errors::default(),
            end_sample: 0,
            p: vec![0.; geometry.roots],
            root_work: vec![0.; geometry.roots],
            mass: 0.,
            focus: 0.,
            conflict: 0.,
            g_c: vec![0.; n],
            g_d: vec![0.; n],
            k: vec![0.; n],
            density: vec![0.; n],
            sum: 0.,
            frequency: vec![Complex64::default(); len],
            work: vec![Complex64::default(); len],
            fft_scratch: vec![
                Complex64::default();
                geometry
                    .forward
                    .get_inplace_scratch_len()
                    .max(geometry.inverse.get_inplace_scratch_len())
            ],
            geometry,
        }
    }
    #[cfg(test)]
    pub(crate) fn observe(&mut self, density: &[f32], eps: f32, end_sample: u64) {
        self.observe_projection(density, None, eps, end_sample);
    }
    pub(crate) fn observe_projection(
        &mut self,
        density: &[f32],
        projected: Option<&[f32]>,
        eps: f32,
        end_sample: u64,
    ) {
        let g = &self.geometry;
        g.space
            .assert_scan_len_named(density, "support_density_scan");
        self.end_sample = end_sample;
        self.p.fill(0.);
        self.root_work.fill(0.);
        let mut erb_mass = 0.;
        self.sum = 0.;
        for (i, &v) in density.iter().enumerate() {
            let v = super::float::sanitize_nonnegative_finite(v);
            self.density[i] = v;
            let v = f64::from(v);
            self.sum += v;
            erb_mass += v * f64::from(g.du[i]);
            if v == 0. {
                continue;
            }
            for &(j, a) in &g.columns[i] {
                if projected.is_none() {
                    self.p[j] += v * a;
                }
                self.root_work[j] += v * a / g.column_norm[i];
            }
        }
        if let Some(projected) = projected {
            assert_eq!(projected.len(), self.p.len());
            for (r, &v) in self.p.iter_mut().zip(projected) {
                *r = f64::from(v);
            }
        }
        let sum: f64 = self.p.iter().sum();
        self.mass = sum * f64::from(g.space.step());
        for (i, out) in self.k.iter_mut().enumerate() {
            *out = g.columns[i]
                .iter()
                .map(|&(j, a)| a * self.root_work[j])
                .sum::<f64>()
                / g.column_norm[i];
        }
        if sum == 0. {
            self.errors = preview::Errors::default();
            self.focus = 0.;
            self.conflict = 0.;
            self.g_c.fill(0.);
            self.g_d.fill(0.);
            return;
        }
        for v in &mut self.p {
            *v /= sum;
        }
        let floor = f64::from(eps.max(1e-12));
        let alpha = erb_mass / erb_mass.max(floor);
        self.frequency.fill(Complex64::default());
        for (out, &v) in self.frequency.iter_mut().zip(&self.p) {
            out.re = v;
        }
        g.forward
            .process_with_scratch(&mut self.frequency, &mut self.fft_scratch);
        for axis in 0..2 {
            for ((out, value), kernel) in self
                .work
                .iter_mut()
                .zip(&self.frequency)
                .zip(&g.kernels[axis])
            {
                *out = value * kernel;
            }
            g.inverse
                .process_with_scratch(&mut self.work, &mut self.fft_scratch);
            let scale = 1. / self.work.len() as f64;
            let form = self
                .p
                .iter()
                .zip(&self.work)
                .map(|(&v, w)| v * w.re * scale)
                .sum::<f64>()
                .clamp(0., 1.);
            let (value, gradient, sign) = if axis == 0 {
                (&mut self.focus, &mut self.g_c, 1.)
            } else {
                (&mut self.conflict, &mut self.g_d, -1.)
            };
            *value = alpha * alpha * if axis == 0 { form } else { 1. - form };
            for (i, out) in gradient.iter_mut().enumerate() {
                let up = g.columns[i]
                    .iter()
                    .map(|&(j, a)| a * self.work[j].re * scale)
                    .sum::<f64>();
                *out = sign * alpha * alpha * 2. / sum * (up - form * g.column_sum[i]);
                // At the floor, additions take the alpha=1 (right) derivative.
                if erb_mass < floor {
                    *out += 2. * alpha * f64::from(g.du[i]) / floor
                        * if axis == 0 { form } else { 1. - form };
                }
            }
        }
        self.errors = self.representation_errors();
    }
    /// Integrate first-order fields and related novelty over declared partials.
    pub(crate) fn candidate(&self, q: &[(usize, f32)]) -> [f64; 3] {
        self.partial_features(q.iter().copied())
    }
    fn partial_features(&self, q: impl Iterator<Item = (usize, f32)>) -> [f64; 3] {
        let mut result = [0.; 3];
        let mut sum = 0.;
        for (i, q) in q {
            assert!(q.is_finite() && q >= 0.);
            let q = f64::from(q);
            result[0] += q * self.g_c[i];
            result[1] += q * self.g_d[i];
            result[2] += self.k[i] * (q - f64::from(self.density[i])).max(0.);
            sum += q;
        }
        result[2] = if self.sum + sum > 0. {
            4. * result[2] / (self.sum + sum).powi(2)
        } else {
            0.
        };
        result
    }
    pub(crate) fn template_features(&self, bin: usize, template: &[(isize, f32)]) -> [f64; 3] {
        self.partial_features(template.iter().filter_map(|&(offset, q)| {
            let index = bin as isize + offset;
            (index >= 0 && index < self.density.len() as isize).then_some((index as usize, q))
        }))
    }
    fn features_at(&self, log2: f32, template: &[(isize, f32)]) -> [f64; 3] {
        let space = &self.geometry.space;
        let position = (log2 - space.centers_log2[0]) / space.step();
        let mut result = [0.; 3];
        let mut sum = 0.;
        for &(offset, q) in template {
            let at = position + offset as f32;
            if at < 0. || at > (space.n_bins() - 1) as f32 {
                continue;
            }
            let lo = at.floor() as usize;
            let hi = (lo + 1).min(space.n_bins() - 1);
            let fraction = f64::from(at - lo as f32);
            let interpolate = |a: f64, b: f64| a + (b - a) * fraction;
            let q = f64::from(q);
            result[0] += q * interpolate(self.g_c[lo], self.g_c[hi]);
            result[1] += q * interpolate(self.g_d[lo], self.g_d[hi]);
            result[2] += interpolate(self.k[lo], self.k[hi])
                * (q - interpolate(f64::from(self.density[lo]), f64::from(self.density[hi])))
                    .max(0.);
            sum += q;
        }
        result[2] = if self.sum + sum > 0. {
            4. * result[2] / (self.sum + sum).powi(2)
        } else {
            0.
        };
        result
    }
    pub(crate) fn template_value(
        &self,
        log2: f32,
        template: &[(isize, f32)],
        weights: Weights,
    ) -> f64 {
        weights.value(self.features_at(log2, template))
    }
    pub(crate) fn single_partial_value(&self, log2: f32, density: f32, weights: Weights) -> f64 {
        self.template_value(log2, &[(0, density)], weights)
    }
    pub(crate) fn single_partial_scan(&self, density: f32, scans: &mut [Vec<f64>; 3]) {
        for scan in scans.iter() {
            self.geometry
                .space
                .assert_scan_len_named(scan, "support_partial_scan");
        }
        for i in 0..self.density.len() {
            for (out, v) in scans.iter_mut().zip(self.candidate(&[(i, density)])) {
                out[i] = v;
            }
        }
    }
}
#[cfg(test)]
mod tests;
