//! Shared density-to-H/R law for live analysis, body analysis, and exact LOO.

use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::log2space::Log2Space;
use crate::core::psycho_state::normalize_density_into;
use crate::core::roughness_kernel::{RoughnessGeometry, RoughnessKernel, erb_grid};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct PotentialScratch {
    erb: Arc<[f32]>,
    pub(crate) du: Arc<[f32]>,
    pub(super) geometry: Arc<RoughnessGeometry>,
    pub(super) normalized: Vec<f32>,
    smeared: Vec<f32>,
    roots: Vec<f32>,
}

impl PotentialScratch {
    pub(crate) fn new(
        space: &Log2Space,
        harmonicity: &HarmonicityKernel,
        roughness: &RoughnessKernel,
    ) -> Self {
        let (erb, du) = erb_grid(space);
        let erb: Arc<[f32]> = erb.into();
        Self {
            geometry: Arc::new(roughness.prepare_geometry(space, &erb)),
            erb,
            du: du.into(),
            normalized: vec![0.; space.n_bins()],
            smeared: vec![0.; space.n_bins()],
            roots: vec![0.; harmonicity.root_buffer_len(space)],
        }
    }

    pub(crate) fn compute_shape(
        &mut self,
        density: &[f32],
        space: &Log2Space,
        kernels: (&HarmonicityKernel, &RoughnessKernel),
        eps: f32,
        harmonicity: &mut [f32],
        roughness_shape: &mut [f32],
    ) -> f32 {
        space.assert_scan_len_named(density, "subjective_intensity_scan");
        space.assert_scan_len_named(roughness_shape, "r_shape_scan");
        space.assert_scan_len_named(harmonicity, "h_pot_scan");
        let (h, r) = kernels;
        let mass = normalize_density_into(density, &self.du, eps, &mut self.normalized);
        // Apply the existing density-normalization floor to both potentials.
        if !mass.is_finite() || mass <= eps {
            harmonicity.fill(0.);
            roughness_shape.fill(0.);
            return 0.;
        }
        self.roots.resize(h.root_buffer_len(space), 0.);
        h.potential_h_into(
            density,
            space,
            &mut self.smeared,
            &mut self.roots,
            harmonicity,
        );
        if !self.geometry.matches(r, space, &self.erb) {
            self.geometry = Arc::new(r.prepare_geometry(space, &self.erb));
        }
        self.geometry
            .potential_r_into(&self.normalized, space, &self.du, roughness_shape)
    }
}
