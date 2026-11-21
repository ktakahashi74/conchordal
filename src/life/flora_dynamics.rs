use super::synth::{noise_draw, random_unit};
use super::*;
use crate::core::float::{finite_or, sanitize_nonnegative_finite, sanitize01, unit_gaussian};
use crate::core::log2space::sample_scan_linear_log2;

impl Flora {
    pub(super) fn seed_biomass(&mut self, seeding: FloraSeeding, landscape: &Landscape) {
        self.space.assert_scan_len_named(
            &landscape.consonance_density_mass,
            "consonance_density_mass",
        );
        if !self.range.is_empty() {
            for i in self.range.clone() {
                self.biomass_mass_scan[i] = match seeding {
                    FloraSeeding::Uniform => 1.0,
                    FloraSeeding::Consonance => {
                        finite_or(landscape.consonance_density_mass[i], 0.0).max(0.0)
                    }
                };
            }
            if self.biomass_mass_scan.iter().sum::<f32>() == 0.0 {
                self.biomass_mass_scan[self.range.clone()].fill(1.0);
            }
            for i in self.range.clone() {
                self.biomass_mass_scan[i] *= 0.9 + 0.2 * random_unit(&mut self.rng);
            }
            // Divide before scaling so tiny seeding weights cannot overflow; weights that
            // underflowed to zero during the perturbation fall back to uniform.
            let total: f32 = self.biomass_mass_scan[self.range.clone()].iter().sum();
            let uniform = INITIAL_BIOMASS / self.range.len() as f32;
            for i in self.range.clone() {
                self.biomass_mass_scan[i] = if total > 0.0 {
                    self.biomass_mass_scan[i] / total * INITIAL_BIOMASS
                } else {
                    uniform
                };
            }
        }
    }

    pub(crate) fn on_space_change(&mut self, space: &Log2Space) {
        if !self.support.choice_scan.is_empty() {
            self.support.choice_scan.resize(space.n_bins(), 0.);
            self.support.native_scores.resize(space.n_bins(), 0.);
        }
        self.support.active = false;
        if !self.biomass_mass_scan.is_empty() {
            self.assert_scans();
        }
        let old_space = std::mem::replace(&mut self.space, space.clone());
        let old_v = std::mem::replace(&mut self.biomass_mass_scan, vec![0.0; space.n_bins()]);
        let old_u = std::mem::replace(&mut self.resource_state_scan, vec![0.0; space.n_bins()]);
        let was_dormant = self.range.is_empty();
        let start = space.centers_hz.partition_point(|&f| f < self.extent_hz.0);
        let end = space.centers_hz.partition_point(|&f| f <= self.extent_hz.1);
        self.range = start..end;
        self.biomass_consumption_scan = vec![0.0; space.n_bins()];
        self.external_consumption_scan = vec![0.0; space.n_bins()];
        self.external_observed_frame = None;
        self.external_current = false;
        self.growth_level_scan = vec![0.0; space.n_bins()];
        self.harmonicity_state_scan = vec![0.0; space.n_bins()];
        self.scratch_scan = vec![0.0; space.n_bins()];
        self.erb_scan = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();
        self.assert_scans();
        self.consumption_rows.clear();
        self.consumption_weights.clear();
        if !self.range.is_empty() {
            self.resource_state_scan[self.range.clone()].fill(1.0);
            if !was_dormant && !old_v.is_empty() {
                old_space.assert_scan_len_named(&old_v, "biomass_mass_scan");
                old_space.assert_scan_len_named(&old_u, "resource_state_scan");
                let mass_scale = old_space.bins_per_oct as f32 / space.bins_per_oct as f32;
                for i in self.range.clone() {
                    let f = space.centers_hz[i];
                    if f >= old_space.centers_hz[0] && f <= *old_space.centers_hz.last().unwrap() {
                        self.biomass_mass_scan[i] =
                            sample_scan_linear_log2(&old_space, &old_v, f).max(0.0) * mass_scale;
                        self.resource_state_scan[i] =
                            sanitize01(sample_scan_linear_log2(&old_space, &old_u, f));
                    }
                }
            } else {
                self.biomass_mass_scan[self.range.clone()]
                    .fill(INITIAL_BIOMASS / self.range.len() as f32);
            }
            for i in self.range.clone() {
                let lo = self
                    .erb_scan
                    .partition_point(|&x| x < self.erb_scan[i] - 1.5)
                    .max(self.range.start);
                let hi = self
                    .erb_scan
                    .partition_point(|&x| x <= self.erb_scan[i] + 1.5)
                    .min(self.range.end);
                let first = self.consumption_weights.len();
                self.consumption_weights.extend(
                    (lo..hi).map(|j| unit_gaussian(self.erb_scan[j] - self.erb_scan[i], 0.5)),
                );
                let weights = &mut self.consumption_weights[first..];
                let total: f32 = weights.iter().sum();
                for weight in weights.iter_mut() {
                    *weight /= total;
                }
                self.consumption_rows
                    .push((lo, first..self.consumption_weights.len()));
            }
        }
        self.diffusion = 0.5 * (10.0 * space.bins_per_oct as f32 / 1200.0).powi(2);
        self.substeps = (((self.hop as f32 / self.fs) * self.pace)
            * (GROWTH_RATE / 0.05).max(self.diffusion / 0.25))
        .ceil()
        .max(1.0) as usize;

        let mut old_lanes = std::mem::take(&mut self.lanes);
        self.cells = partition_erb_cells(space, &self.erb_scan, self.extent_hz)
            .into_iter()
            .map(|cell| Cell {
                lo_erb: cell.lo_erb,
                hi_erb: cell.hi_erb,
                bins: cell.bins,
                floor: 0.0,
                excess: 0.0,
                tonal_mass: 0.0,
                residual_mass: 0.0,
                peak: None,
            })
            .collect();
        self.tonal_len = 2 * self.cells.len();
        self.residual_end = self.tonal_len + self.cells.len();
        self.lanes = (0..self.tonal_len)
            .map(|_| {
                Lane::new(
                    self.spec.tone == FloraTone::Resonant,
                    self.extent_hz.0.log2(),
                )
            })
            .collect();
        for cell in &self.cells {
            let position = erb_to_hz(0.5 * (cell.lo_erb + cell.hi_erb)).log2();
            let mut lane = Lane::new(true, position);
            lane.tonal = false;
            lane.mode = Mode::Current;
            lane.e = noise_draw(&mut self.rng);
            self.lanes.push(lane);
        }
        for mut lane in old_lanes.drain(..) {
            if lane.mode == Mode::Free || (!lane.tonal && lane.r == 0.0) {
                continue;
            }
            if lane.tonal && !self.range.is_empty() {
                lane.cell = None;
            } else {
                lane.retire();
            }
            self.lanes.push(lane);
        }
        self.noisy_len = 0;
        for lane in &mut self.lanes {
            if lane.noisy {
                lane.noise_index = self.noisy_len;
                self.noisy_len += 1;
            }
        }
        if !self.support.choice_scan.is_empty() {
            self.support
                .scratch
                .reserve(self.hop, self.space.n_bins(), self.lanes.len());
        }
        self.temporal
            .as_mut()
            .unwrap()
            .resize(self.cells.len(), self.lanes.len(), space.n_bins());
        for lane in &mut self.lanes {
            lane.window = 0..0;
            lane.pending_since = None;
        }
        self.order = Vec::with_capacity(self.lanes.len());
        self.noise = vec![0.0; 2 * self.noisy_len * self.hop];
        if let Some(respond) = &mut self.respond {
            respond.reset_grid(space.n_bins());
        }
        self.assert_scans();
        self.refresh_support_template();
    }

    pub(crate) fn observe_external(&mut self, landscape: &Landscape, frame: Option<u64>) {
        self.external_current = false;
        let Some(frame) = frame else {
            return;
        };
        let Some(power) = landscape
            .ecology
            .flora_observations
            .iter()
            .find(|s| s.id == self.id && s.generation == 0)
            .and_then(|s| s.power_scan.as_deref())
            .or_else(|| {
                (self.rendered_samples == 0)
                    .then_some(landscape.ecology.site_resource_power.as_deref())
                    .flatten()
            })
        else {
            return;
        };
        self.space
            .assert_scan_len_named(power, "external_power_scan");
        self.external_current = true;
        if self.external_observed_frame == Some(frame) {
            return;
        }
        // Fixed author unit: nominal sine amplitude 0.03 has mean square 0.00045.
        let scale = self.space.bins_per_oct as f32 / REFERENCE_DENSITY / 0.00045;
        for (row, (lo, weights)) in self.consumption_rows.iter().enumerate() {
            self.external_consumption_scan[self.range.start + row] = self.consumption_weights
                [weights.clone()]
            .iter()
            .enumerate()
            .map(|(j, &w)| w * sanitize_nonnegative_finite(power[lo + j]))
            .sum::<f32>()
                * scale;
        }
        self.external_observed_frame = Some(frame);
    }

    pub(crate) fn advance(&mut self, landscape: &Landscape) {
        self.harmonic.weight = landscape
            .harmonic
            .relations_weight
            .abs()
            .max(landscape.harmonic.continuation_weight.abs());
        self.harmonic.query_version = landscape.harmonic.query_version;
        if landscape.harmonic.relations_weight != 0. || landscape.harmonic.continuation_weight != 0.
        {
            use crate::core::harmonic_support::preview::Observer;
            use crate::temporal_cognition::harmonic_relations::consumer::Use;
            let excluded = self.needs_hr_observer() && self.rendered_samples > 0;
            let snapshot = if excluded {
                landscape
                    .ecology
                    .flora_observations
                    .iter()
                    .find(|s| s.id == self.id && s.generation == 0)
                    .and_then(|s| s.perception.as_ref())
                    .and_then(|s| s.support.clone())
            } else {
                landscape.harmonic.support.clone()
            };
            let baseline_present = snapshot.is_some();
            self.harmonic.availability =
                crate::temporal_cognition::harmonic_relations::runtime::Availability::from_decision(
                    landscape.harmonic.relations.as_deref(),
                );
            Use::refresh(
                &mut self.harmonic.consumer,
                landscape,
                None,
                if excluded {
                    Observer::Excluded {
                        kind: 1,
                        id: self.id,
                        generation: 0,
                    }
                } else {
                    Observer::Mixed
                },
                snapshot,
                Some(self.support.template.clone()),
                0.,
                self.fs,
                landscape.harmonic.now_sample,
            );
            if baseline_present && let Some(use_relations) = &self.harmonic.consumer {
                self.harmonic.availability = use_relations.availability();
            }
            if self.harmonic.cells.capacity() < self.cells.len() {
                self.harmonic
                    .cells
                    .reserve(self.cells.len() - self.harmonic.cells.len());
            }
        } else {
            // Both weights are zero: refresh clears weight, continuation and decision.
            crate::temporal_cognition::harmonic_relations::consumer::Use::refresh(
                &mut self.harmonic.consumer,
                landscape,
                None,
                crate::core::harmonic_support::preview::Observer::Mixed,
                None,
                None,
                0.,
                self.fs,
                landscape.harmonic.now_sample,
            );
        }
        self.support.active = false;
        self.support.action = None;
        if landscape.harmonic.support_weights.active() {
            let snapshot = if !self.needs_hr_observer() || self.rendered_samples == 0 {
                landscape.harmonic.support.as_ref()
            } else {
                landscape
                    .ecology
                    .flora_observations
                    .iter()
                    .find(|s| s.id == self.id && s.generation == 0)
                    .and_then(|s| s.perception.as_ref())
                    .and_then(|s| s.support.as_ref())
            };
            if let Some(snapshot) = snapshot {
                tracing::trace!(
                    support_evaluation = "body_partials",
                    flora_id = self.id,
                    "support Flora choice scan"
                );
                self.space
                    .assert_scan_len_named(&self.support.choice_scan, "flora_support_choice_scan");
                for (i, value) in self.support.choice_scan.iter_mut().enumerate() {
                    *value = (f64::from(landscape.consonance_representation.beta)
                        * landscape
                            .harmonic
                            .support_weights
                            .value(snapshot.template_features(i, &self.support.template)))
                    .clamp(-f64::MAX, f64::MAX);
                }
                self.support.active = true;
                if crate::life::support_action::active(landscape.harmonic.support_weights) {
                    self.support.action =
                        Some((snapshot.clone(), landscape.harmonic.support_weights));
                }
            }
        }

        self.assert_scans();
        self.space
            .assert_scan_len_named(&landscape.consonance_field_level, "consonance_field_level");
        self.space
            .assert_scan_len_named(&landscape.harmonicity01, "harmonicity01");
        assert_eq!(
            (self.space.fmin, self.space.step()),
            (landscape.space.fmin, landscape.space.step()),
            "Flora space change was not applied"
        );
        // Before the first render there is no own sound in the mixed history.
        let fields = if !self.needs_hr_observer() || self.rendered_samples == 0 {
            Some((&landscape.consonance_field_level, &landscape.harmonicity01))
        } else if let Some(observation) = landscape
            .ecology
            .flora_observations
            .iter()
            .find(|s| s.id == self.id && s.generation == 0)
            && let Some(perception) = &observation.perception
        {
            Some((&observation.level_scan, &perception.h_scan))
        } else {
            // Runtime stops on invalidated analysis; direct callers retain the last fields.
            None
        };
        if let Some((growth, harmonicity)) = fields {
            self.space
                .assert_scan_len_named(growth, "consonance_field_level");
            self.space
                .assert_scan_len_named(harmonicity, "harmonicity01");
            self.growth_level_scan.copy_from_slice(growth);
            self.harmonicity_state_scan.copy_from_slice(harmonicity);
        }
        if self.range.is_empty() {
            return;
        }
        self.substeps = (((self.hop as f32 / self.fs) * self.pace)
            * (GROWTH_RATE / 0.05).max(self.diffusion / 0.25))
        .ceil()
        .max(1.0) as usize;
        let substeps = self.substeps;
        let dt = (self.hop as f32 / self.fs * self.pace) / substeps as f32;
        self.advance_ecology(dt, substeps);
    }

    pub(super) fn advance_ecology(&mut self, dt: f32, substeps: usize) {
        let bins_per_oct = self.space.bins_per_oct as f32;
        let allee_mass = ALLEE_DENSITY / bins_per_oct;
        for _ in 0..substeps {
            let maximum = self.biomass_mass_scan[self.range.clone()]
                .iter()
                .copied()
                .fold(0.0, f32::max);
            for i in self.range.clone() {
                let left = i.saturating_sub(1).max(self.range.start);
                let right = (i + 1).min(self.range.end - 1);
                let v = self.biomass_mass_scan[i];
                let u = self.resource_state_scan[i];
                let growth = GROWTH_RATE
                    * v
                    * (sanitize01(self.growth_level_scan[i]) * u * v / (v + allee_mass)
                        - METABOLISM);
                let spread = self.diffusion
                    * (self.biomass_mass_scan[left] - 2.0 * v + self.biomass_mass_scan[right]);
                let seeds = u
                    * (BACKGROUND_SEED_DENSITY
                        + SEED_RAIN_DENSITY * sanitize01(self.harmonicity_state_scan[i]))
                    / bins_per_oct;
                let next = v + dt * (growth + spread + seeds);
                debug_assert!(next >= -1e-6 * maximum, "negative Flora biomass: {next}");
                self.scratch_scan[i] = next.max(0.0);
            }
            std::mem::swap(&mut self.biomass_mass_scan, &mut self.scratch_scan);
            self.advance_resource(dt);
        }
    }

    pub(super) fn advance_resource(&mut self, dt: f32) {
        self.assert_scans();
        let bins_per_oct = self.space.bins_per_oct as f32;
        let renewal = 1.0 / RENEWAL_SEC;
        for (row, (lo, weights)) in self.consumption_rows.iter().enumerate() {
            let i = self.range.start + row;
            let biomass_consumption_density: f32 = self.consumption_weights[weights.clone()]
                .iter()
                .enumerate()
                .map(|(j, &weight)| weight * self.biomass_mass_scan[lo + j])
                .sum::<f32>()
                * bins_per_oct
                / REFERENCE_DENSITY;
            self.biomass_consumption_scan[i] = biomass_consumption_density;
            // Unknown initial consumption cannot be interpreted as silence.
            if self.external_observed_frame.is_none() {
                continue;
            }
            let a = renewal
                + (biomass_consumption_density + self.external_consumption_scan[i])
                    / CONSUMPTION_SEC;
            let equilibrium = renewal / a;
            self.resource_state_scan[i] =
                equilibrium + (self.resource_state_scan[i] - equilibrium) * (-a * dt).exp();
        }
    }

    pub(super) fn decompose(&mut self) {
        self.assert_scans();
        let p = &self.biomass_mass_scan;
        let maximum = p[self.range.clone()].iter().copied().fold(0.0, f32::max);
        for cell in &mut self.cells {
            cell.peak = None;
            cell.floor = p[cell.bins.clone()]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            cell.excess = cell.bins.clone().map(|i| p[i] - cell.floor).sum();
            cell.tonal_mass = 0.0;
            cell.residual_mass = cell.excess;
        }
        let mut first = self.range.start;
        while first < self.range.end {
            let mut end = first + 1;
            while end < self.range.end && p[end] == p[first] {
                end += 1;
            }
            let peak = (first == self.range.start || p[first - 1] < p[first])
                && (end == self.range.end || p[end] < p[first])
                && (first != self.range.start || end != self.range.end);
            if peak {
                let bin = first + (end - first - 1) / 2;
                let ci = self.cells.partition_point(|cell| cell.bins.end <= bin);
                let cell = &mut self.cells[ci];
                if cell.peak.as_ref().is_none_or(|old| p[bin] > p[old.bin]) {
                    let mut delta = 0.0;
                    let mut sigma = 1.0;
                    if end - first == 1 && bin > self.range.start && bin + 1 < self.range.end {
                        let l = p[bin - 1].max(1e-12).ln();
                        let c = p[bin].max(1e-12).ln();
                        let r = p[bin + 1].max(1e-12).ln();
                        let curvature = l - 2.0 * c + r;
                        if curvature < -1e-9 {
                            delta = ((l - r) / (2.0 * curvature)).clamp(-0.5, 0.5);
                            sigma = (-1.0 / curvature).sqrt().clamp(0.5, 4.0);
                        }
                    }
                    // Stop at the left bin of a minimum plateau, assigned to its left basin.
                    let mut left = first;
                    while left > self.range.start && p[left - 1] <= p[left] {
                        left -= 1;
                    }
                    if left > self.range.start {
                        left += 1;
                    }
                    let mut right = end - 1;
                    while right + 1 < self.range.end && p[right + 1] <= p[right] {
                        right += 1;
                    }
                    while right > end - 1 && p[right - 1] == p[right] {
                        right -= 1;
                    }
                    let window = ((bin as f32 - 2.0 * sigma).ceil() as usize)
                        .max(left)
                        .max(cell.bins.start)
                        ..(((bin as f32 + 2.0 * sigma).floor() as usize + 1)
                            .min(right + 1)
                            .min(cell.bins.end));
                    let position_bin = if end - first > 1 {
                        0.5 * (first + end - 1) as f32
                    } else {
                        bin as f32 + delta
                    };
                    cell.peak = Some(Peak {
                        bin,
                        position: self.space.centers_log2[0] + position_bin * self.space.step(),
                        sigma,
                        mass: window.clone().map(|i| p[i] - cell.floor).sum(),
                        window,
                        enter: p[bin] >= 0.02 * maximum,
                        stay: p[bin] >= 0.01 * maximum,
                        claimed: None,
                        suppressed: false,
                        visited: false,
                    });
                }
            }
            first = end;
        }
    }
}
