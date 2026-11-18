//! Landscape-level habituation: a per-bin state that erodes the consonance
//! evaluation terrain under sustained perceived-consonance activity and
//! recovers after release. See
//! docs/superpowers/specs/2026-07-17-habituation-field-design.md.

const LN_STALE: f32 = std::f32::consts::LN_10; // "stale" = 90% eroded

/// Research alternatives; channel-to-field mappings remain author rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HabituationDrive {
    #[default]
    RootProjection,
    ChannelQ,
    ChannelP,
}

/// Relax a signed score toward its level-neutral baseline `theta`.
#[inline]
pub fn erode_score(raw: f32, h: f32, theta: f32) -> f32 {
    let h = h.clamp(0.0, 1.0);
    theta + (raw - theta) * (1.0 - h)
}

/// Erode a non-negative mass multiplicatively.
#[inline]
pub fn erode_mass(raw: f32, h: f32) -> f32 {
    raw * (1.0 - h.clamp(0.0, 1.0))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HabituationParams {
    pub enabled: bool,
    pub drive: HabituationDrive,
    pub satiation_sec: f32,
    pub recovery_sec: f32,
    pub ref_drive: f32,
}

impl Default for HabituationParams {
    fn default() -> Self {
        Self {
            enabled: true,
            drive: HabituationDrive::default(),
            satiation_sec: 5.0,
            recovery_sec: 8.0,
            ref_drive: 0.25,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HabituationField {
    enabled: bool,
    drive: HabituationDrive,
    tau_e: f32,
    tau_r: f32,
    ref_drive: f32,
    theta: f32,
    h: Vec<f32>,
    channels: Vec<f32>,
}

impl HabituationField {
    pub fn new(params: &HabituationParams, theta: f32, n_bins: usize) -> Self {
        Self {
            enabled: params.enabled,
            drive: params.drive,
            tau_e: (params.satiation_sec.max(1e-3)) / LN_STALE,
            tau_r: (params.recovery_sec.max(1e-3)) / LN_STALE,
            ref_drive: params.ref_drive.max(1e-6),
            theta,
            h: vec![0.0; n_bins],
            channels: if params.drive == HabituationDrive::ChannelQ {
                vec![0.0; n_bins]
            } else {
                Vec::new()
            },
        }
    }

    pub fn ensure_len(&mut self, n_bins: usize) {
        if self.h.len() != n_bins {
            self.h.resize(n_bins, 0.0);
            if self.drive == HabituationDrive::ChannelQ {
                self.channels.resize(n_bins, 0.0);
            }
        }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn drive(&self) -> HabituationDrive {
        self.drive
    }

    /// Peak-normalized S6 density drives frequency channels without extra smoothing.
    /// Q averages adaptation at a candidate's partials using the H kernel's count
    /// and weights. Out-of-grid partials are unadapted, with their weights retained.
    pub fn advance_from_channels(
        &mut self,
        density: &[f32],
        space: &crate::core::log2space::Log2Space,
        harmonics: &crate::core::harmonicity_kernel::HarmonicityParams,
        dt: f32,
    ) {
        use crate::core::float::sanitize_nonnegative_finite;
        assert_ne!(self.drive, HabituationDrive::RootProjection);
        space.assert_scan_len(density);
        assert_eq!(density.len(), self.h.len());
        if !self.enabled || density.is_empty() {
            return;
        }
        let peak = density
            .iter()
            .copied()
            .map(sanitize_nonnegative_finite)
            .fold(0.0, f32::max);
        let a_e = (-dt.max(0.0) / self.tau_e).exp();
        let a_r = (-dt.max(0.0) / self.tau_r).exp();
        let channels = if self.drive == HabituationDrive::ChannelQ {
            &mut self.channels
        } else {
            &mut self.h
        };
        for (h, &s) in channels.iter_mut().zip(density) {
            let raw = if peak > 0.0 {
                sanitize_nonnegative_finite(s) / peak
            } else {
                0.0
            };
            let d = raw / (raw + self.ref_drive);
            let a = if d > *h { a_e } else { a_r };
            *h = (a * *h + (1.0 - a) * d).clamp(0.0, 1.0);
        }
        if self.drive == HabituationDrive::ChannelQ {
            self.h.fill(0.0);
            let mut total_weight = 0.0;
            for k in 1..=harmonics.num_harmonics.max(1) {
                let weight = (k as f32).powf(-harmonics.rho_common_root);
                let offset = (k as f32).log2() * space.bins_per_oct as f32;
                total_weight += weight;
                for (i, h) in self.h.iter_mut().enumerate() {
                    let pos = i as f32 + offset;
                    if pos <= (self.channels.len() - 1) as f32 {
                        let lo = pos.floor() as usize;
                        let hi = (lo + 1).min(self.channels.len() - 1);
                        let frac = pos - lo as f32;
                        *h +=
                            weight * (self.channels[lo] * (1.0 - frac) + self.channels[hi] * frac);
                    }
                }
            }
            for h in &mut self.h {
                *h = (*h / total_weight).clamp(0.0, 1.0);
            }
        }
    }

    #[inline]
    pub fn theta(&self) -> f32 {
        self.theta
    }

    #[inline]
    pub fn state(&self) -> &[f32] {
        &self.h
    }

    #[inline]
    fn transfer(&self, drive_raw: f32) -> f32 {
        let d = drive_raw.max(0.0);
        d / (d + self.ref_drive)
    }

    /// Advance `h` one step. `drive_raw[i] = level[i] * proj[i]` (perceived-
    /// consonance activity in root coordinate); asymmetric relaxation:
    /// rising uses tau_e (satiation), falling uses tau_r (recovery).
    ///
    /// F2: both inputs must be aligned to the field's bins. Call `ensure_len`
    /// first when the space can change.
    pub fn advance_from_parts(&mut self, level: &[f32], proj: &[f32], dt: f32) {
        assert_eq!(
            level.len(),
            self.h.len(),
            "habituation level scan length mismatch"
        );
        assert_eq!(
            proj.len(),
            self.h.len(),
            "habituation proj scan length mismatch"
        );
        if !self.enabled || self.h.is_empty() {
            return;
        }
        let dt = dt.max(0.0);
        let a_e = (-dt / self.tau_e).exp();
        let a_r = (-dt / self.tau_r).exp();
        for i in 0..self.h.len() {
            let d = self.transfer(level[i] * proj[i]).clamp(0.0, 1.0);
            let a = if d > self.h[i] { a_e } else { a_r };
            self.h[i] = (a * self.h[i] + (1.0 - a) * d).clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_drive_matches_relaxation_and_partial_overlap() {
        use crate::core::{harmonicity_kernel::HarmonicityParams, log2space::Log2Space};
        let space = Log2Space::new(100.0, 1600.0, 4);
        let mut density = vec![0.0; space.n_bins()];
        density[4] = 2.0;
        let hp = HarmonicityParams {
            num_harmonics: 3,
            ..Default::default()
        };
        let mut p = HabituationField::new(
            &HabituationParams {
                drive: HabituationDrive::ChannelP,
                ..Default::default()
            },
            0.0,
            density.len(),
        );
        let mut q = HabituationField::new(
            &HabituationParams {
                drive: HabituationDrive::ChannelQ,
                ..Default::default()
            },
            0.0,
            density.len(),
        );
        p.advance_from_channels(&density, &space, &hp, 5.0);
        q.advance_from_channels(&density, &space, &hp, 5.0);
        // At 5 seconds the driven channel has reached 90% of 1/(1+0.25).
        assert!((p.state()[4] - 0.72).abs() < 1e-6);
        assert_eq!(p.state()[0], 0.0);
        assert_eq!(q.channels, p.h);
        let w2 = 2.0_f32.powf(-0.4);
        let w3 = 3.0_f32.powf(-0.4);
        assert!((q.state()[0] - 0.72 * w2 / (1.0 + w2 + w3)).abs() < 1e-6);
        assert!((q.state()[4] - 0.72 / (1.0 + w2 + w3)).abs() < 1e-6);
        // The third partial of bin 0 lies between bins 6 and 7.
        density.fill(0.0);
        density[6] = 1.0;
        p.advance_from_channels(&density, &space, &hp, 8.0);
        q.advance_from_channels(&density, &space, &hp, 8.0);
        assert!((p.state()[4] - 0.072).abs() < 1e-6);
        let frac = 3.0_f32.log2() * 4.0 - 6.0;
        let expected = (w2 * p.state()[4] + w3 * (1.0 - frac) * p.state()[6]) / (1.0 + w2 + w3);
        assert!((q.state()[0] - expected).abs() < 1e-6);
        // Above-grid partials keep their denominator weight, as in HB-1.
        density.fill(1.0);
        q.advance_from_channels(&density, &space, &hp, 5.0);
        let last = density.len() - 1;
        assert!((q.state()[last] - q.channels[last] / (1.0 + w2 + w3)).abs() < 1e-6);
    }

    #[test]
    fn channel_drive_silence_disabled_and_scale_invariance() {
        use crate::core::{harmonicity_kernel::HarmonicityParams, log2space::Log2Space};
        let space = Log2Space::new(100.0, 800.0, 4);
        for drive in [HabituationDrive::ChannelP, HabituationDrive::ChannelQ] {
            let params = HabituationParams {
                drive,
                ..Default::default()
            };
            let mut a = HabituationField::new(&params, 0.0, space.n_bins());
            let mut b = a.clone();
            let mut off = HabituationField::new(
                &HabituationParams {
                    enabled: false,
                    ..params
                },
                0.0,
                space.n_bins(),
            );
            let mut density = vec![0.0; space.n_bins()];
            density[2] = 1.0;
            let hp = HarmonicityParams::default();
            for _ in 0..100 {
                a.advance_from_channels(&density, &space, &hp, 0.01);
                off.advance_from_channels(&density, &space, &hp, 0.01);
            }
            density[2] = 8.0;
            for _ in 0..100 {
                b.advance_from_channels(&density, &space, &hp, 0.01);
            }
            assert_eq!(a.h, b.h);
            assert!(off.h.iter().all(|&h| h == 0.0));
            let before = a.h.clone();
            density.fill(0.0);
            a.advance_from_channels(&density, &space, &hp, 8.0);
            for (&after, &before) in a.h.iter().zip(&before) {
                assert!((after - before * 0.1).abs() < 1e-6);
            }
        }
    }

    fn full_drive_params() -> HabituationParams {
        HabituationParams::default()
    }

    #[test]
    fn disabled_is_identity() {
        let f = HabituationField::new(
            &HabituationParams {
                enabled: false,
                ..Default::default()
            },
            0.0,
            4,
        );
        assert!(!f.is_enabled());
        assert_eq!(erode_score(0.7, 0.0, 0.0), 0.7);
        assert_eq!(erode_mass(0.7, 0.0), 0.7);
    }

    #[test]
    fn disabled_advance_is_noop() {
        let mut f = HabituationField::new(
            &HabituationParams {
                enabled: false,
                ..Default::default()
            },
            0.0,
            3,
        );
        assert!(!f.is_enabled());
        for _ in 0..100 {
            f.advance_from_parts(&[1.0, 1.0, 1.0], &[1e6, 1e6, 1e6], 0.05);
        }
        for &h in f.state() {
            assert_eq!(h, 0.0, "disabled field must not change state");
        }
    }

    #[test]
    fn erode_relaxes_score_toward_theta() {
        assert!((erode_score(0.9, 1.0, 0.2) - 0.2).abs() < 1e-6);
        assert!((erode_score(0.9, 0.5, 0.2) - 0.55).abs() < 1e-6);
        assert!(erode_score(-0.5, 0.5, 0.0) > -0.5);
    }

    #[test]
    fn erode_mass_is_multiplicative() {
        assert!((erode_mass(0.8, 0.5) - 0.4).abs() < 1e-6);
        assert_eq!(erode_mass(0.8, 1.0), 0.0);
        assert_eq!(erode_mass(0.8, 0.0), 0.8);
    }

    #[test]
    fn satiation_reaches_0_9_in_satiation_sec_under_full_drive() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 1);
        let level = [1.0f32];
        let proj = [1000.0f32]; // transfer(1000) ~= 1.0
        let dt = 0.01;
        let mut t = 0.0;
        while f.state()[0] < 0.9 {
            f.advance_from_parts(&level, &proj, dt);
            t += dt;
        }
        assert!((t - 5.0).abs() < 0.15, "reached 0.9 at t={t}");
    }

    #[test]
    fn recovery_falls_to_0_1_in_recovery_sec_at_zero_drive() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 1);
        for _ in 0..2000 {
            f.advance_from_parts(&[1.0], &[1000.0], 0.01);
        }
        let start = f.state()[0];
        assert!(start > 0.99);
        let dt = 0.01;
        let mut t = 0.0;
        while f.state()[0] > 0.1 {
            f.advance_from_parts(&[0.0], &[0.0], dt);
            t += dt;
        }
        assert!((t - 8.0).abs() < 0.2, "fell to 0.1 at t={t}");
    }

    #[test]
    #[should_panic(expected = "habituation level scan length mismatch")]
    fn advance_panics_on_level_len_mismatch() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 3);
        f.advance_from_parts(&[1.0, 1.0], &[1.0, 1.0, 1.0], 0.05);
    }

    #[test]
    #[should_panic(expected = "habituation proj scan length mismatch")]
    fn advance_panics_on_proj_len_mismatch() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 3);
        f.advance_from_parts(&[1.0, 1.0, 1.0], &[1.0, 1.0], 0.05);
    }

    #[test]
    fn state_stays_bounded() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 3);
        for _ in 0..10_000 {
            f.advance_from_parts(&[1.0, 0.0, 0.5], &[1e6, 1e6, 1e6], 0.05);
        }
        for &h in f.state() {
            assert!((0.0..=1.0).contains(&h), "h={h}");
        }
    }
}
