//! Landscape-level habituation: a per-bin state that erodes the consonance
//! evaluation terrain under sustained perceived-consonance activity and
//! recovers after release. See
//! docs/superpowers/specs/2026-07-17-habituation-field-design.md.

const LN_STALE: f32 = std::f32::consts::LN_10; // "stale" = 90% eroded

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
    pub satiation_sec: f32,
    pub recovery_sec: f32,
    pub ref_drive: f32,
}

impl Default for HabituationParams {
    fn default() -> Self {
        Self {
            enabled: false,
            satiation_sec: 5.0,
            recovery_sec: 8.0,
            ref_drive: 0.25,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HabituationField {
    enabled: bool,
    tau_e: f32,
    tau_r: f32,
    ref_drive: f32,
    theta: f32,
    h: Vec<f32>,
}

impl HabituationField {
    pub fn new(params: &HabituationParams, theta: f32, n_bins: usize) -> Self {
        Self {
            enabled: params.enabled,
            tau_e: (params.satiation_sec.max(1e-3)) / LN_STALE,
            tau_r: (params.recovery_sec.max(1e-3)) / LN_STALE,
            ref_drive: params.ref_drive.max(1e-6),
            theta,
            h: vec![0.0; n_bins],
        }
    }

    pub fn ensure_len(&mut self, n_bins: usize) {
        if self.h.len() != n_bins {
            self.h.resize(n_bins, 0.0);
        }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
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
    pub fn advance_from_parts(&mut self, level: &[f32], proj: &[f32], dt: f32) {
        if !self.enabled || self.h.is_empty() {
            return;
        }
        let dt = dt.max(0.0);
        let a_e = (-dt / self.tau_e).exp();
        let a_r = (-dt / self.tau_r).exp();
        for i in 0..self.h.len() {
            let lvl = level.get(i).copied().unwrap_or(0.0);
            let pj = proj.get(i).copied().unwrap_or(0.0);
            let d = self.transfer(lvl * pj).clamp(0.0, 1.0);
            let a = if d > self.h[i] { a_e } else { a_r };
            self.h[i] = (a * self.h[i] + (1.0 - a) * d).clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_drive_params() -> HabituationParams {
        HabituationParams {
            enabled: true,
            satiation_sec: 5.0,
            recovery_sec: 8.0,
            ref_drive: 0.25,
        }
    }

    #[test]
    fn disabled_is_identity() {
        let f = HabituationField::new(&HabituationParams::default(), 0.0, 4);
        assert!(!f.is_enabled());
        assert_eq!(erode_score(0.7, 0.0, 0.0), 0.7);
        assert_eq!(erode_mass(0.7, 0.0), 0.7);
    }

    #[test]
    fn disabled_advance_is_noop() {
        let mut f = HabituationField::new(&HabituationParams::default(), 0.0, 3);
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
