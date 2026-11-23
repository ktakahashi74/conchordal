use super::*;

impl ModalEngine {
    pub(crate) fn phase3_sample(&mut self, pitch: f32, drive: f32, open: bool) -> f32 {
        if open {
            if self.counter == 0 || (pitch - self.last_built_pitch_hz).abs() > 1e-6 {
                self.rebuild_modes(pitch.max(1.0));
                assert!(
                    self.phase3_residual_bound().is_finite(),
                    "phase3 Modal unsupported: free-response coefficient bound"
                );
            }
            self.counter = (self.counter + 1) % self.update_period_samples;
        }
        self.bank.process_sample(drive)
    }

    /// MCF invariant Q = r*x^2-r*e*x*y+y^2. Include binary32 operation
    /// errors and gradual-underflow residue in the future free-response bound.
    pub(crate) fn phase3_residual_bound(&self) -> f64 {
        let u = f64::from(f32::EPSILON) / 2.0;
        let gamma = 6.0 * u / (1.0 - 6.0 * u);
        let tiny = f64::from(f32::from_bits(1));
        let mut bound = 0.0;
        for i in 0..self.bank.active_len() {
            let Some(state) = self.bank.mode_state(i) else {
                return f64::INFINITY;
            };
            let [x, y, r, e, gain] = state.map(f64::from);
            let disc = ((r - 1.0).powi(2) + (r * e).powi(2)).sqrt();
            let hi = (r + 1.0 + disc) / 2.0;
            let lo = r * (1.0 - r * e * e / 4.0) / hi;
            if !(lo > 0.0 && r > 0.0 && r < 1.0) {
                return f64::INFINITY;
            }
            // A majorizes all exact intermediate magnitudes, including x1's
            // propagation into y1. Its Frobenius norm bounds its operator norm.
            let a = r;
            let b = r * e.abs();
            let c = r * r * e.abs();
            let d = r * (1.0 + r * e * e);
            let major = (a * a + b * b + c * c + d * d).sqrt();
            let contraction = r + gamma * major * (hi / lo).sqrt();
            if contraction >= 1.0 {
                return f64::INFINITY;
            }
            let q = r * x * x - r * e * x * y + y * y;
            let norm = q.max(0.0).sqrt() * (1.0 + 8.0 * f64::EPSILON * hi / lo);
            let residue = 6.0 * tiny * (1.0 + r * e.abs()) * hi.sqrt() / (1.0 - contraction);
            let output = (norm + residue) / (1.0 - r * e * e / 4.0).sqrt();
            bound += gain.abs() * output;
        }
        let nu = u * (self.bank.active_len() + 2) as f64;
        if nu >= 1.0 {
            return f64::INFINITY;
        }
        bound / (1.0 - nu) + self.bank.active_len() as f64 * tiny
    }
}
