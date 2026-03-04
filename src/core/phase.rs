use std::f32::consts::{PI, TAU};

#[inline]
pub fn wrap_0_tau(x: f32) -> f32 {
    x.rem_euclid(TAU)
}

/// Normalize to the range [-PI, PI).
#[inline]
pub fn wrap_pm_pi(x: f32) -> f32 {
    (x + PI).rem_euclid(TAU) - PI
}

#[inline]
pub fn angle_diff_pm_pi(a: f32, b: f32) -> f32 {
    wrap_pm_pi(a - b)
}

/// Sliding-window Phase Locking Value (PLV).
///
/// Accumulates phase differences in a circular buffer and computes
/// R = |mean(exp(i·θ))| over the window.
#[derive(Debug, Clone)]
pub struct SlidingPlv {
    buf_cos: Vec<f32>,
    buf_sin: Vec<f32>,
    sum_cos: f32,
    sum_sin: f32,
    idx: usize,
    len: usize,
}

impl SlidingPlv {
    pub fn new(window: usize) -> Self {
        Self {
            buf_cos: vec![0.0; window],
            buf_sin: vec![0.0; window],
            sum_cos: 0.0,
            sum_sin: 0.0,
            idx: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, angle: f32) {
        let window = self.buf_cos.len();
        if window == 0 {
            return;
        }
        if self.len >= window {
            self.sum_cos -= self.buf_cos[self.idx];
            self.sum_sin -= self.buf_sin[self.idx];
        }
        let c = angle.cos();
        let s = angle.sin();
        self.buf_cos[self.idx] = c;
        self.buf_sin[self.idx] = s;
        self.sum_cos += c;
        self.sum_sin += s;
        self.idx = (self.idx + 1) % window;
        if self.len < window {
            self.len += 1;
        }
    }

    pub fn plv(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let n = self.len as f32;
        let mean_cos = self.sum_cos / n;
        let mean_sin = self.sum_sin / n;
        (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
    }

    pub fn is_full(&self) -> bool {
        self.len >= self.buf_cos.len()
    }

    pub fn window(&self) -> usize {
        self.buf_cos.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_diff_is_wrapped() {
        let pairs = [
            (0.0, 0.0),
            (TAU, 0.0),
            (PI, -PI),
            (0.25 * PI, -0.25 * PI),
            (3.0 * PI, PI),
        ];
        for (a, b) in pairs {
            let d = angle_diff_pm_pi(a, b);
            assert!(d >= -PI && d < PI, "angle_diff out of range: {d}");
            let d2 = angle_diff_pm_pi(a + TAU, b);
            assert!((d - d2).abs() < 1e-5, "angle_diff periodicity failed");
        }
    }

    #[test]
    fn wrap_0_tau_in_range() {
        let values = [-10.0 * TAU, -TAU, -PI, -0.1, 0.0, PI, TAU, 3.5 * TAU];
        for v in values {
            let w = wrap_0_tau(v);
            assert!(w >= 0.0 && w < TAU, "wrap_0_tau out of range: {w}");
        }
    }

    #[test]
    fn wrap_pm_pi_in_range() {
        let values = [-10.0 * TAU, -TAU, -PI, -0.1, 0.0, PI, TAU, 3.5 * TAU];
        for v in values {
            let w = wrap_pm_pi(v);
            assert!(w >= -PI && w < PI, "wrap_pm_pi out of range: {w}");
        }
    }

    #[test]
    fn plv_full_agreement() {
        let mut plv = SlidingPlv::new(10);
        for _ in 0..10 {
            plv.push(0.5);
        }
        assert!(plv.is_full());
        assert!(
            (plv.plv() - 1.0).abs() < 1e-5,
            "PLV should be ~1.0 for identical angles"
        );
    }

    #[test]
    fn plv_uniform_spread() {
        let mut plv = SlidingPlv::new(100);
        for i in 0..100 {
            plv.push(TAU * i as f32 / 100.0);
        }
        assert!(
            plv.plv() < 0.1,
            "PLV should be ~0 for uniform spread, got {}",
            plv.plv()
        );
    }

    #[test]
    fn plv_window_sliding() {
        let mut plv = SlidingPlv::new(5);
        // Fill with scattered angles
        for i in 0..5 {
            plv.push(TAU * i as f32 / 5.0);
        }
        let scattered = plv.plv();
        // Now push 5 identical angles to replace the scattered ones
        for _ in 0..5 {
            plv.push(1.0);
        }
        let coherent = plv.plv();
        assert!(
            coherent > scattered + 0.5,
            "PLV should increase after coherent input"
        );
        assert!((coherent - 1.0).abs() < 1e-5);
    }

    #[test]
    fn plv_empty() {
        let plv = SlidingPlv::new(0);
        assert_eq!(plv.plv(), 0.0);
        assert_eq!(plv.window(), 0);

        let mut plv_zero = SlidingPlv::new(0);
        plv_zero.push(1.0); // should not panic
        assert_eq!(plv_zero.plv(), 0.0);
    }
}
