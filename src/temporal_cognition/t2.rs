//! Named AMT16 component composition, before author-defined event mapping.
use std::f64::consts::PI;
use std::time::Instant;

pub(crate) const CHANNELS: usize = 30;
pub(crate) const MODEL: &str = "amt16-t2-limit10-lp2.5-bp5-bp10-bp50over3-v1";

// LTFAT ERB units differ from the instrument's rounded f32 ERB helper.
pub(super) fn centers_hz() -> [f64; CHANNELS] {
    let low = 9.2645 * (1.0_f64 + 80.0 * 0.00437).ln();
    let high = 9.2645 * (1.0_f64 + 8000.0 * 0.00437).ln();
    let anchor = 9.2645 * (1.0_f64 + 1000.0 * 0.00437).ln();
    let below = (anchor - low).floor() as i32;
    let above = (high - anchor).floor() as i32;
    assert_eq!(below + above + 1, CHANNELS as i32);
    std::array::from_fn(|j| {
        (1.0 / 0.00437) * ((((j as i32 - below) as f64 + anchor) / 9.2645).exp() - 1.0)
    })
}

pub(crate) fn validate_config(c: crate::config::TemporalT2Config) -> Result<(), &'static str> {
    if c.component_weights
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
        || !c.component_weights.iter().any(|x| *x > 0.0)
        || !c.gain.is_finite()
        || c.gain <= 0.0
        || !c.threshold.is_finite()
        || c.threshold < 0.0
        || !c.weight_gain.is_finite()
        || c.weight_gain <= 0.0
    {
        return Err("invalid explicit T2 event weights, gains or threshold");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Frame {
    pub epoch_start: u64,
    pub start: u64,
    pub end: u64,
    pub means: [[f64; 4]; CHANNELS],
    pub processing_us: u64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub model: &'static str,
    pub sample_rate: u32,
    pub epoch_start: u64,
    pub start: u64,
    pub end: u64,
    pub processing_us: u64,
}

pub(crate) struct Producer {
    channels: [Channel; CHANNELS],
    epoch_start: u64,
    next_sample: u64,
}

impl Producer {
    pub(crate) fn new(sample_rate: u32) -> Result<Self, &'static str> {
        let centers = centers_hz();
        if f64::from(sample_rate) / 2.0 <= centers[CHANNELS - 1] {
            return Err("T2 requires every native center below Nyquist");
        }
        Ok(Self {
            channels: centers.map(|center| Channel::new(f64::from(sample_rate), center)),
            epoch_start: 0,
            next_sample: 0,
        })
    }

    pub(crate) fn reset(&mut self, start: u64) {
        for channel in &mut self.channels {
            channel.gt = [Complex::default(); 4];
            channel.ihc = [0.0; 2];
            channel.lp = [0.0; 4];
            channel.z = [Complex::default(); 3];
            let mut value: f64 = 1e-5;
            for q in &mut channel.q {
                value = value.sqrt();
                *q = value;
            }
        }
        self.epoch_start = start;
        self.next_sample = start;
    }

    pub(crate) fn process(&mut self, start: u64, pcm: &[f32]) -> Result<Frame, &'static str> {
        if start != self.next_sample || pcm.is_empty() || pcm.iter().any(|x| !x.is_finite()) {
            return Err("invalid or discontinuous T2 PCM input");
        }
        let end = start
            .checked_add(pcm.len() as u64)
            .ok_or("T2 sample clock overflow")?;
        let began = Instant::now();
        let mut means = [[0.0; 4]; CHANNELS];
        for (channel, mean) in self.channels.iter_mut().zip(&mut means) {
            for &x in pcm {
                let output = channel.one(f64::from(x)).0;
                for (sum, value) in mean.iter_mut().zip(output) {
                    *sum += value;
                }
            }
            for value in mean {
                *value /= pcm.len() as f64;
            }
        }
        if means.iter().flatten().any(|x| !x.is_finite()) {
            return Err("nonfinite T2 output");
        }
        self.next_sample = end;
        Ok(Frame {
            epoch_start: self.epoch_start,
            start,
            end,
            means,
            processing_us: began.elapsed().as_micros().min(u64::MAX as u128) as u64,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

#[derive(Clone)]
struct Channel {
    gt_pole: Complex,
    gt_gain: f64,
    ihc_b: f64,
    ihc_a: f64,
    adapt_a: [f64; 5],
    adapt_b: [f64; 5],
    factor: [f64; 5],
    expfac: [f64; 5],
    offset: [f64; 5],
    corr: f64,
    mult: f64,
    lp_b: [f64; 3],
    lp_a: [f64; 2],
    mod_pole: [Complex; 3],
    mod_gain: [f64; 3],
    gt: [Complex; 4],
    ihc: [f64; 2],
    q: [f64; 5],
    lp: [f64; 4],
    z: [Complex; 3],
}

impl Channel {
    fn new(fs: f64, center: f64) -> Self {
        assert!(fs.is_finite() && fs > 2000.0);
        assert!(center.is_finite() && center > 0.0 && center < fs / 2.0);
        let beta = (36.0 / (PI * 720.0 * 2.0_f64.powi(-6))) * (24.7 + center / 9.265);
        let radius = (-2.0 * PI * beta / fs).exp();
        let angle = -2.0 * PI * center / fs;
        let gt_pole = Complex {
            re: radius * angle.cos(),
            im: radius * angle.sin(),
        };
        let ihc_k = (PI * 1000.0 / fs).tan();
        let tau = [0.005, 0.050, 0.129, 0.253, 0.500];
        let mut q = [0.0; 5];
        let mut root: f64 = 1e-5;
        for value in &mut q {
            root = root.sqrt();
            *value = root;
        }
        let adapt_a = tau.map(|t| (-1.0 / (t * fs)).exp());
        let adapt_b = adapt_a.map(|a| 1.0 - a);
        let amplitude = q.map(|q| (1.0 - q * q) * 10.0 - 1.0);
        let k = (PI * 2.5 / fs).tan();
        let d = 1.0 + 2.0_f64.sqrt() * k + k * k;
        let frequency = [5.0, 10.0, 50.0 / 3.0];
        let width = [5.0, 5.0, (50.0 / 3.0) / 2.0];
        let radius_mod = width.map(|bw| (-PI * bw / fs).exp());
        let mod_pole = std::array::from_fn(|j| {
            let angle = -2.0 * PI * frequency[j] / fs;
            Complex {
                re: radius_mod[j] * angle.cos(),
                im: radius_mod[j] * angle.sin(),
            }
        });
        Self {
            gt_pole,
            gt_gain: 1.0 - radius,
            ihc_b: ihc_k / (1.0 + ihc_k),
            ihc_a: (ihc_k - 1.0) / (ihc_k + 1.0),
            adapt_a,
            adapt_b,
            factor: amplitude.map(|a| 2.0 * a),
            expfac: amplitude.map(|a| -2.0 / a),
            offset: amplitude.map(|a| a - 1.0),
            corr: q[4],
            mult: 100.0 / (1.0 - q[4]),
            lp_b: [k * k / d, 2.0 * k * k / d, k * k / d],
            lp_a: [
                2.0 * (k * k - 1.0) / d,
                (1.0 - 2.0_f64.sqrt() * k + k * k) / d,
            ],
            mod_pole,
            mod_gain: radius_mod.map(|r| 1.0 - r),
            gt: [Complex::default(); 4],
            ihc: [0.0; 2],
            q,
            lp: [0.0; 4],
            z: [Complex::default(); 3],
        }
    }

    fn one(&mut self, x: f64) -> ([f64; 4], f64, f64) {
        let mut v = Complex { re: x, im: 0.0 };
        for state in &mut self.gt {
            v = Complex {
                re: (self.gt_pole.re * state.re - self.gt_pole.im * state.im) + self.gt_gain * v.re,
                im: (self.gt_pole.re * state.im + self.gt_pole.im * state.re) + self.gt_gain * v.im,
            };
            *state = v;
        }
        let rect = (2.0 * v.re).max(0.0);
        let ihc = self.ihc_b * (rect + self.ihc[0]) - self.ihc_a * self.ihc[1];
        self.ihc = [rect, ihc];
        let mut value = ihc.max(1e-5);
        for j in 0..5 {
            value /= self.q[j];
            if value > 1.0 {
                value = self.factor[j] / (1.0 + (self.expfac[j] * (value - 1.0)).exp())
                    - self.offset[j];
            }
            self.q[j] = self.adapt_a[j] * self.q[j] + self.adapt_b[j] * value;
        }
        let mu = (value - self.corr) * self.mult;
        let [x1, x2, y1, y2] = self.lp;
        let lp = self.lp_b[0] * mu + self.lp_b[1] * x1 + self.lp_b[2] * x2
            - self.lp_a[0] * y1
            - self.lp_a[1] * y2;
        self.lp = [mu, x1, lp, y1];
        for j in 0..3 {
            let z = self.z[j];
            self.z[j] = Complex {
                re: (self.mod_pole[j].re * z.re - self.mod_pole[j].im * z.im)
                    + self.mod_gain[j] * mu,
                im: self.mod_pole[j].re * z.im + self.mod_pole[j].im * z.re,
            };
        }
        (
            [
                lp,
                2.0 * self.z[0].re,
                2.0 * self.z[1].re,
                2.0 * self.z[2].re.hypot(self.z[2].im),
            ],
            ihc,
            mu,
        )
    }
}

pub(super) mod adapter;
#[cfg(test)]
mod tests;
