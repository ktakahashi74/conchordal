//! IPEM sensory-memory candidate; probe decoding and generation use are author rules.
// IPEM-derived equations: Copyright (C) 2005 Ghent University, GPL-2.0-or-later.
#[cfg(test)]
pub(crate) mod acquisition;
mod ani;
mod decoder;
pub(crate) mod pitch_set;
mod runtime;
#[cfg(test)]
mod tests;
use ani::Ani;
pub(crate) use decoder::Decoder;
pub(crate) use runtime::{Context, Record, Snapshot, Use};
use std::sync::Arc;
const W: usize = 176;
const STEP: usize = 28;
const FA: f64 = 2756.25;
#[derive(Clone)]
struct Resampler<const C: usize> {
    p: usize,
    q: usize,
    h: Arc<[f64]>,
    ring: Vec<[f64; C]>,
    count: usize,
    next: usize,
}
impl<const C: usize> Resampler<C> {
    fn new(p: usize, q: usize) -> Self {
        let half = 10 * p.max(q);
        let mut h = Vec::with_capacity(2 * half + 1);
        let i0 = |x: f64| {
            let mut total = 1.;
            let mut term = 1.;
            for k in 1..50 {
                term *= x * x / (4. * (k * k) as f64);
                total += term;
                if term < total * 1e-17 {
                    break;
                }
            }
            total
        };
        for i in 0..=2 * half {
            let u = (i as f64 - half as f64) / p.max(q) as f64;
            let sinc = if u == 0. {
                1.
            } else {
                (std::f64::consts::PI * u).sin() / (std::f64::consts::PI * u)
            };
            let r = (i as f64 - half as f64) / half as f64;
            h.push(sinc * i0(5. * (1. - r * r).max(0.).sqrt()) / i0(5.));
        }
        let scale = p as f64 / h.iter().sum::<f64>();
        for v in &mut h {
            *v *= scale;
        }
        Self {
            p,
            q,
            h: h.into(),
            ring: vec![[0.; C]; 2 * half / p + 3],
            count: 0,
            next: 0,
        }
    }
    fn push(&mut self, x: [f64; C]) -> Option<[f64; C]> {
        let size = self.ring.len();
        self.ring[self.count % size] = x;
        self.count += 1;
        let center = self.next * self.q;
        let half = self.h.len() / 2;
        let right = (center + half) / self.p;
        if right >= self.count {
            return None;
        }
        let left = (center.saturating_sub(half)).div_ceil(self.p);
        let mut y = [0.; C];
        for n in left..=right {
            let w = self.h[center + half - n * self.p];
            let x = &self.ring[n % size];
            for c in 0..C {
                y[c] += w * x[c];
            }
        }
        self.next += 1;
        Some(y)
    }
}
#[derive(Clone)]
struct Pitch {
    b: [f64; 3],
    a: [f64; 3],
    z: [[f64; 40]; 2],
    ani: [[f64; 40]; 6],
    count: usize,
    fani: Vec<[f64; 40]>,
    fcount: usize,
    frame: usize,
    local: [f64; W],
    glob: [f64; W],
}
#[derive(Clone)]
struct Frame {
    index: usize,
    pp: [f64; W],
    #[cfg(test)]
    local: [f64; W],
    #[cfg(test)]
    glob: [f64; W],
    #[cfg(test)]
    tc: Option<f64>,
}
fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let (mut vx, mut vy, mut cross) = (0., 0., 0.);
    for (&x, &y) in x.iter().zip(y) {
        let a = x - mx;
        let b = y - my;
        vx += a * a;
        vy += b * b;
        cross += a * b;
    }
    if vx <= 0. || vy <= 0. {
        return None;
    }
    let r = cross / (vx * vy).sqrt();
    r.is_finite().then(|| r.clamp(-1., 1.))
}
impl Pitch {
    fn new() -> Self {
        let k = (std::f64::consts::PI * 80. / FA).tan();
        let norm = 1. / (1. + 2_f64.sqrt() * k + k * k);
        Self {
            b: [k * k * norm, 2. * k * k * norm, k * k * norm],
            a: [
                1.,
                2. * (k * k - 1.) * norm,
                (1. - 2_f64.sqrt() * k + k * k) * norm,
            ],
            z: [[0.; 40]; 2],
            ani: [[0.; 40]; 6],
            count: 0,
            fani: vec![[0.; 40]; 2 * W],
            fcount: 0,
            frame: 0,
            local: [0.; W],
            glob: [0.; W],
        }
    }
    fn push(&mut self, x: [f64; 40]) -> Option<Frame> {
        // The second-order 80 Hz impulse response peaks at index 6.
        let old = self.ani[self.count % 6];
        self.ani[self.count % 6] = x;
        let mut f = [0.; 40];
        for c in 0..40 {
            let low = self.b[0] * x[c] + self.z[0][c];
            self.z[0][c] = self.b[1] * x[c] - self.a[1] * low + self.z[1][c];
            self.z[1][c] = self.b[2] * x[c] - self.a[2] * low;
            f[c] = (old[c] - low).max(0.);
        }
        self.count += 1;
        if self.count <= 6 {
            return None;
        }
        self.fani[self.fcount % (2 * W)] = f;
        self.fcount += 1;
        if self.fcount < 2 * W || !(self.fcount - 2 * W).is_multiple_of(STEP) {
            return None;
        }
        let start = self.fcount - 2 * W;
        let mut pp = [0.; W];
        for c in 0..40 {
            // Contiguous lag lanes preserve each sum's original n/c order.
            let mut channel = [0.; 2 * W];
            for (n, value) in channel.iter_mut().enumerate() {
                *value = self.fani[(start + n) % (2 * W)][c];
            }
            let mut sums = [0.; W];
            for n in 0..W {
                for lag in 0..W {
                    sums[lag] += channel[n] * channel[n + lag];
                }
            }
            for (total, sum) in pp.iter_mut().zip(sums) {
                *total += sum;
            }
        }
        self.integrate(pp)
    }
    fn integrate(&mut self, pp: [f64; W]) -> Option<Frame> {
        let al = 2_f64.powf(-1. / (FA / STEP as f64 * 0.1));
        let ag = 2_f64.powf(-1. / (FA / STEP as f64 * 1.5));
        for (i, &p) in pp.iter().enumerate() {
            self.local[i] = al * self.local[i] + p;
            self.glob[i] = ag * self.glob[i] + p;
        }
        let frame = Frame {
            index: self.frame,
            pp,
            #[cfg(test)]
            local: self.local,
            #[cfg(test)]
            glob: self.glob,
            #[cfg(test)]
            tc: pearson(&self.local, &self.glob),
        };
        self.frame += 1;
        Some(frame)
    }
}
#[derive(Clone)]
struct Model {
    input: Resampler<1>,
    ani: Ani,
    down: Resampler<40>,
    pitch: Pitch,
    n0: usize,
    #[cfg(test)]
    sample_rate: usize,
    valid: bool,
    raw_count: usize,
}
impl Model {
    fn new(fs: usize) -> Self {
        let (p, q) = match fs {
            48000 => (147, 320),
            44100 => (1, 2),
            _ => panic!("unsupported research sample rate"),
        };
        let mut model = Self {
            input: Resampler::new(p, q),
            ani: Ani::new(),
            down: Resampler::new(1, 4),
            pitch: Pitch::new(),
            n0: 0,
            #[cfg(test)]
            sample_rate: fs,
            valid: true,
            raw_count: 0,
        };
        for _ in 0..441 {
            model.ani.tick(0.);
        }
        model.raw_count = 221;
        model
    }
    fn raw(&mut self, x: f64) -> (Option<[f64; 40]>, Option<Frame>) {
        let raw = self.ani.tick(x);
        let mut frame = None;
        if let Some(mut a) = raw {
            self.raw_count += 1;
            // Reproduce the C ASCII boundary as a separately documented choice.
            for v in &mut a {
                *v = (*v * 1e10).round() / 1e10;
            }
            if let Some(ani) = self.down.push(a) {
                frame = self.pitch.push(ani);
            }
        }
        (raw, frame)
    }
    fn push(&mut self, x: f64) -> Option<Frame> {
        self.n0 += 1;
        self.valid &= x.is_finite();
        if !self.valid {
            return None;
        }
        if let Some([x]) = self.input.push([x]) {
            self.raw((x * 32768.).round().clamp(-32768., 32767.) / 32768.)
                .1
        } else {
            None
        }
    }
    #[cfg(test)]
    fn query(&self, hz: f64) -> Option<(f64, usize, usize)> {
        if !self.valid
            || self.sample_rate != 48000
            || !hz.is_finite()
            || !(55. * 2_f64.powf(-7. / 8.)..=8000.).contains(&hz)
        {
            return None;
        }
        let mut clone = self.clone();
        let mut sum = 0.;
        let mut count = 0;
        let mut j: usize = 0;
        let first = ((self.n0 + 9600) * 147).div_ceil(STEP * 2560);
        let end = ((self.n0 + 28800) * 147).div_ceil(STEP * 2560);
        loop {
            let x = if j < 28800 {
                let u = j as f64 / 48000.;
                let e = if u < 0.01 {
                    (1. - (std::f64::consts::PI * u / 0.01).cos()) / 2.
                } else if u >= 0.59 {
                    (1. - (std::f64::consts::PI * (0.6 - u) / 0.01).cos()) / 2.
                } else {
                    1.
                };
                0.1 * e * (2. * std::f64::consts::PI * hz * u).sin()
            } else {
                0.
            };
            if let Some(f) = clone.push(x) {
                if f.index >= first && f.index < end {
                    sum += f.tc?;
                    count += 1;
                }
                if f.index + 1 >= end {
                    break;
                }
            }
            j += 1;
        }
        if count != end - first || count == 0 {
            None
        } else {
            Some((sum / count as f64, count, (j + 1).saturating_sub(28800)))
        }
    }
}
