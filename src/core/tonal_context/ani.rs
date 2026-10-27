// Research translation of IPEMToolbox 6907d7373d134b6d4856033473177606a11ad61f.
// Copyright (C) 2005 Ghent University. GPL-2.0-or-later for this derived module.
// The historical constants and update order are intentional.
#![allow(clippy::approx_constant)]
const PI: f64 = 3.14159;
const FS: f64 = 44.1;
const FSS: f64 = 22.05;
#[derive(Clone, Copy, Default)]
struct Cell {
    a1: f64,
    a2: f64,
    b1: f64,
    b2: f64,
    w1: f64,
    w2: f64,
}
impl Cell {
    fn tick(&mut self, y: f64) -> f64 {
        let x = y - self.b1 * self.w1 - self.b2 * self.w2;
        let y = x + self.a1 * self.w1 + self.a2 * self.w2;
        self.w2 = self.w1;
        self.w1 = x;
        y
    }
}
#[derive(Clone, Copy, Default)]
struct Channel {
    step: usize,
    index: usize,
    gain: f64,
    cells: [Cell; 2],
    c1: f64,
    c2: f64,
    g1q: f64,
    a1q: f64,
    a2q: f64,
    zn: f64,
    w1n: f64,
    w2n: f64,
    qn: f64,
    qfac: f64,
    fn_: f64,
    b: f64,
    b1: f64,
    b2: f64,
    g1: f64,
    g2: f64,
    y1n: f64,
    wn1: f64,
    wn2: f64,
}
#[derive(Clone)]
pub(super) struct Ani {
    channels: [Channel; 40],
    n: usize,
    zhp: f64,
    gain: f64,
    b1: f64,
    b2: f64,
    xhp: f64,
    yhp: f64,
    yn1: f64,
    yn2: f64,
    df: [Cell; 3],
    h: [f64; 17],
    d: [[f64; 112]; 3],
    ptr: [usize; 3],
    dec: [f64; 4],
    bias: f64,
    factor2: f64,
}
fn inverse(u: f64) -> f64 {
    let cb = (0.2 * 1.5 / 0.07_f64 - 1.).sqrt() / 1.5;
    let ca = 1. / 0.07 / cb;
    let cc = 5.;
    let u0 = ca * (cb * 1.5).atan();
    let cd = u0 - cc * 1.5_f64.ln();
    if u <= u0 {
        (u / ca).sin() / (u / ca).cos() / cb
    } else {
        ((u - cd) / cc).exp()
    }
}
impl Ani {
    pub(super) fn new() -> Self {
        let theta = 2. * PI * 4. / FSS;
        let mut r = 1. - 1. / 2.25;
        let mut b2 = r * r;
        let mut b1 = -2. * r * theta.cos();
        loop {
            let rz = theta.cos();
            let iz = theta.sin();
            let rz2 = (2. * theta).cos();
            let iz2 = (2. * theta).sin();
            let resp = 1. / ((rz2 + b1 * rz + b2).powi(2) + (iz2 + b1 * iz).powi(2));
            let alpha = resp.sqrt() * (1. + b1 + b2) / 2.25;
            r = 1. - alpha.sqrt() * (1. - r);
            b2 = r * r;
            b1 = -2. * r * theta.cos();
            if (alpha - 1.).abs() < 0.05 {
                break;
            }
        }
        let mut this = Self {
            channels: [Channel::default(); 40],
            n: 0,
            zhp: 1. - 2. * PI * 0.25 / FSS,
            gain: 1. + b1 + b2,
            b1,
            b2,
            xhp: 0.,
            yhp: 0.,
            yn1: 0.,
            yn2: 0.,
            df: [Cell::default(); 3],
            h: [0.; 17],
            d: [[0.; 112]; 3],
            ptr: [0; 3],
            dec: [0.; 4],
            bias: (0.001_f64 * 0.150 / 0.05).sqrt() - 0.001_f64.sqrt(),
            factor2: 0.,
        };
        this.factor2 = 0.150 / this.bias.powi(2);
        for (c, values) in this.df.iter_mut().zip([
            [0.35876, -0.20127, 0.86735],
            [0.79104, -0.36732, 0.54560],
            [1.74419, -0.61855, 0.18040],
        ]) {
            c.a1 = values[0];
            c.a2 = 1.;
            c.b1 = values[1];
            c.b2 = values[2];
        }
        this.h[..9].copy_from_slice(&[
            0.0030106236,
            0.016853167,
            0.021166865,
            -0.0095273824,
            -0.057905637,
            -0.042390779,
            0.096695431,
            0.29275990,
            0.38665709,
        ]);
        for i in 1..=8 {
            this.h[8 + i] = this.h[8 - i];
        }
        for p in 0..40 {
            let uc = 2. + p as f64 * 0.5;
            let fc = inverse(uc);
            let mut fs = FS;
            let mut step = 1;
            let mut index = 0;
            while fc / fs <= 0.125 && step < 4 {
                step *= 2;
                index += 1;
                fs *= 0.5;
            }
            let mut c = Channel {
                step,
                index,
                ..Channel::default()
            };
            let f1 = inverse(uc - 0.64);
            let f2 = inverse(uc + 0.64);
            let mut fh = inverse(uc + 6.);
            let wc = PI * (f2 + f1) / fs;
            let hdw = PI * (f2 - f1) / fs;
            let k = hdw.cos() / hdw.sin();
            let cos_fie = wc.cos() / hdw.cos();
            let angle = PI / 4.;
            let re = -angle.sin();
            let im = angle.cos();
            let rx = k * cos_fie;
            let ry = (re - k).powi(2) + im * im;
            let e = re - k;
            let aa = (rx - k) * (rx + k) + (re - im) * (re + im);
            let bb = 2. * re * im;
            let mag = (aa * aa + bb * bb).powf(0.25);
            let mut arg = (bb / aa).atan();
            if aa < 0. {
                arg += PI;
            }
            arg /= 2.;
            let rs = mag * arg.cos();
            let is = mag * arg.sin();
            let v = rs - rx;
            let z = rs + rx;
            c.cells[0].b1 = -2. * (e * v - im * is) / ry;
            c.cells[0].b2 = (v * v + is * is) / ry;
            c.cells[1].b1 = 2. * (e * z - im * is) / ry;
            c.cells[1].b2 = (z * z + is * is) / ry;
            if fh > 0.5 * fs {
                let r = (-1.5 * (fh - 0.5 * fs) / fs).exp();
                c.cells[0].a1 = 2. * r;
                c.cells[0].a2 = r * r;
            } else {
                fh = fh.max(0.25 * fs);
                c.cells[0].a1 = -2. * (2. * PI * fh / fs).cos();
                c.cells[0].a2 = 1.;
            }
            let scale = if step == 4 { 1. } else { 2. };
            let r = 1. - (-0.25 * uc * scale).exp();
            c.cells[1].a1 = -2. * r;
            c.cells[1].a2 = r * r;
            let ang = 2. * PI * fc / fs;
            let co = ang.cos();
            let co2 = (2. * ang).cos();
            let si = ang.sin();
            let si2 = 2. * si * co;
            let mut resp = 1.;
            for cell in &c.cells {
                resp *= ((1. + cell.a1 * co + cell.a2 * co2).powi(2)
                    + (cell.a1 * si + cell.a2 * si2).powi(2))
                    / ((1. + cell.b1 * co + cell.b2 * co2).powi(2)
                        + (cell.b1 * si + cell.b2 * si2).powi(2));
            }
            c.gain = 1. / resp.sqrt();
            c.c1 = (2. * fs * 8. - 1.) / (2. * fs * 8. + 1.);
            c.g1q = 0.5 * (1. - c.c1);
            c.c2 = (2. * fs * 40. - 1.) / (2. * fs * 40. + 1.);
            let tau = 0.860 * 8. + (1. - 0.860) * 40.;
            c.a1q = (1. - c.c2) * (0.5 + fs * 8. * 40. / tau);
            c.a2q = (2. * fs * 8. * 40. - tau) * c.a1q / (2. * fs * 8. * 40. + tau);
            let kb = PI * 1.250 / fs;
            let kb = kb.cos() / kb.sin();
            c.b = (kb - 1.) / (kb + 1.);
            c.g1 = 0.5 * (1. - c.b);
            c.b2 = (1. + kb * kb - kb) / (1. + kb * kb + kb);
            c.b1 = 2. * (1. - kb * kb) / (1. + kb * kb + kb);
            c.g2 = 0.25 * (1. + c.b1 + c.b2);
            c.zn = 0.001;
            c.qn = 0.001;
            c.w1n = 0.001;
            c.w2n = 0.001 / (c.a1q - c.a2q);
            c.fn_ = 0.05;
            c.y1n = 0.05;
            c.wn1 = 0.25 * 0.05;
            c.wn2 = c.wn1;
            this.channels[p] = c;
        }
        this
    }
    pub(super) fn tick(&mut self, x: f64) -> Option<[f64; 40]> {
        self.yhp = self.zhp * self.yhp + x - self.xhp;
        self.xhp = x;
        let y = self.gain * self.yhp - self.b1 * self.yn1 - self.b2 * self.yn2;
        self.yn2 = self.yn1;
        self.yn1 = y;
        let out = self.internal(y);
        self.internal(0.);
        out
    }
    fn internal(&mut self, mut x: f64) -> Option<[f64; 40]> {
        let mut y = 0.04435 * x;
        for c in &mut self.df {
            y = c.tick(y);
        }
        self.ptr[0] = (self.ptr[0] + 111) % 112;
        self.d[0][self.ptr[0]] = 2. * y;
        self.dec[0] = self.d[0][(self.ptr[0] + 40) % 112];
        if self.n.is_multiple_of(2) {
            for j in 1..=2 {
                if !self.n.is_multiple_of(1 << j) {
                    break;
                }
                self.ptr[j] = (self.ptr[j] + 111) % 112;
                self.d[j][self.ptr[j]] = x;
                x *= self.h[0];
                for m in 1..17 {
                    x += self.h[m] * self.d[j][(self.ptr[j] + m) % 112];
                }
                let delay = if j == 1 { 24 } else { 8 };
                self.dec[j] = self.d[j][(self.ptr[j] + delay) % 112];
            }
        }
        let compute = self.n.is_multiple_of(4);
        let mut output = [0.; 40];
        for (p, c) in self.channels.iter_mut().enumerate() {
            if !self.n.is_multiple_of(c.step) {
                continue;
            }
            let mut y = c.gain * self.dec[c.index];
            for cell in &mut c.cells {
                y = cell.tick(y);
            }
            let zn1 = c.zn;
            let fn1 = c.fn_;
            c.zn = (y + 0.001).max(0.);
            c.w1n = c.g1q * (c.zn + zn1) + c.c1 * c.w1n;
            let new_w2 = c.w1n + c.c2 * c.w2n;
            if compute {
                c.qn = c.a1q * new_w2 - c.a2q * c.w2n;
                c.qfac = 0.;
            }
            c.w2n = new_w2;
            c.fn_ = if c.zn <= 0. {
                0.
            } else if c.qn <= 0. {
                self.factor2 * c.zn
            } else {
                if c.qfac == 0. {
                    c.qfac = 0.150 / (self.bias + c.qn.sqrt()).powi(2);
                }
                c.qfac * c.zn
            };
            c.y1n = c.g1 * (c.fn_ + fn1) + c.b * c.y1n;
            let w = c.g2 * c.y1n - c.b1 * c.wn1 - c.b2 * c.wn2;
            if compute {
                output[p] = (w + 2. * c.wn1 + c.wn2).max(0.);
            }
            c.wn2 = c.wn1;
            c.wn1 = w;
        }
        self.n += 1;
        if compute { Some(output) } else { None }
    }
}
