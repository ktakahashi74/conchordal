//! Finite room responses at 48 kHz. See docs/room-generation.md.

use rustfft::{FftNum, num_complex::Complex};

use std::sync::Arc;

pub(crate) const SAMPLE_RATE: u32 = 48_000;
const FS: f64 = SAMPLE_RATE as f64;
const LENGTH: usize = 351_360;
const DELAY: usize = 121;
const CENTRES: [f64; 10] = [
    31.5, 63., 125., 250., 500., 1000., 2000., 4000., 8000., 16000.,
];
const DECAY: [f64; 10] = [5.90, 6.88, 7.24, 7.22, 6.92, 6.10, 4.75, 3.26, 1.80, 1.03];

struct Radix2<T> {
    reverse: Vec<usize>,
    roots: Vec<Complex<T>>,
}

impl<T: FftNum> Radix2<T> {
    fn new(n: usize) -> Self {
        let reverse = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - n.ilog2()))
            .collect();
        let roots = (0..n / 2)
            .map(|k| {
                let phase = -std::f64::consts::TAU * k as f64 / n as f64;
                let (s, c) = phase.sin_cos();
                Complex::new(T::from_f64(c).unwrap(), T::from_f64(s).unwrap())
            })
            .collect();
        Self { reverse, roots }
    }

    fn transform(&self, x: &mut [Complex<T>], inverse: bool) {
        let n = x.len();
        for (i, &j) in self.reverse.iter().enumerate() {
            if j > i {
                x.swap(i, j);
            }
        }
        let mut width = 2;
        while width <= n {
            for chunk in x.chunks_exact_mut(width) {
                let (lo, hi) = chunk.split_at_mut(width / 2);
                for (k, (a, b)) in lo.iter_mut().zip(hi).enumerate() {
                    let mut root = self.roots[k * (n / width)];
                    if inverse {
                        root.im = -root.im;
                    }
                    let t = *b * root;
                    *b = *a - t;
                    *a = *a + t;
                }
            }
            width *= 2;
        }
        if inverse {
            let scale = T::from_f64(1.0 / n as f64).unwrap();
            for z in x {
                *z = *z * scale;
            }
        }
    }
}

// PCG XSL RR 128/64, fixed state and odd increment.
fn phase(state: &mut u128) -> f64 {
    let old = *state;
    *state = old
        .wrapping_mul(47026247687942121848144207491837523525)
        .wrapping_add(1);
    let word = ((old >> 64) as u64 ^ old as u64).rotate_right((old >> 122) as u32);
    (word >> 11) as f64 * (std::f64::consts::TAU / (1u64 << 53) as f64)
}

/// Synthesize the complete impulse response, including its unit direct tap.
pub fn york_response() -> Vec<f32> {
    synthesize(false)
}

/// Presentation-only right ear, sharing the left ear's direct tap and decay targets.
pub fn york_right_response() -> Vec<f32> {
    synthesize(true)
}

fn coherence(hz: f64) -> f64 {
    let table = include_bytes!("room_coherence.f64le");
    let at = hz / 25.;
    let index = at.floor() as usize;
    let value = |i: usize| f64::from_le_bytes(table[i * 8..i * 8 + 8].try_into().unwrap());
    value(index) + at.fract() * (value(index + 1) - value(index))
}

fn synthesize(right: bool) -> Vec<f32> {
    let m = LENGTH.next_power_of_two();
    let plan = Radix2::new(m);
    let mut spectrum = vec![Complex::default(); m];
    let mut tail = vec![0.0; LENGTH - DELAY];
    let mut state = 17;
    let mut independent = 31;
    for (k, (&centre, &rt)) in CENTRES.iter().zip(&DECAY).enumerate() {
        let low = if k == 0 {
            centre / 2f64.sqrt()
        } else {
            (CENTRES[k - 1] * centre).sqrt()
        };
        let high = if k + 1 == CENTRES.len() {
            centre * 2f64.sqrt()
        } else {
            (CENTRES[k + 1] * centre).sqrt()
        };
        let bin = |hz: f64| (hz * m as f64 / FS).ceil() as usize;
        let bins = bin(low)..bin(high);
        // Parseval normalization gives each carrier unit mean square.
        let mut amplitude = m as f64 / (2.0 * bins.len() as f64).sqrt();
        spectrum.fill(Complex::default());
        for j in bins {
            let (s, c) = phase(&mut state).sin_cos();
            spectrum[j] = Complex::new(c, s);
            if right {
                let (s, c) = phase(&mut independent).sin_cos();
                let gamma = coherence(j as f64 * FS / m as f64);
                spectrum[j] =
                    spectrum[j] * gamma + Complex::new(c, s) * (1. - gamma * gamma).sqrt();
            }
            spectrum[m - j] = spectrum[j].conj();
        }
        plan.transform(&mut spectrum, true);
        let decay = 10f64.powf(-3.0 / (FS * rt));
        for (out, z) in tail.iter_mut().zip(&spectrum) {
            *out += z.re * amplitude;
            amplitude *= decay;
        }
    }
    for (i, v) in tail.iter_mut().take(96).enumerate() {
        *v *= 0.5 - 0.5 * (std::f64::consts::PI * (i + 1) as f64 / 96.).cos();
    }
    let gain = (10f64.powf(5.16 / 10.0) / tail.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let mut ir = vec![0.; LENGTH];
    ir[0] = 1.;
    for (h, x) in ir[DELAY..].iter_mut().zip(tail) {
        *h = (gain * x) as f32;
    }
    ir
}

/// Immutable partition plan. The FIR is supplied independently of the hop size.
pub struct RoomKernel(Arc<RoomPlan>);

struct RoomPlan {
    blocks: Vec<Complex<f32>>,
    fft: Radix2<f32>,
    hop: usize,
    length: usize,
    direct: f32,
}

impl RoomKernel {
    /// Prepare once, only for an enabled room. Tap zero is added in the time domain.
    pub fn new(response: &[f32], hop: usize) -> Self {
        assert!(hop > 0 && !response.is_empty());
        let m = (2 * hop).next_power_of_two();
        let bins = m / 2 + 1;
        let fft = Radix2::new(m);
        let mut scratch = vec![Complex::default(); m];
        let mut blocks = Vec::with_capacity(response.len().div_ceil(hop) * bins);
        for (i, part) in response.chunks(hop).enumerate() {
            scratch.fill(Complex::default());
            for (z, &x) in scratch.iter_mut().zip(part) {
                z.re = x;
            }
            if i == 0 {
                scratch[0].re = 0.;
            }
            fft.transform(&mut scratch, false);
            blocks.extend_from_slice(&scratch[..bins]);
        }
        Self(Arc::new(RoomPlan {
            blocks,
            fft,
            hop,
            length: response.len(),
            direct: response[0],
        }))
    }

    /// Bytes of fixed storage in one source history (including its slot).
    pub(crate) fn stream_bytes(&self) -> usize {
        let plan = &self.0;
        let fft_len = plan.fft.reverse.len();
        size_of::<RoomStream>()
            + (plan.blocks.len() + fft_len) * size_of::<Complex<f32>>()
            + plan.blocks.len() / (fft_len / 2 + 1)
            + plan.hop * size_of::<f32>()
    }

    /// Reserve independently addressable slots for all simultaneous generations.
    pub fn stream_pool(
        &self,
        slots: usize,
        required: usize,
    ) -> Result<Vec<RoomStream>, &'static str> {
        if slots < required {
            return Err("room pool cannot hold all live and retiring streams");
        }
        let mut pool = Vec::new();
        pool.try_reserve_exact(slots)
            .map_err(|_| "room pool allocation failed")?;
        for _ in 0..slots {
            pool.push(RoomStream::try_new(&self.0)?);
        }
        Ok(pool)
    }
}

/// History for one continuous input. Slots share the same immutable plan.
pub struct RoomStream {
    kernel: Arc<RoomPlan>,
    history: Vec<Complex<f32>>,
    nonzero: Vec<bool>,
    scratch: Vec<Complex<f32>>,
    overlap: Vec<f32>,
    head: usize,
    valid: usize,
    remaining: usize,
}

impl RoomStream {
    fn try_new(kernel: &Arc<RoomPlan>) -> Result<Self, &'static str> {
        fn zeroed<T: Default + Clone>(len: usize) -> Result<Vec<T>, &'static str> {
            let mut v = Vec::new();
            v.try_reserve_exact(len)
                .map_err(|_| "room history allocation failed")?;
            v.resize(len, T::default());
            Ok(v)
        }
        let m = kernel.fft.reverse.len();
        Ok(Self {
            kernel: Arc::clone(kernel),
            history: zeroed(kernel.blocks.len())?,
            nonzero: zeroed(kernel.blocks.len() / (m / 2 + 1))?,
            scratch: zeroed(m)?,
            overlap: zeroed(kernel.hop)?,
            head: 0,
            valid: 0,
            remaining: 0,
        })
    }

    /// After processing a hop, no sample of the current FIR support lies beyond it.
    pub(crate) fn drained(&self) -> bool {
        self.remaining == 0
    }

    /// Reuse a retired slot for an unrelated input.
    pub fn reset(&mut self) {
        self.valid = 0;
        self.remaining = 0;
        self.overlap.fill(0.);
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let hop = self.kernel.hop;
        let length = self.kernel.length;
        assert_eq!(input.len(), hop);
        assert_eq!(output.len(), hop);
        // A tail and a fresh input can meet inside the same external hop.
        let mut start = 0;
        let mut end = self.remaining;
        if end < hop {
            for (i, &x) in input.iter().enumerate() {
                if x == 0. {
                    continue;
                }
                if end > start && i >= end {
                    self.render_part(start, &input[start..i], &mut output[start..i]);
                    start = i;
                }
                end = i.saturating_add(length);
                if end >= hop {
                    break;
                }
            }
        }
        self.render_part(start, &input[start..], &mut output[start..]);
    }

    fn render_part(&mut self, offset: usize, input: &[f32], output: &mut [f32]) {
        let k = &self.kernel;
        let live = input.iter().any(|&x| x != 0.);
        if !live && self.remaining == 0 {
            output.fill(0.);
            return;
        }
        let m = self.scratch.len();
        let bins = m / 2 + 1;
        let slots = self.nonzero.len();
        self.nonzero[self.head] = live;
        if live {
            self.scratch.fill(Complex::default());
            for (z, &x) in self.scratch[offset..].iter_mut().zip(input) {
                z.re = x;
            }
            k.fft.transform(&mut self.scratch, false);
            self.history[self.head * bins..(self.head + 1) * bins]
                .copy_from_slice(&self.scratch[..bins]);
        }
        self.scratch.fill(Complex::default());
        self.valid = (self.valid + 1).min(slots);
        let mut slot = self.head;
        for filter in k.blocks.chunks_exact(bins).take(self.valid) {
            if self.nonzero[slot] {
                let history = &self.history[slot * bins..(slot + 1) * bins];
                for ((sum, x), h) in self.scratch[..bins].iter_mut().zip(history).zip(filter) {
                    *sum += *x * *h;
                }
            }
            slot = if slot == 0 { slots - 1 } else { slot - 1 };
        }
        self.head = (self.head + 1) % slots;
        for j in 1..m / 2 {
            self.scratch[m - j] = self.scratch[j].conj();
        }
        k.fft.transform(&mut self.scratch, true);
        for (i, (&x, y)) in input.iter().zip(output).enumerate() {
            if x != 0. {
                self.remaining = k.length;
            }
            *y = if self.remaining > 0 {
                self.remaining -= 1;
                x * k.direct + (self.scratch[offset + i].re + self.overlap[offset + i])
            } else {
                0.
            };
        }
        for (dst, z) in self.overlap[..k.hop - 1]
            .iter_mut()
            .zip(&self.scratch[k.hop..])
        {
            *dst = z.re;
        }
        if self.remaining == 0 {
            self.reset();
        }
    }
}

#[cfg(test)]
#[path = "room_tests.rs"]
mod tests;
