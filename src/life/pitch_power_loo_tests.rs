//! Bounded pre-peak power subtraction candidate. Not promoted into the live path.
use super::*;
use crate::core::landscape::{Landscape, LandscapeParams};
use crate::core::landscape_spectral::SpectralFrontEnd;
use crate::core::nsgt_kernel::{NsgtKernelLog2, PowerMode};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use rustfft::num_complex::Complex64;

pub(super) struct PowerLoo {
    kernel: Arc<NsgtKernelLog2>,
    alpha: Vec<f32>,
    positive: Vec<Complex64>,
    negative: Vec<Complex64>,
    instantaneous: Vec<f32>,
    smooth: Vec<f32>,
    residual: Vec<f32>,
    frontend: SpectralFrontEnd,
    potentials: PotentialScratch,
    h: Vec<f32>,
    r: Vec<f32>,
    pub(super) score: Vec<f32>,
    pub(super) density: Vec<f32>,
}

impl PowerLoo {
    pub(super) fn new(rt: &RtNsgtKernelLog2, params: &LandscapeParams) -> Self {
        let (kernel, alpha) = rt.power_response_metadata();
        let n = kernel.space.n_bins();
        let m = kernel.nfft();
        Self {
            frontend: SpectralFrontEnd::new(kernel.space.clone()),
            potentials: PotentialScratch::new(
                &kernel.space,
                &params.harmonicity_kernel,
                &params.roughness_kernel,
            ),
            kernel,
            alpha,
            positive: vec![Complex64::default(); m],
            negative: vec![Complex64::default(); m],
            instantaneous: vec![0.; n],
            smooth: vec![0.; n],
            residual: vec![0.; n],
            h: vec![0.; n],
            r: vec![0.; n],
            score: vec![0.; n],
            density: vec![0.; n],
        }
    }

    // E_phase |A sin(w n + phase) . h*|^2. Use the actual sparse FFT kernel,
    // including both real-signal sidebands and the producer's normalization.
    pub(super) fn project(&mut self, partials: &[(f32, f32)], occupied: usize) {
        let m = self.kernel.nfft();
        let len = occupied.min(m);
        let start = m - len;
        self.instantaneous.fill(0.);
        for &(hz, amp) in partials {
            if amp == 0. {
                continue;
            }
            for k in 0..m {
                for (sign, buffer) in [(1., &mut self.positive), (-1., &mut self.negative)] {
                    let cycles = sign * hz as f64 / self.kernel.cfg.fs as f64 - k as f64 / m as f64;
                    let theta = std::f64::consts::TAU * (cycles - cycles.round());
                    let magnitude = if theta == 0. {
                        len as f64
                    } else {
                        (0.5 * len as f64 * theta).sin() / (0.5 * theta).sin()
                    };
                    let phase = theta * (start as f64 + (len as f64 - 1.) * 0.5);
                    buffer[k] = Complex64::from_polar(magnitude, phase);
                }
            }
            for (i, band) in self.kernel.bands().iter().enumerate() {
                let energy = match self.kernel.power_mode {
                    PowerMode::Coherent => {
                        let mut pos = Complex64::default();
                        let mut neg = Complex64::default();
                        for &(k, w) in band.spec_conj_sparse.as_ref().unwrap() {
                            let w = Complex64::new(w.re as f64, w.im as f64);
                            pos += self.positive[k] * w;
                            neg += self.negative[k] * w;
                        }
                        pos.norm_sqr() + neg.norm_sqr()
                    }
                    PowerMode::Incoherent => band
                        .spec_pow_sparse
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&(k, w)| {
                            (self.positive[k].norm_sqr() + self.negative[k].norm_sqr()) * w as f64
                        })
                        .sum(),
                };
                self.instantaneous[i] +=
                    (amp as f64 * amp as f64 * energy / (4. * (m * m) as f64)) as f32;
            }
        }
    }

    pub(super) fn witness(&self, mixed: &Landscape, reference: &Landscape) -> serde_json::Value {
        let du = &self.potentials.du;
        serde_json::json!({
            "estimated_density_mass": self.density.iter().zip(du.iter()).map(|(a,b)| a*b).sum::<f32>(),
            "reference_density_mass": reference.subjective_intensity.iter().zip(du.iter()).map(|(a,b)| a*b).sum::<f32>(),
            "floor": mixed.roughness_ref_eps.max(1e-12),
            "bins": (0..self.density.len()).filter(|&i| self.density[i] > 0. || reference.subjective_intensity[i] > 0.).map(|i| {
                serde_json::json!([i,mixed.space.centers_hz[i],mixed.nsgt_power[i],self.smooth[i],self.residual[i],reference.nsgt_power[i],self.density[i],reference.subjective_intensity[i]])
            }).collect::<Vec<_>>()
        })
    }

    pub(super) fn process(&mut self, observed: &Landscape, params: &LandscapeParams) {
        self.kernel
            .space
            .assert_scan_len_named(&observed.nsgt_power, "nsgt_power_scan");
        assert_eq!(self.kernel.space.centers_hz, observed.space.centers_hz);
        for i in 0..self.smooth.len() {
            let y = (1. - self.alpha[i]) * self.instantaneous[i] + self.alpha[i] * self.smooth[i];
            self.smooth[i] = if y.is_normal() { y } else { 0. };
            self.residual[i] = (observed.nsgt_power[i] - self.smooth[i]).max(0.);
        }
        let density = self
            .frontend
            .process_nsgt_power(&self.residual, self.kernel.hop() as f32 / params.fs, params)
            .subjective_intensity;
        self.density.copy_from_slice(density);
        let eps = observed.roughness_ref_eps.max(1e-12);
        self.potentials.compute_shape(
            density,
            &observed.space,
            (&params.harmonicity_kernel, &params.roughness_kernel),
            eps,
            &mut self.h,
            &mut self.r,
        );
        for i in 0..self.score.len() {
            let h = self.h[i].clamp(0., 1.);
            let r = roughness_ratio_to_state01(
                self.r[i] / observed.roughness_ref_peak.max(eps),
                observed.roughness_k.max(1e-6),
            );
            self.score[i] = observed.consonance_kernel.score(h, r);
        }
    }
}

#[test]
fn sparse_response_matches_phase_averaged_actual_nsgt() {
    let (rt, params) = crate::runtime::body_metabolism_test_core();
    let mut candidate = PowerLoo::new(&rt, &params);
    let m = rt.nfft();
    let hop = rt.hop();
    println!(
        "T38_RESPONSE_META bins={} nfft={} sparse_weights={}",
        candidate.kernel.space.n_bins(),
        m,
        candidate
            .kernel
            .bands()
            .iter()
            .map(|b| b.spec_conj_sparse.as_ref().map_or(0, Vec::len))
            .sum::<usize>()
    );
    for hz in [440., 466., 1800.] {
        let mut sin = rt.clone();
        let mut cos = rt.clone();
        let mut pcm_sin = vec![0.; hop];
        let mut pcm_cos = vec![0.; hop];
        for frame in 0..(m / hop + 4) {
            for i in 0..hop {
                let phase =
                    std::f64::consts::TAU * hz as f64 * (frame * hop + i) as f64 / params.fs as f64;
                pcm_sin[i] = (0.35 * phase.sin()) as f32;
                pcm_cos[i] = (0.35 * phase.cos()) as f32;
            }
            let a = sin.process_hop(&pcm_sin);
            let b = cos.process_hop(&pcm_cos);
            candidate.project(&[(hz, 0.35)], (frame + 1) * hop);
            for i in 0..a.len() {
                candidate.smooth[i] = (1. - candidate.alpha[i]) * candidate.instantaneous[i]
                    + candidate.alpha[i] * candidate.smooth[i];
                let expected = 0.5 * (a[i] + b[i]);
                // This validates the response evaluator, not a new consumer tolerance.
                assert!(
                    (candidate.smooth[i] - expected).abs() < 32. * f32::EPSILON * expected.max(1.),
                    "hz={hz} frame={frame} bin={i} got={} expected={expected}",
                    candidate.smooth[i]
                );
            }
        }
        candidate.smooth.fill(0.);
    }
}

#[test]
#[cfg(feature = "profile-alloc")]
fn power_loo_steady_state_allocates_nothing() {
    let (rt, params) = crate::runtime::body_metabolism_test_core();
    let mut stream = crate::core::stream::analysis::AnalysisStream::new(params.clone(), rt.clone());
    let mut pcm = vec![0.; rt.hop()];
    let mut candidates: Vec<_> = (0..3).map(|_| PowerLoo::new(&rt, &params)).collect();
    for frame in 0..64 {
        for (i, sample) in pcm.iter_mut().enumerate() {
            let t = (frame * rt.hop() + i) as f64 / params.fs as f64;
            *sample = [440., 466., 880.]
                .iter()
                .map(|f| 0.35 * (std::f64::consts::TAU * f * t).sin())
                .sum::<f64>() as f32;
        }
        let observed = stream.process(&pcm);
        for (candidate, hz) in candidates.iter_mut().zip([440., 466., 880.]) {
            candidate.project(&[(hz, 0.35)], (frame + 1) * rt.hop());
            candidate.process(observed, &params);
        }
    }
    let observed = stream.last();
    assert!(candidates.iter().all(|c| c.density.iter().any(|&v| v > 0.)));
    crate::runtime_profile::begin_allocations();
    for _ in 0..3 {
        for (candidate, hz) in candidates.iter_mut().zip([440., 466., 880.]) {
            candidate.project(&[(hz, 0.35)], rt.nfft());
            candidate.process(observed, &params);
        }
    }
    let counts = crate::runtime_profile::finish_allocations().unwrap();
    println!(
        "T38_POWER_ALLOC voices=3 iterations=3 bins={} {counts:?}",
        observed.space.n_bins()
    );
    assert_eq!((counts.count, counts.bytes), (0, 0));
}
