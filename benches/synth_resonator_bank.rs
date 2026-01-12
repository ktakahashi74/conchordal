//! Benchmarks for ResonatorBank.
//!
//! Run:
//! - cargo bench
//! - cargo bench --features "bench-hooks" --no-default-features
//! - CONCHORDAL_BENCH_FULL=1 cargo bench

#[cfg(feature = "bench-hooks")]
use std::env;
#[cfg(feature = "bench-hooks")]
use std::f32::consts::PI;
#[cfg(feature = "bench-hooks")]
use std::time::Instant;

#[cfg(feature = "bench-hooks")]
use conchordal::synth::modes::ModeParams;
#[cfg(feature = "bench-hooks")]
use conchordal::synth::resonator::ResonatorBank;
#[cfg(feature = "bench-hooks")]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[cfg(feature = "bench-hooks")]
const FS: f32 = 48_000.0;
#[cfg(feature = "bench-hooks")]
const BLOCK_LENS_QUICK: [usize; 2] = [64, 256];
#[cfg(feature = "bench-hooks")]
const BLOCK_LENS_FULL: [usize; 3] = [64, 256, 1024];
#[cfg(feature = "bench-hooks")]
const MODES_LENS_QUICK: [usize; 3] = [1, 8, 32];
#[cfg(feature = "bench-hooks")]
const MODES_LENS_FULL: [usize; 4] = [1, 8, 32, 128];

#[cfg(feature = "bench-hooks")]
fn build_modes(modes_len: usize) -> Vec<ModeParams> {
    let mut modes = Vec::with_capacity(modes_len);
    for i in 0..modes_len {
        let idx = i as f32;
        modes.push(ModeParams {
            freq_hz: 110.0 + idx * 3.7,
            t60_s: 0.05 + idx * 0.001,
            gain: 0.2 + idx * 0.0001,
            in_gain: 0.1 + idx * 0.0002,
        });
    }
    modes
}

#[cfg(feature = "bench-hooks")]
fn make_sine(block_len: usize, fs: f32) -> Vec<f32> {
    let step = 2.0 * PI * 440.0 / fs;
    (0..block_len).map(|i| (step * i as f32).sin()).collect()
}

#[cfg(feature = "bench-hooks")]
fn make_impulse(block_len: usize) -> Vec<f32> {
    let mut input = vec![0.0; block_len];
    if block_len > 0 {
        input[0] = 1.0;
    }
    input
}

#[cfg(feature = "bench-hooks")]
fn bench_config() -> (&'static [usize], &'static [usize], usize) {
    let full = env::var("CONCHORDAL_BENCH_FULL")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if full {
        (&MODES_LENS_FULL, &BLOCK_LENS_FULL, 50)
    } else {
        (&MODES_LENS_QUICK, &BLOCK_LENS_QUICK, 20)
    }
}

#[cfg(feature = "bench-hooks")]
fn rough_ns_per_sample_scalar(
    bank: &mut ResonatorBank,
    input: &[f32],
    output: &mut [f32],
    iters: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        bank.reset_state();
        bank.process_block_mono_scalar(black_box(input), black_box(&mut output[..]));
        black_box(&output[..]);
    }
    let total_samples = input.len() * iters;
    start.elapsed().as_secs_f64() * 1.0e9 / total_samples as f64
}

#[cfg(all(feature = "bench-hooks", feature = "simd-wide"))]
fn rough_ns_per_sample_simd(
    bank: &mut ResonatorBank,
    input: &[f32],
    output: &mut [f32],
    iters: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        bank.reset_state();
        bank.process_block_mono_simd(black_box(input), black_box(&mut output[..]));
        black_box(&output[..]);
    }
    let total_samples = input.len() * iters;
    start.elapsed().as_secs_f64() * 1.0e9 / total_samples as f64
}

#[cfg(feature = "bench-hooks")]
fn bench_block_sine(c: &mut Criterion) {
    let mut group = c.benchmark_group("bank_block_sine");
    let (modes_lens, block_lens, sample_size) = bench_config();
    group.sample_size(sample_size);

    for &modes_len in modes_lens {
        let modes = build_modes(modes_len);
        for &block_len in block_lens {
            let input = make_sine(block_len, FS);
            let mut output_scalar = vec![0.0; block_len];
            let mut bank = ResonatorBank::new(FS, modes_len).unwrap();
            bank.set_modes(&modes).unwrap();
            let mut bank_scalar = bank.clone();

            let scalar_id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}_scalar"));
            group.bench_with_input(scalar_id, &input, |b, input| {
                b.iter(|| {
                    bank_scalar.reset_state();
                    bank_scalar.process_block_mono_scalar(
                        black_box(input),
                        black_box(&mut output_scalar[..]),
                    );
                    black_box(&output_scalar[..]);
                });
            });

            #[cfg(feature = "simd-wide")]
            {
                let mut output_simd = vec![0.0; block_len];
                let mut bank_simd = bank.clone();
                let simd_id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}_simd"));
                group.bench_with_input(simd_id, &input, |b, input| {
                    b.iter(|| {
                        bank_simd.reset_state();
                        bank_simd.process_block_mono_simd(
                            black_box(input),
                            black_box(&mut output_simd[..]),
                        );
                        black_box(&output_simd[..]);
                    });
                });

                let mut bank_scalar_speed = bank.clone();
                let mut bank_simd_speed = bank.clone();
                let mut output_scalar_speed = vec![0.0; block_len];
                let mut output_simd_speed = vec![0.0; block_len];
                let ns_scalar = rough_ns_per_sample_scalar(
                    &mut bank_scalar_speed,
                    &input,
                    &mut output_scalar_speed,
                    50,
                );
                let ns_simd = rough_ns_per_sample_simd(
                    &mut bank_simd_speed,
                    &input,
                    &mut output_simd_speed,
                    50,
                );
                let speedup = ns_scalar / ns_simd;
                println!(
                    "speedup bank_block_sine m{modes_len}_b{block_len}: scalar {:.2} ns/sample, simd {:.2} ns/sample, speedup {:.2}x",
                    ns_scalar, ns_simd, speedup
                );
            }
        }
    }

    group.finish();
}

#[cfg(feature = "bench-hooks")]
fn bench_block_impulse(c: &mut Criterion) {
    let mut group = c.benchmark_group("bank_block_impulse");
    let (modes_lens, block_lens, sample_size) = bench_config();
    group.sample_size(sample_size);

    for &modes_len in modes_lens {
        let modes = build_modes(modes_len);
        for &block_len in block_lens {
            let input = make_impulse(block_len);
            let mut output_scalar = vec![0.0; block_len];
            let mut bank = ResonatorBank::new(FS, modes_len).unwrap();
            bank.set_modes(&modes).unwrap();
            let mut bank_scalar = bank.clone();

            let scalar_id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}_scalar"));
            group.bench_with_input(scalar_id, &input, |b, input| {
                b.iter(|| {
                    bank_scalar.reset_state();
                    bank_scalar.process_block_mono_scalar(
                        black_box(input),
                        black_box(&mut output_scalar[..]),
                    );
                    black_box(&output_scalar[..]);
                });
            });

            #[cfg(feature = "simd-wide")]
            {
                let mut output_simd = vec![0.0; block_len];
                let mut bank_simd = bank.clone();
                let simd_id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}_simd"));
                group.bench_with_input(simd_id, &input, |b, input| {
                    b.iter(|| {
                        bank_simd.reset_state();
                        bank_simd.process_block_mono_simd(
                            black_box(input),
                            black_box(&mut output_simd[..]),
                        );
                        black_box(&output_simd[..]);
                    });
                });

                let mut bank_scalar_speed = bank.clone();
                let mut bank_simd_speed = bank.clone();
                let mut output_scalar_speed = vec![0.0; block_len];
                let mut output_simd_speed = vec![0.0; block_len];
                let ns_scalar = rough_ns_per_sample_scalar(
                    &mut bank_scalar_speed,
                    &input,
                    &mut output_scalar_speed,
                    50,
                );
                let ns_simd = rough_ns_per_sample_simd(
                    &mut bank_simd_speed,
                    &input,
                    &mut output_simd_speed,
                    50,
                );
                let speedup = ns_scalar / ns_simd;
                println!(
                    "speedup bank_block_impulse m{modes_len}_b{block_len}: scalar {:.2} ns/sample, simd {:.2} ns/sample, speedup {:.2}x",
                    ns_scalar, ns_simd, speedup
                );
            }
        }
    }

    group.finish();
}

#[cfg(feature = "bench-hooks")]
criterion_group!(synth_resonator_bank, bench_block_sine, bench_block_impulse);
#[cfg(feature = "bench-hooks")]
criterion_main!(synth_resonator_bank);

#[cfg(not(feature = "bench-hooks"))]
fn main() {}
