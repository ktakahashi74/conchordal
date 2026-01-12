//! Benchmarks for ResonatorBank.
//!
//! Run:
//! - cargo bench
//! - cargo bench --no-default-features
//! - cargo bench --features simd-wide

use std::f32::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use conchordal::synth::modes::ModeParams;
use conchordal::synth::resonator::ResonatorBank;

const FS: f32 = 48_000.0;
const BLOCK_LENS: [usize; 3] = [64, 256, 1024];
const MODES_LENS: [usize; 4] = [1, 8, 32, 128];

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

fn make_sine(block_len: usize, fs: f32) -> Vec<f32> {
    let step = 2.0 * PI * 440.0 / fs;
    (0..block_len)
        .map(|i| (step * i as f32).sin())
        .collect()
}

fn make_impulse(block_len: usize) -> Vec<f32> {
    let mut input = vec![0.0; block_len];
    if block_len > 0 {
        input[0] = 1.0;
    }
    input
}

fn bench_block_sine(c: &mut Criterion) {
    let mut group = c.benchmark_group("bank_block_sine");
    group.sample_size(50);

    for &modes_len in &MODES_LENS {
        let modes = build_modes(modes_len);
        for &block_len in &BLOCK_LENS {
            let input = make_sine(block_len, FS);
            let mut output = vec![0.0; block_len];
            let mut bank = ResonatorBank::new(FS, modes_len).unwrap();
            bank.set_modes(&modes).unwrap();

            let id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}"));
            group.bench_with_input(id, &input, |b, input| {
                b.iter(|| {
                    bank.reset_state();
                    bank.process_block_mono(black_box(input), black_box(&mut output));
                    black_box(&output);
                });
            });
        }
    }

    group.finish();
}

fn bench_block_impulse(c: &mut Criterion) {
    let mut group = c.benchmark_group("bank_block_impulse");
    group.sample_size(50);

    for &modes_len in &MODES_LENS {
        let modes = build_modes(modes_len);
        for &block_len in &BLOCK_LENS {
            let input = make_impulse(block_len);
            let mut output = vec![0.0; block_len];
            let mut bank = ResonatorBank::new(FS, modes_len).unwrap();
            bank.set_modes(&modes).unwrap();

            let id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}"));
            group.bench_with_input(id, &input, |b, input| {
                b.iter(|| {
                    bank.reset_state();
                    bank.process_block_mono(black_box(input), black_box(&mut output));
                    black_box(&output);
                });
            });
        }
    }

    group.finish();
}

fn bench_sample_sine(c: &mut Criterion) {
    let mut group = c.benchmark_group("bank_sample_sine");
    group.sample_size(50);

    for &modes_len in &MODES_LENS {
        let modes = build_modes(modes_len);
        for &block_len in &BLOCK_LENS {
            let input = make_sine(block_len, FS);
            let mut bank = ResonatorBank::new(FS, modes_len).unwrap();
            bank.set_modes(&modes).unwrap();

            let id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}"));
            group.bench_with_input(id, &input, |b, input| {
                b.iter(|| {
                    bank.reset_state();
                    let mut acc = 0.0;
                    for &u in input.iter() {
                        acc += bank.process_sample(u);
                    }
                    black_box(acc);
                });
            });
        }
    }

    group.finish();
}

fn bench_sample_impulse(c: &mut Criterion) {
    let mut group = c.benchmark_group("bank_sample_impulse");
    group.sample_size(50);

    for &modes_len in &MODES_LENS {
        let modes = build_modes(modes_len);
        for &block_len in &BLOCK_LENS {
            let input = make_impulse(block_len);
            let mut bank = ResonatorBank::new(FS, modes_len).unwrap();
            bank.set_modes(&modes).unwrap();

            let id = BenchmarkId::new("case", format!("m{modes_len}_b{block_len}"));
            group.bench_with_input(id, &input, |b, input| {
                b.iter(|| {
                    bank.reset_state();
                    let mut acc = 0.0;
                    for &u in input.iter() {
                        acc += bank.process_sample(u);
                    }
                    black_box(acc);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    synth_resonator_bank,
    bench_block_sine,
    bench_block_impulse,
    bench_sample_sine,
    bench_sample_impulse
);
criterion_main!(synth_resonator_bank);
