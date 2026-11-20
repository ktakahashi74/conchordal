use super::*;
use rustfft::{FftPlanner, num_complex::Complex64};
use sha2::{Digest, Sha256};

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&x| f64::from(x).powi(2)).sum()
}

fn spectrum(x: &[f32]) -> Vec<Complex64> {
    let n = (2 * x.len()).next_power_of_two();
    let mut z = vec![Complex64::default(); n];
    for (z, &x) in z.iter_mut().zip(x) {
        z.re = x.into();
    }
    FftPlanner::new().plan_fft_forward(n).process(&mut z);
    z
}

// Independent zero-phase analysis, with zero padding around the finite input.
fn decays(x: &[f32], check: bool) {
    let original = spectrum(x);
    let n = original.len();
    let ifft = FftPlanner::new().plan_fft_inverse(n);
    let centres = [
        31.5_f64, 63., 125., 250., 500., 1000., 2000., 4000., 8000., 16000.,
    ];
    let expected = [5.90, 6.88, 7.24, 7.22, 6.92, 6.10, 4.75, 3.26, 1.80, 1.03];
    for (k, &hz) in centres.iter().enumerate() {
        let lo = if k == 0 {
            hz / 2f64.sqrt()
        } else {
            (hz * centres[k - 1]).sqrt()
        };
        let hi = if k == 9 {
            hz * 2f64.sqrt()
        } else {
            (hz * centres[k + 1]).sqrt()
        };
        let mut band = original.clone();
        for (i, v) in band.iter_mut().enumerate() {
            let f = i.min(n - i) as f64 * 48_000. / n as f64;
            if !(lo..hi).contains(&f) {
                *v = Complex64::default();
            }
        }
        ifft.process(&mut band);
        let mut sum = 0.;
        let mut edc: Vec<_> = band[..x.len()]
            .iter()
            .rev()
            .map(|z| {
                sum += z.re * z.re;
                sum
            })
            .collect();
        edc.reverse();
        for y in &mut edc {
            *y = 10. * (*y / sum).log10();
        }
        let start = edc.partition_point(|&y| y > -5.);
        let stop = edc.partition_point(|&y| y >= -35.);
        let ys = &edc[start..stop];
        let count = ys.len() as f64;
        let mean_x = (count - 1.) / 2.;
        let mean_y = ys.iter().sum::<f64>() / count;
        let mut xx = 0.;
        let mut xy = 0.;
        let mut yy = 0.;
        for (i, &y) in ys.iter().enumerate() {
            let dx = i as f64 - mean_x;
            xx += dx * dx;
            xy += dx * (y - mean_y);
            yy += (y - mean_y).powi(2);
        }
        let slope = xy / xx;
        let rt = -60. / (slope * 48_000.);
        let r2 = xy * xy / (xx * yy);
        let rms = (yy * (1. - r2) / count).sqrt();
        eprintln!(
            "room band late={check} Hz={hz} T30={rt:.4} R2={r2:.6} rms_dB={rms:.4} range={start}..{stop}"
        );
        if check {
            assert!((rt / expected[k] - 1.).abs() <= 0.15 && r2 >= 0.95);
        }
    }
}

#[test]
fn impulse_statistics() {
    for (h, right) in [(york_response(), false), (york_right_response(), true)] {
        assert_eq!(h.len(), 351_360);
        assert_eq!(h[0], 1.);
        assert!(h[1..121].iter().all(|x| x.to_bits() == 0));
        assert!(h[121] != 0. && h[3839] != 0. && h[3840] != 0. && h[h.len() - 1] != 0.);
        let digest = |h: &[f32]| {
            let mut d = Sha256::new();
            for x in h {
                d.update(x.to_le_bytes());
            }
            d.finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        let hash = digest(&h);
        assert_eq!(digest(&synthesize(right)), hash);
        let de = -10. * energy(&h[1..]).log10();
        let c80 = 10. * (energy(&h[..3840]) / energy(&h[3840..])).log10();
        let peak = spectrum(&h).iter().map(|z| z.norm()).fold(0., f64::max);
        let rms = (energy(&h[121..601]) / 480.).sqrt();
        eprintln!(
            "room SHA256={hash} D_E={de:.6} C80={c80:.6} Hpeak={peak:.6} first10msRMS={rms:.8}"
        );
        assert!((de + 5.16).abs() <= 0.1);
        decays(&h[3840..], true);
        decays(&h, false);
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(
            hash,
            if right {
                "c06ba2ee89027a7fd6700ac3fb18495e8d7506b6d6b36b4f9702f39117f482e1"
            } else {
                "46355804385b9ffbd73a6852b9fe3e2998574be6c42babcc63862f84c8d2710f"
            }
        );
        assert!((c80 + 2.429).abs() <= 0.1);
    }
}

#[test]
fn finite_streams_agree_with_direct_convolution() {
    let h = york_response();
    for hop in [512, 480] {
        let k = RoomKernel::new(&h, hop);
        let mut streams = k.stream_pool(2, 2).unwrap();
        let mut x = vec![0.; hop];
        let mut y = vec![0.; hop];
        let mut other = vec![0.; hop];
        let second = (h.len() + 3 * hop).next_multiple_of(hop) + 17;
        let events = [
            (0, 1.),
            (hop - 1, -0.75),
            (hop, 0.4),
            (2 * hop - 1, 0.5),
            (second, -1.),
        ];
        let mut max_err = 0f64;
        let mut alternate = vec![0.; hop];
        alternate[3] = 1.;
        for t in (0..second + h.len() + hop).step_by(hop) {
            x.fill(0.);
            for &(s, a) in &events {
                if (t..t + hop).contains(&s) {
                    x[s - t] = a;
                }
            }
            streams[1].process(&alternate, &mut other);
            alternate.fill(0.);
            streams[0].process(&x, &mut y);
            for (i, &sample) in y.iter().enumerate() {
                let at = t + i;
                let mut expected = 0.;
                let mut active = false;
                for &(s, a) in &events {
                    if let Some(lag) = at.checked_sub(s)
                        && let Some(&tap) = h.get(lag)
                    {
                        active = true;
                        expected += f64::from(tap) * f64::from(a);
                    }
                }
                assert!(sample.is_finite());
                if !active {
                    assert_eq!(sample.to_bits(), 0);
                }
                max_err = max_err.max((f64::from(sample) - expected).abs());
                let isolated = at
                    .checked_sub(3)
                    .and_then(|j| h.get(j))
                    .copied()
                    .unwrap_or(0.);
                assert!((other[i] - isolated).abs() < 1e-5);
            }
        }
        eprintln!("room direct hop={hop} maximum error={max_err:e}");
        assert!(max_err <= 1e-5);
        // Dense short input exercises cancellation in the direct reference.
        streams[0].reset();
        let short: Vec<f32> = (0..41)
            .map(|i| ((i * 47 % 67) as f32 - 33.) / 33.)
            .collect();
        for t in (0..h.len() + hop).step_by(hop) {
            x.fill(0.);
            if t == 0 {
                x[..short.len()].copy_from_slice(&short);
            }
            streams[0].process(&x, &mut y);
            for (i, &sample) in y.iter().enumerate() {
                let expected = short
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &v)| {
                        Some(f64::from(v) * f64::from(*h.get((t + i).checked_sub(j)?)?))
                    })
                    .sum::<f64>();
                assert!((expected - f64::from(sample)).abs() <= 1e-5);
            }
        }
    }
}

#[test]
fn restart_inside_expiring_hop_matches_fresh_slot() {
    let h = [1., 0., 0.37, -0.14, 0.22, -0.17];
    let k = RoomKernel::new(&h, 8);
    let mut slots = k.stream_pool(2, 2).unwrap();
    let mut y = [0.; 8];
    let mut fresh = y;
    slots[0].process(&[0., 0., 0., 0., 0., 0., 1., 0.], &mut y);
    let input = [0., 0., 0., 0., 0., -0.5, 0., 0.];
    slots[0].process(&input, &mut y);
    slots[1].process(&input, &mut fresh);
    assert_eq!(y[4].to_bits(), 0);
    assert_eq!(y[5..], fresh[5..]);
    slots[0].process(&[0.; 8], &mut y);
    slots[1].process(&[0.; 8], &mut fresh);
    assert_eq!(y, fresh);
    assert!(y[3..].iter().all(|x| x.to_bits() == 0));
}

#[test]
fn bounded_noise_drains_and_recycles() {
    let h = york_response();
    let bound = h.iter().map(|x| f64::from(x.abs())).sum::<f64>() + 1e-5;
    for b in [480, 512] {
        let k = RoomKernel::new(&h, b);
        assert!(k.stream_pool(1, 2).is_err());
        let mut streams = k.stream_pool(1, 1).unwrap();
        let mut x = vec![0.; b];
        let mut y = vec![0.; b];
        let mut rng = 17u32;
        let stop = h.len().next_multiple_of(b) + b - 7;
        #[cfg(feature = "profile-alloc")]
        crate::runtime_profile::begin_allocations();
        for t in (0..stop + h.len() + b).step_by(b) {
            for (i, v) in x.iter_mut().enumerate() {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = if t + i < stop {
                    (rng as i32) as f32 / 2147483648.0
                } else {
                    0.
                };
            }
            streams[0].process(&x, &mut y);
            for (i, &v) in y.iter().enumerate() {
                assert!(v.is_finite() && f64::from(v.abs()) <= bound);
                if t + i >= stop - 1 + h.len() {
                    assert_eq!(v.to_bits(), 0);
                }
            }
        }
        streams[0].reset();
        x[0] = 1.;
        streams[0].process(&x, &mut y);
        assert_eq!(y[0], 1.);
        #[cfg(feature = "profile-alloc")]
        {
            let a = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!((a.count, a.bytes), (0, 0));
        }
    }
}

#[cfg(all(feature = "native", feature = "profile-alloc"))]
#[test]
#[ignore = "release, exclusive timing window"]
fn room_cost() {
    use crate::core::{auditory_frontend::AuditoryFrontEnd, log2space::Log2Space};
    use crate::runtime_profile::{begin_allocations, finish_allocations, heap_live_bytes};
    use std::{hint::black_box, time::Instant};

    fn run(streams: &mut [RoomStream], x: &[f32], y: &mut [f32]) {
        for stream in streams {
            stream.process(black_box(x), y);
            black_box(&*y);
        }
    }
    fn summary(label: &str, v: &mut [f64]) {
        v.sort_by(f64::total_cmp);
        let q = |p: usize| v[(p * v.len()).div_ceil(100) - 1];
        eprintln!(
            "room time {label} n={} p50={:.3} p95={:.3} p99={:.3} max={:.3} over={}",
            v.len(),
            q(50),
            q(95),
            q(99),
            v[v.len() - 1],
            v.iter().filter(|&&us| us > 1e6 * 512. / 48000.).count()
        );
    }

    const B: usize = 512;
    const REPS: usize = 512;
    let start = Instant::now();
    let baseline = heap_live_bytes();
    begin_allocations();
    let h = york_response();
    let gen_alloc = finish_allocations().unwrap();
    let generation = start.elapsed();
    let start = Instant::now();
    let k = RoomKernel::new(&h, B);
    drop(h);
    let plan_time = start.elapsed();
    let shared = heap_live_bytes() - baseline;
    let start = Instant::now();
    let mut one = k.stream_pool(1, 1).unwrap();
    let one_heap = heap_live_bytes() - baseline - shared;
    let mut nine = k.stream_pool(9, 9).unwrap();
    let nine_heap = heap_live_bytes() - baseline - shared - one_heap;
    eprintln!(
        "room prepare gen_ms={:.3} plan_ms={:.3} pools_ms={:.3} generation_alloc_bytes={} shared_live={} one_live={} nine_live={}",
        generation.as_secs_f64() * 1e3,
        plan_time.as_secs_f64() * 1e3,
        start.elapsed().as_secs_f64() * 1e3,
        gen_alloc.bytes,
        shared,
        one_heap,
        nine_heap
    );

    let (_, params) = crate::runtime::body_metabolism_test_core();
    let mut frontend = AuditoryFrontEnd::selected(Log2Space::new(55., 8000., 96), 48000., B, 16384);
    let mut rng = 17u32;
    let pcm: Vec<[f32; B]> = (0..32)
        .map(|_| {
            std::array::from_fn(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                ((rng as i32) as f32 + 0.5) / 2147483648.0
            })
        })
        .collect();
    let mut y = [0.; B];
    for i in 0..LENGTH.div_ceil(B) {
        let x = &pcm[i % pcm.len()];
        run(&mut one, x, &mut y);
        run(&mut nine, x, &mut y);
        frontend.process(x, &params);
    }
    let mut times: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(REPS));
    for i in 0..REPS {
        let x = &pcm[i % pcm.len()];
        let order = if i % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        };
        for arm in order {
            begin_allocations();
            let start = Instant::now();
            match arm {
                0 => {
                    black_box(black_box(x));
                }
                1 => run(&mut one, x, &mut y),
                2 => run(&mut nine, x, &mut y),
                _ => {
                    black_box(frontend.process(black_box(x), &params));
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            let a = finish_allocations().unwrap();
            assert_eq!((a.count, a.bytes), (0, 0));
            times[arm].push(elapsed);
        }
    }
    for (label, v) in ["off", "one", "nine", "ear"].into_iter().zip(&mut times) {
        summary(label, v);
    }
    let mut edge: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(32));
    let silence = [0.; B];
    for i in 0..32 {
        let x = &pcm[i];
        let start = Instant::now();
        for stream in &mut nine {
            stream.reset();
        }
        run(&mut nine, x, &mut y);
        edge[0].push(start.elapsed().as_secs_f64() * 1e6);
        for _ in 1..LENGTH.div_ceil(B) {
            run(&mut nine, &silence, &mut y);
        }
        let start = Instant::now();
        run(&mut nine, &silence, &mut y);
        edge[1].push(start.elapsed().as_secs_f64() * 1e6);
        let start = Instant::now();
        run(&mut nine, x, &mut y);
        edge[2].push(start.elapsed().as_secs_f64() * 1e6);
    }
    for (label, v) in ["birth9", "expire9", "restart9"].into_iter().zip(&mut edge) {
        summary(label, v);
    }
}
