//! Bounded release measurements; never an acceptance benchmark.
use super::*;
use std::{hint::black_box, time::Instant};

fn statistics(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"samples":samples.len(),"mean_us":samples.iter().sum::<f64>()/samples.len() as f64,
        "p99_us":sorted[((samples.len() as f64*0.99).ceil() as usize-1).min(samples.len()-1)]})
}

fn host_snapshot() -> Value {
    let ps = std::process::Command::new("ps")
        .args(["-eo", "pid,ppid,pcpu,rss,comm", "--sort=-pcpu"])
        .output()
        .unwrap();
    json!({"time":String::from_utf8(std::process::Command::new("date").arg("-Iseconds").output().unwrap().stdout).unwrap().trim(),
        "loadavg":std::fs::read_to_string("/proc/loadavg").unwrap().trim(),
        "processes":String::from_utf8(ps.stdout).unwrap().lines().take(13).collect::<Vec<_>>()})
}

#[test]
#[ignore = "AUD-M1 release-only frontend, kernel, and LOO cost diagnostic"]
fn aud_m1_v5_cost() {
    assert!(
        !black_box(cfg!(debug_assertions)),
        "run the registered release build"
    );
    let dir = output_dir();
    let before = host_snapshot();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let mut old = SynchronousAnalysis::new(kernel.clone(), &params);
    let mut new = ShadowFrontEnd::new(space.clone(), params.fs, 0.25, false, kernel.nfft())
        .with_statistics(hop, true);
    let mut scratch = [
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel),
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel),
    ];
    let mut h = [vec![0.0; space.n_bins()], vec![0.0; space.n_bins()]];
    let mut r = h.clone();
    let mut pcm = vec![0.0; hop];
    let mut frontend = [Vec::with_capacity(256), Vec::with_capacity(256)];
    let mut kernels = [Vec::with_capacity(256), Vec::with_capacity(256)];
    for frame in 0..320 {
        for (i, x) in pcm.iter_mut().enumerate() {
            let t = (frame * hop + i) as f32 / params.fs;
            *x = 0.1 * (TAU * 110.0 * t).cos()
                + 0.05 * (TAU * 150.0 * t + 0.7).cos()
                + 0.03 * (TAU * 1003.1 * t).sin();
        }
        for offset in 0..2 {
            let which = (frame + offset) % 2;
            let start = Instant::now();
            let density = if which == 0 {
                old.density(black_box(&pcm), &params).subjective_intensity
            } else {
                new.process(black_box(&pcm), &params);
                &new.density_scan
            };
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            let start = Instant::now();
            black_box(scratch[which].compute_shape(
                black_box(density),
                &space,
                (&params.harmonicity_kernel, &params.roughness_kernel),
                params.roughness_ref_eps,
                &mut h[which],
                &mut r[which],
            ));
            let hrc = start.elapsed().as_secs_f64() * 1e6;
            if frame >= 64 {
                frontend[which].push(elapsed);
                kernels[which].push(hrc);
            }
        }
    }
    let mut old_template =
        crate::core::stream::analysis::AnalysisStream::new(params.clone(), kernel.clone());
    let mut new_template = with_selected_test_frontend(|| {
        crate::core::stream::analysis::AnalysisStream::new(params.clone(), kernel)
    });
    for _ in 0..64 {
        old_template.process(&pcm);
        new_template.process(&pcm);
    }
    let mut loo = Vec::new();
    for count in [8, 16, 32] {
        let mut observers = [
            (0..count)
                .map(|i| crate::core::stream::loo::Observer::new(i as u64, 0, &old_template))
                .collect::<Vec<_>>(),
            (0..count)
                .map(|i| crate::core::stream::loo::Observer::new(i as u64, 0, &new_template))
                .collect::<Vec<_>>(),
        ];
        let mut own = vec![vec![0.0; hop]; count];
        let mut timings = [Vec::with_capacity(64), Vec::with_capacity(64)];
        for frame in 0..80 {
            for (i, x) in pcm.iter_mut().enumerate() {
                let t = (frame * hop + i) as f32 / params.fs;
                *x = 0.1 * (TAU * 110.0 * t).cos() + 0.05 * (TAU * 150.0 * t + 0.7).cos();
                for (voice, signal) in own.iter_mut().enumerate() {
                    signal[i] = 0.02 * (TAU * (220.0 + 17.0 * voice as f32) * t + 0.3).sin();
                    *x += signal[i];
                }
            }
            for offset in 0..2 {
                let which = (frame + offset) % 2;
                let start = Instant::now();
                for (observer, signal) in observers[which].iter_mut().zip(&own) {
                    observer.process(black_box(&pcm), black_box(signal));
                    black_box(observer.snapshot());
                }
                if frame >= 16 {
                    timings[which].push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        loo.push(json!({"voices":count,"current":statistics(&timings[0]),"selected_adapter":statistics(&timings[1])}));
    }
    let after = host_snapshot();
    std::fs::write(dir.join("cost.json"),serde_json::to_vec_pretty(&json!({"profile":"cargo test --release; opt-level=3, LTO=true; debug assertions disabled",
        "fs":params.fs,"hop":hop,"hop_budget_us":hop as f64/f64::from(params.fs)*1e6,
        "frontend_order_alternates":true,"frontend_current":statistics(&frontend[0]),"frontend_selected":statistics(&frontend[1]),
        "hr_kernel_current":statistics(&kernels[0]),"hr_kernel_selected":statistics(&kernels[1]),"loo":loo,
        "scope":"frontend includes S1-S6 or NSGT+SpectralFrontEnd; H/R compute_shape is measured separately; LOO is Observer::process plus snapshot creation and destruction",
        "adapter_note":"selected LOO streams retain NSGT power alongside the shadow frontend; totals include both. No listener, T2, transport queue, GUI, or resource-power tracker",
        "timing_lock":false,"host_before":before,"host_after":after})).unwrap()).unwrap();
}
