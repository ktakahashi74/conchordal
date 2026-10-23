#[test]
fn parseval_resource_power_units_support_boundaries_and_amplitude() {
    let (mut analyzer, _) = crate::runtime::body_metabolism_test_core();
    analyzer.enable_resource_power();
    for state in &mut analyzer.bands_state {
        state.alpha = 0.0;
    }
    let n = analyzer.nfft;
    let fs = analyzer.fs;
    let weights = analyzer.resource_weights.get().unwrap();
    let mut support = vec![false; n / 2 + 1];
    for row in weights {
        for &(k, w) in row {
            support[k] |= w > 0.0;
        }
    }
    let tones = [
        vec![(440.0, 0.03, 0.0)],
        vec![(440.0, 0.03, 0.0), (445.0, 0.02, 1.0)],
        vec![(440.0, 0.03, 0.0), (440.0, 0.03, 0.7)],
        vec![(440.0, 0.03, 0.0), (440.0, 0.03, std::f32::consts::PI)],
        vec![(fs * 0.49, 0.03, 0.0)],
        vec![(0.0, 0.03, 1.57)],
        vec![(fs * 0.5, 0.03, 1.57)],
        vec![(analyzer.space().fmin, 0.03, 0.2)],
        vec![(analyzer.space().fmax, 0.03, 0.2)],
        vec![(
            440.0 * 2_f32.powf(0.5 / analyzer.space().bins_per_oct as f32),
            0.03,
            0.2,
        )],
    ];
    let mut cases: Vec<Vec<f32>> = tones
        .iter()
        .map(|tones| {
            (0..n)
                .map(|i| {
                    tones
                        .iter()
                        .map(|&(f, a, phase)| {
                            a * (std::f32::consts::TAU * f * i as f32 / fs + phase).sin()
                        })
                        .sum()
                })
                .collect()
        })
        .collect();
    // A deterministic broadband sequence exercises every supported FFT bin.
    cases.push(
        (0..n)
            .map(|i| {
                ((i as u32).wrapping_mul(2654435761) as f64 / u32::MAX as f64 - 0.5) as f32 * 0.1
            })
            .collect(),
    );
    for (case, pcm) in cases.iter().enumerate() {
        let mut fft: Vec<_> = pcm
            .iter()
            .map(|&x| rustfft::num_complex::Complex64::new(x as f64, 0.0))
            .collect();
        rustfft::FftPlanner::<f64>::new()
            .plan_fft_forward(n)
            .process(&mut fft);
        let supported: f64 = (0..=n / 2)
            .filter(|&k| support[k])
            .map(|k| {
                fft[k].norm_sqr() * if k == 0 || k == n / 2 { 1.0 } else { 2.0 } / (n * n) as f64
            })
            .sum();
        analyzer.reset();
        for hop in pcm.chunks(analyzer.hop) {
            analyzer.process_resource(hop);
        }
        let actual: f64 = analyzer
            .resource_power()
            .unwrap()
            .iter()
            .map(|&x| x as f64)
            .sum();
        // f32 FFT and sparse sums are compared with a separate f64 Parseval oracle.
        let roundoff =
            128.0 * f32::EPSILON as f64 * pcm.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
                / n as f64;
        assert!(
            (actual - supported).abs() <= roundoff,
            "case {case}: {actual} vs {supported}"
        );
        let original = analyzer.resource_power().unwrap().to_vec();
        analyzer.reset();
        let doubled: Vec<_> = pcm.iter().map(|x| 2.0 * x).collect();
        for hop in doubled.chunks(analyzer.hop) {
            analyzer.process_resource(hop);
        }
        for (&a, b) in analyzer.resource_power().unwrap().iter().zip(original) {
            assert_eq!(a, 4.0 * b, "amplitude scaling case {case}");
        }
    }
    // FFT-bin support is a partition, including DC and Nyquist when supported.
    for k in 0..=n / 2 {
        let sum: f64 = analyzer
            .resource_weights
            .get()
            .unwrap()
            .iter()
            .flat_map(|r| r.iter())
            .filter(|(j, _)| *j == k)
            .map(|(_, w)| *w as f64)
            .sum();
        assert!((sum - if support[k] { 1.0 } else { 0.0 }).abs() <= 8.0 * f32::EPSILON as f64);
    }
}

#[test]
fn resource_history_clone_reset_and_warmup() {
    let (mut mixed, _) = crate::runtime::body_metabolism_test_core();
    mixed.enable_resource_power();
    let pcm = vec![0.03; mixed.hop];
    mixed.process_hop(&pcm);
    assert!(mixed.resource_power().is_none());
    for _ in 0..mixed.nfft.div_ceil(mixed.hop) {
        mixed.process_hop(&pcm);
    }
    let mut observer = mixed.clone();
    mixed.process_hop(&pcm);
    observer.process_resource(&pcm);
    assert_eq!(mixed.resource_power(), observer.resource_power());
    observer.reset();
    assert!(observer.resource_power().is_none());
    for i in 1..=observer.nfft.div_ceil(observer.hop) {
        observer.process_resource(&pcm);
        assert_eq!(
            observer.resource_power().is_some(),
            i * observer.hop >= observer.nfft
        );
    }
}
