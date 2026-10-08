use super::*;

#[test]
fn native_grid_matches_registered_ltfat_readout() {
    let centers = centers_hz();
    assert_eq!(centers.len(), 30);
    assert!(centers.windows(2).all(|f| f[0] < f[1]));
    assert_eq!((centers[0] * 1000.0).round() / 1000.0, 107.652);
    assert_eq!((centers[29] * 1000.0).round() / 1000.0, 7469.673);
    assert_eq!(centers[12], 1000.0000000000001);
    assert!(Producer::new(8000).is_err());
}

fn fixture(kind: usize, n: usize) -> f64 {
    let tone = |amplitude| amplitude * (2.0 * PI * 1000.0 * n as f64 / 48000.0).sin();
    match kind {
        0 => {
            if n == 0 {
                0.001
            } else {
                0.0
            }
        }
        1 => {
            if n < 4096 {
                tone(0.001)
            } else {
                0.0
            }
        }
        2 => {
            (if n < 4800 { tone(0.001) } else { 0.0 })
                + if (2400..2880).contains(&n) {
                    tone(0.0001)
                } else {
                    0.0
                }
        }
        _ => {
            (if n < 4800 { tone(0.001) } else { 0.0 })
                + if (6240..6720).contains(&n) {
                    tone(0.0001)
                } else {
                    0.0
                }
        }
    }
}

fn product(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

#[test]
fn recovered_same_input_adaptation_and_modulation_margins() {
    use std::io::Write;
    let evidence = std::env::var_os("T2_REFERENCE_DIRECTORY").map(std::path::PathBuf::from);
    if let Some(path) = &evidence {
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(
            path.join("centers.json"),
            serde_json::to_vec(&centers_hz()).unwrap(),
        )
        .unwrap();
    }
    let epsilon = 2.0_f64.powi(-52);
    let gamma32 = 32.0 * epsilon / (1.0 - 32.0 * epsilon);
    let gamma64 = 64.0 * epsilon / (1.0 - 64.0 * epsilon);
    for (band, center) in centers_hz().into_iter().enumerate() {
        for kind in 0..4 {
            let mut trace = evidence.as_ref().map(|path| {
                std::io::BufWriter::new(
                    std::fs::File::create(path.join(format!("band-{band:02}-fixture-{kind}.f64")))
                        .unwrap(),
                )
            });
            let mut native = Channel::new(48000.0, center);
            let mut q: [f64; 5] =
                std::array::from_fn(|j| 1e-5_f64.powf(2.0_f64.powi(-(j as i32 + 1))));
            let amplitude = q.map(|v| (1.0 - v * v) * 10.0 - 1.0);
            let corr = q[4];
            let a =
                [0.005, 0.050, 0.129, 0.253, 0.500].map(|tau| (-1.0_f64 / (tau * 48000.0)).exp());
            let mut z = [Complex::default(); 3];
            let frequencies = [5.0, 10.0, 50.0 / 3.0];
            let radius = [5.0, 5.0, (50.0 / 3.0) / 2.0].map(|bw| (-PI * bw / 48000.0).exp());
            let k = (PI * 2.5 / 48000.0).tan();
            let real = -k / 2.0_f64.sqrt();
            let imag = k / 2.0_f64.sqrt();
            let den = (1.0 - real) * (1.0 - real) + imag * imag;
            let pole = Complex {
                re: ((1.0 + real) * (1.0 - real) - imag * imag) / den,
                im: 2.0 * imag / den,
            };
            let magnitude = pole.re.hypot(pole.im);
            let mut lp = [Complex::default(); 4];
            let mut maximum: f64 = 0.0;
            let samples = if kind == 0 { 4096 } else { 8192 };
            for n in 0..samples {
                let (output, ihc, mu) = native.one(fixture(kind, n));
                if let Some(trace) = &mut trace {
                    for value in [2.0 * native.gt[3].re, ihc, mu]
                        .into_iter()
                        .chain(output)
                        .chain(native.q)
                    {
                        trace.write_all(&value.to_le_bytes()).unwrap();
                    }
                }
                let mut reference = ihc.max(1e-5);
                for j in 0..5 {
                    let u = reference / q[j];
                    reference = if u <= 1.0 {
                        u
                    } else {
                        1.0 + amplitude[j] * ((u - 1.0) / amplitude[j]).tanh()
                    };
                    q[j] = a[j] * q[j] + (1.0 - a[j]) * reference;
                    assert!((native.q[j] - q[j]).abs() <= 1e-12 + 1e-11 * q[j].abs());
                }
                let mu_reference = (reference - corr) * 100.0 / (1.0 - corr);
                assert!((mu - mu_reference).abs() <= 1e-9 + 1e-11 * mu_reference.abs());
                maximum = maximum.max(mu.abs());
                // Independent AMT-form LP sections use exactly the native MU input.
                let mut v = Complex { re: mu, im: 0.0 };
                for (section, p) in [
                    pole,
                    Complex {
                        re: pole.re,
                        im: -pole.im,
                    },
                ]
                .into_iter()
                .enumerate()
                {
                    let sum = Complex {
                        re: v.re + lp[2 * section].re,
                        im: v.im + lp[2 * section].im,
                    };
                    let direct = product(
                        Complex {
                            re: (1.0 - p.re) / 2.0,
                            im: -p.im / 2.0,
                        },
                        sum,
                    );
                    let feedback = product(p, lp[2 * section + 1]);
                    let y = Complex {
                        re: direct.re + feedback.re,
                        im: direct.im + feedback.im,
                    };
                    lp[2 * section] = v;
                    lp[2 * section + 1] = y;
                    v = y;
                }
                let bound =
                    8.0 * gamma64 * maximum * ((1.0 + magnitude) / (1.0 - magnitude)).powi(2);
                assert!((output[0] - v.re).abs() <= bound);
                for j in 0..3 {
                    let w = 2.0 * PI * frequencies[j] / 48000.0;
                    let previous = z[j];
                    z[j] = Complex {
                        re: radius[j] * (w.cos() * previous.re - w.sin() * previous.im)
                            + (1.0 - radius[j]) * mu,
                        im: radius[j] * (w.sin() * previous.re + w.cos() * previous.im),
                    };
                    let expected = if j < 2 {
                        2.0 * z[j].re
                    } else {
                        2.0 * z[j].re.hypot(z[j].im)
                    };
                    let bound = 2.0 * 2.0_f64.sqrt() * 8.0 * gamma32 * maximum / (1.0 - radius[j]);
                    assert!((output[j + 1] - expected).abs() <= bound);
                }
            }
            println!("T2_LOCAL_REFERENCE center={center:.17} fixture={kind} samples={samples}");
        }
    }
}

#[test]
fn producer_hop_partition_and_cold_gap_reset_preserve_every_filter_state() {
    let pcm: Vec<f32> = (0..2048).map(|n| fixture(2, n) as f32).collect();
    let mut whole = Producer::new(48000).unwrap();
    let mut split = Producer::new(48000).unwrap();
    whole.process(0, &pcm).unwrap();
    for (start, end) in [
        (0, 1),
        (1, 17),
        (17, 511),
        (511, 512),
        (512, 513),
        (513, 1024),
        (1024, 2048),
    ] {
        split.process(start as u64, &pcm[start..end]).unwrap();
    }
    for (a, b) in whole.channels.iter().zip(&split.channels) {
        assert_eq!(a.q, b.q);
        assert_eq!(a.ihc, b.ihc);
        assert_eq!(a.lp, b.lp);
        for (x, y) in a.gt.iter().chain(&a.z).zip(b.gt.iter().chain(&b.z)) {
            assert_eq!(x.re.to_bits(), y.re.to_bits());
            assert_eq!(x.im.to_bits(), y.im.to_bits());
        }
    }
    assert!(split.process(4096, &pcm[..512]).is_err());
    split.reset(4096);
    let restarted = split.process(4096, &pcm[..512]).unwrap();
    let fresh = Producer::new(48000)
        .unwrap()
        .process(0, &pcm[..512])
        .unwrap();
    assert_eq!(restarted.means, fresh.means);
    assert_eq!(restarted.epoch_start, 4096);
}
