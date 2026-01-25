use rand::{Rng, SeedableRng};

/// Calculate A-weighting gain (linear) for a given frequency.
/// Standard curve approximation normalized to 1.0 at 1000 Hz.
pub fn a_weighting_gain(f_hz: f32) -> f32 {
    let f2 = f_hz * f_hz;
    let c1 = 12194.0 * 12194.0;
    let c2 = 20.6 * 20.6;
    let c3 = 107.7 * 107.7;
    let c4 = 737.9 * 737.9;

    let num = c1 * f2 * f2;
    let den = (f2 + c2) * (f2 + c3).sqrt() * (f2 + c4).sqrt() * (f2 + c1);

    let ra = num / den;
    // Normalization: A-weighting at 1kHz is usually 0dB (gain=1.0).
    // The standard formula gives ~0.794 at 1kHz. We normalize so 1kHz = 1.0.
    ra * 1.2589
}

/// Single-sample pink noise (Paul Kellet 3-pole filter).
pub fn pink_noise_tick<R: Rng + ?Sized>(
    rng: &mut R,
    b0: &mut f32,
    b1: &mut f32,
    b2: &mut f32,
) -> f32 {
    let white = rng.random_range(-1.0..1.0);
    *b0 = 0.99765 * *b0 + white * 0.099_046_0;
    *b1 = 0.96300 * *b1 + white * 0.296_516_4;
    *b2 = 0.57000 * *b2 + white * 1.052_691_3;
    let pink = *b0 + *b1 + *b2 + white * 0.1848;
    pink * 0.03
}

// --- noise generators ---
pub fn white_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Pink noise via simple 3-pole filter approximation (Voss–McCartney not required)
pub fn pink_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(pink_noise_tick(&mut rng, &mut b0, &mut b1, &mut b2));
    }
    out
}

/// Brown (red) noise via single-pole low-pass filter (−6 dB/oct)
pub fn brown_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut y = 0.0f32;
    let mut out = Vec::with_capacity(n);

    // Simple 1-pole IIR with pole near DC
    // y[n] = a*y[n-1] + (1 - a)*x[n]
    // choose a close to 1.0 to get strong low-frequency emphasis
    let a = 0.995; // ≈ −6 dB/oct

    for _ in 0..n {
        let white = rng.random_range(-1.0..1.0);
        y = a * y + (1.0 - a) * white;
        out.push(y);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::db;

    fn ensure_plots_dir() -> std::io::Result<()> {
        std::fs::create_dir_all("target/plots")
    }

    #[test]
    fn test_a_weighting_reference_values() {
        let cases = [
            (1000.0, 0.0),   // 0 dB
            (100.0, -19.1),  // approx -19.1 dB
            (20.0, -50.5),   // approx -50.5 dB
            (20000.0, -9.3), // approx -9.3 dB
        ];

        for (f, expected_db) in cases {
            let gain = a_weighting_gain(f);
            let db = db::amp_to_db(gain);
            assert!(
                (db - expected_db).abs() < 0.5,
                "A-weighting mismatch at {} Hz: expected ~{} dB, got {:.2} dB",
                f,
                expected_db,
                db
            );
        }
    }

    #[test]
    #[ignore]
    fn plot_a_weighting_curve() -> Result<(), Box<dyn std::error::Error>> {
        use plotters::prelude::*;
        use std::path::Path;

        ensure_plots_dir()?;
        let path = Path::new("target/plots/it_a_weighting_curve.png");
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("A-Weighting Curve (IEC 61672-1)", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d((20.0f32..24000.0f32).log_scale(), -80.0f32..10.0f32)?;

        chart
            .configure_mesh()
            .x_desc("Frequency [Hz]")
            .y_desc("Gain [dB]")
            .x_labels(10)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.0}", v))
            .draw()?;

        // Plot curve (log-spaced points for smoothness)
        let n_points = 500;
        let log_min = (20.0f32).ln();
        let log_max = (24000.0f32).ln();
        let points: Vec<(f32, f32)> = (0..=n_points)
            .map(|i| {
                let t = i as f32 / n_points as f32;
                let f = (log_min + t * (log_max - log_min)).exp();
                let g = a_weighting_gain(f);
                let db = db::amp_to_db(g);
                (f, db)
            })
            .collect();

        chart
            .draw_series(LineSeries::new(points, &BLUE))?
            .label("A-Weighting")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        println!("Saved plot to {:?}", path);
        Ok(())
    }
}
