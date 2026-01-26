use std::error::Error;
use std::fs::{create_dir_all, write};
use std::path::Path;

use plotters::prelude::*;

use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{LandscapeParams, RoughnessScalarMode};
use conchordal::core::log2space::Log2Space;
use conchordal::core::psycho_state;
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel};
use conchordal::paper::sim::{E4_ANCHOR_HZ, E4_WINDOW_CENTS, interval_metrics, run_e4_condition};

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = Path::new("target/plots/paper");
    create_dir_all(out_dir)?;

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(20.0, 8000.0, 200);

    plot_e1_landscape_scan(out_dir, &space, anchor_hz)?;
    plot_e4_mirror_sweep(out_dir, anchor_hz)?;

    println!("Saved paper plots to {}", out_dir.display());
    Ok(())
}

fn plot_e1_landscape_scan(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let anchor_idx = nearest_bin(space, anchor_hz);

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let mut anchor_density_scan = vec![0.0f32; space.n_bins()];
    let denom = du_scan[anchor_idx].max(1e-12);
    anchor_density_scan[anchor_idx] = 1.0 / denom;

    let mut anchor_env_scan = vec![0.0f32; space.n_bins()];
    anchor_env_scan[anchor_idx] = 1.0;

    space.assert_scan_len_named(&anchor_density_scan, "anchor_density_scan");
    space.assert_scan_len_named(&anchor_env_scan, "anchor_env_scan");

    let roughness_kernel = RoughnessKernel::new(KernelParams::default(), 0.005);
    let harmonicity_kernel = HarmonicityKernel::new(space, HarmonicityParams::default());

    let (perc_h_pot_scan, _) =
        harmonicity_kernel.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_r_pot_scan, _) =
        roughness_kernel.potential_r_from_log2_spectrum_density(&anchor_density_scan, space);

    space.assert_scan_len_named(&perc_h_pot_scan, "perc_h_pot_scan");
    space.assert_scan_len_named(&perc_r_pot_scan, "perc_r_pot_scan");

    let params = LandscapeParams {
        fs: 48_000.0,
        max_hist_cols: 1,
        alpha: 0.0,
        roughness_kernel: roughness_kernel.clone(),
        harmonicity_kernel: harmonicity_kernel.clone(),
        roughness_scalar_mode: RoughnessScalarMode::Total,
        roughness_half: 0.1,
        consonance_roughness_weight: 0.5,
        loudness_exp: 1.0,
        ref_power: 1.0,
        tau_ms: 1.0,
        roughness_k: 1.0,
        roughness_ref_f0_hz: 1000.0,
        roughness_ref_sep_erb: 0.25,
        roughness_ref_mass_split: 0.5,
        roughness_ref_eps: 1e-12,
    };

    let r_ref = psycho_state::compute_roughness_reference(&params, space);
    let mut perc_r_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &perc_r_pot_scan,
        r_ref.peak,
        params.roughness_k,
        &mut perc_r_state01_scan,
    );

    let h_ref_max = perc_h_pot_scan
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let mut perc_h_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::h_pot_scan_to_h_state01_scan(
        &perc_h_pot_scan,
        h_ref_max,
        &mut perc_h_state01_scan,
    );

    let mut perc_c_state01_scan = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let (_c_signed, c01) = psycho_state::compose_c_statepm1(
            perc_h_state01_scan[i],
            perc_r_state01_scan[i],
            params.consonance_roughness_weight,
        );
        perc_c_state01_scan[i] = c01;
    }

    let anchor_log2 = anchor_hz.log2();
    let log2_ratio_scan: Vec<f32> = space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect();

    space.assert_scan_len_named(&perc_r_state01_scan, "perc_r_state01_scan");
    space.assert_scan_len_named(&perc_h_state01_scan, "perc_h_state01_scan");
    space.assert_scan_len_named(&perc_c_state01_scan, "perc_c_state01_scan");
    space.assert_scan_len_named(&log2_ratio_scan, "log2_ratio_scan");

    let out_path = out_dir.join("paper_e1_landscape_scan_anchor220.png");
    render_e1_plot(
        &out_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan,
        &perc_r_state01_scan,
        &perc_c_state01_scan,
    )?;

    Ok(())
}

fn render_e1_plot(
    out_path: &Path,
    anchor_hz: f32,
    log2_ratio_scan: &[f32],
    perc_h_pot_scan: &[f32],
    perc_r_state01_scan: &[f32],
    perc_c_state01_scan: &[f32],
) -> Result<(), Box<dyn Error>> {
    let x_min = -1.0f32;
    let x_max = 1.0f32;

    let mut h_points = Vec::new();
    let mut r_points = Vec::new();
    let mut c_points = Vec::new();
    for i in 0..log2_ratio_scan.len() {
        let x = log2_ratio_scan[i];
        if x < x_min || x > x_max {
            continue;
        }
        h_points.push((x, perc_h_pot_scan[i]));
        r_points.push((x, perc_r_state01_scan[i]));
        c_points.push((x, perc_c_state01_scan[i]));
    }

    let h_max = h_points
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f32, f32::max)
        .max(1e-6)
        * 1.1;

    let root = BitMapBackend::new(out_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((3, 1));

    let ratio_guides = [0.5f32, 6.0 / 5.0, 1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0];
    let ratio_guides_log2: Vec<f32> = ratio_guides.iter().map(|r| r.log2()).collect();

    let mut chart_h = ChartBuilder::on(&panels[0])
        .caption(
            format!("E1 Harmonicity Potential H(f) | anchor {} Hz", anchor_hz),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..h_max)?;

    chart_h
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("H potential")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_h.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, h_max)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart_h.draw_series(LineSeries::new(h_points, &BLUE))?;

    let mut chart_r = ChartBuilder::on(&panels[1])
        .caption("E1 Roughness State R01(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..1.05f32)?;

    chart_r
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("R01")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_r.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, 1.05)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart_r.draw_series(LineSeries::new(r_points, &RED))?;

    let mut chart_c = ChartBuilder::on(&panels[2])
        .caption("E1 Consonance State C01(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..1.05f32)?;

    chart_c
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("C01")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_c.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, 1.05)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart_c.draw_series(LineSeries::new(c_points, &GREEN))?;

    root.present()?;
    Ok(())
}

fn plot_e4_mirror_sweep(out_dir: &Path, anchor_hz: f32) -> Result<(), Box<dyn Error>> {
    let seed = 7;
    let mut rows: Vec<(f32, f32, f32, f32, usize)> = Vec::new();
    for i in 0..=10 {
        let mirror_weight = i as f32 / 10.0;
        let voice_freqs = run_e4_condition(mirror_weight, seed);
        let metrics = interval_metrics(anchor_hz, &voice_freqs, E4_WINDOW_CENTS);
        rows.push((
            mirror_weight,
            metrics.mass_maj3,
            metrics.mass_min3,
            metrics.mass_p5,
            metrics.n_voices,
        ));
    }

    let mut csv = String::from("mirror_weight,mass_M3,mass_m3,mass_P5,n_voices,window_cents\n");
    for (w, mass_maj3, mass_min3, mass_p5, n_voices) in &rows {
        csv.push_str(&format!(
            "{w:.3},{mass_maj3:.3},{mass_min3:.3},{mass_p5:.3},{n_voices},{}\n",
            E4_WINDOW_CENTS
        ));
    }

    let csv_path = out_dir.join("paper_e4_mirror_sweep_metrics.csv");
    write(&csv_path, csv)?;

    let mut y_max = rows
        .iter()
        .map(|(_, m3, m3min, _p5, _)| (*m3).max(*m3min))
        .fold(0.0f32, f32::max);
    if y_max <= 0.0 {
        y_max = 1.0;
    }
    let y_hi = y_max * 1.1;

    let plot_path = out_dir.join("paper_e4_mirror_sweep.png");
    let root = BitMapBackend::new(&plot_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E4 Mirror Weight Sweep (Interval Mass)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("mirror_weight")
        .y_desc("interval mass (count)")
        .draw()?;

    let maj_points: Vec<(f32, f32)> = rows.iter().map(|(w, m3, _, _, _)| (*w, *m3)).collect();
    let min_points: Vec<(f32, f32)> = rows.iter().map(|(w, _, m3, _, _)| (*w, *m3)).collect();

    chart
        .draw_series(LineSeries::new(maj_points, &BLUE))?
        .label("mass_M3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(min_points, &RED))?
        .label("mass_m3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn erb_grid_for_space(space: &Log2Space) -> (Vec<f32>, Vec<f32>) {
    let erb_scan: Vec<f32> = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();
    let du_scan = local_du_from_grid(&erb_scan);
    (erb_scan, du_scan)
}

fn local_du_from_grid(grid: &[f32]) -> Vec<f32> {
    if grid.is_empty() {
        return Vec::new();
    }
    if grid.len() == 1 {
        return vec![1.0];
    }

    let n = grid.len();
    let mut du = vec![0.0f32; n];
    du[0] = (grid[1] - grid[0]).max(0.0);
    du[n - 1] = (grid[n - 1] - grid[n - 2]).max(0.0);
    for i in 1..n - 1 {
        du[i] = (0.5 * (grid[i + 1] - grid[i - 1])).max(0.0);
    }
    du
}

fn nearest_bin(space: &Log2Space, hz: f32) -> usize {
    let mut best_idx = 0;
    let mut best_diff = f32::MAX;
    for (i, &f) in space.centers_hz.iter().enumerate() {
        let diff = (f - hz).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }
    best_idx
}
