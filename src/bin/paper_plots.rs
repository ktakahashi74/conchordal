use std::error::Error;
use std::f32::consts::PI;
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
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = Path::new("target/plots/paper");
    create_dir_all(out_dir)?;

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(20.0, 8000.0, 200);

    plot_e1_landscape_scan(out_dir, &space, anchor_hz)?;
    plot_e2_emergent_harmony(out_dir, &space, anchor_hz)?;
    plot_e3_metabolic_selection(out_dir, &space, anchor_hz)?;
    plot_e4_mirror_sweep(out_dir, anchor_hz)?;
    plot_e5_rhythmic_entrainment(out_dir)?;

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
        consonance_harmonicity_deficit_weight: 1.0,
        consonance_roughness_weight_floor: 0.35,
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

    let mut perc_c_raw_scan = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let h01 = perc_h_state01_scan[i];
        let r01 = perc_r_state01_scan[i];
        let dh = 1.0 - h01;
        let w = params.consonance_roughness_weight_floor + params.consonance_roughness_weight * dh;
        let d = params.consonance_harmonicity_deficit_weight * dh + w * r01;
        let c_raw = 1.0 / (1.0 + d);
        perc_c_raw_scan[i] = if c_raw.is_finite() { c_raw } else { 0.0 };
    }

    let anchor_log2 = anchor_hz.log2();
    let log2_ratio_scan: Vec<f32> = space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect();

    space.assert_scan_len_named(&perc_r_state01_scan, "perc_r_state01_scan");
    space.assert_scan_len_named(&perc_h_state01_scan, "perc_h_state01_scan");
    space.assert_scan_len_named(&perc_c_raw_scan, "perc_c_raw_scan");
    space.assert_scan_len_named(&log2_ratio_scan, "log2_ratio_scan");

    let out_path = out_dir.join("paper_e1_landscape_scan_anchor220.png");
    render_e1_plot(
        &out_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan,
        &perc_r_state01_scan,
        &perc_c_raw_scan,
    )?;

    Ok(())
}

fn plot_e2_emergent_harmony(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let mut rng = seeded_rng(0xC0FFEE);
    let anchor_idx = nearest_bin(space, anchor_hz);
    let log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz);
    let (min_idx, max_idx) = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);
    let allowed_indices: Vec<usize> = (min_idx..=max_idx).collect();
    let n_agents = 24;

    let mut agent_indices: Vec<usize> = (0..n_agents)
        .map(|_| {
            let pick = rng.random_range(0..allowed_indices.len());
            allowed_indices[pick]
        })
        .collect();

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let workspace = build_c01_workspace(space);

    let steps = 400;
    let burn_in = 100usize;
    let mut semitone_samples: Vec<f32> = Vec::new();
    let mut mean_c01_series: Vec<(f32, f32)> = Vec::with_capacity(steps);

    for step in 0..steps {
        let (env_scan, density_scan) = build_env_scans(space, anchor_idx, &agent_indices, &du_scan);
        let c01_scan = compute_c01_scan(space, &workspace, &env_scan, &density_scan);
        let mean_c01 = mean_at_indices(&c01_scan, &agent_indices);
        mean_c01_series.push((step as f32, mean_c01));

        update_agent_indices(
            &mut agent_indices,
            &c01_scan,
            &log2_ratio_scan,
            min_idx,
            max_idx,
            2,
            0.15,
            0.06,
        );

        if step >= burn_in {
            semitone_samples.extend(agent_indices.iter().map(|&idx| 12.0 * log2_ratio_scan[idx]));
        }
    }

    let mut csv_ts = String::from("t,mean_c01\n");
    for (t, mean) in &mean_c01_series {
        csv_ts.push_str(&format!("{t:.0},{mean:.6}\n"));
    }
    write(out_dir.join("paper_e2_timeseries.csv"), csv_ts)?;

    let mut csv_agents = String::from("agent_id,freq_hz,log2_ratio,semitones\n");
    for (id, &idx) in agent_indices.iter().enumerate() {
        let freq_hz = space.centers_hz[idx];
        let log2_ratio = log2_ratio_scan[idx];
        let semitones = 12.0 * log2_ratio;
        csv_agents.push_str(&format!(
            "{id},{freq_hz:.6},{log2_ratio:.6},{semitones:.3}\n"
        ));
    }
    write(out_dir.join("paper_e2_final_agents.csv"), csv_agents)?;

    let mean_plot_path = out_dir.join("paper_e2_mean_consonance_over_time.png");
    render_mean_consonance_plot(&mean_plot_path, &mean_c01_series)?;

    let hist_path = out_dir.join("paper_e2_interval_histogram.png");
    render_interval_histogram(
        &hist_path,
        "E2 Interval Histogram (Semitones)",
        &semitone_samples,
        -12.0,
        12.0,
        0.5,
    )?;

    Ok(())
}

fn plot_e3_metabolic_selection(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let mut rng = seeded_rng(0xC0FFEE_u64 + 1);
    let anchor_idx = nearest_bin(space, anchor_hz);
    let log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz);
    let (min_idx, max_idx) = log2_ratio_bounds(&log2_ratio_scan, -1.0, 1.0);
    let allowed_indices: Vec<usize> = (min_idx..=max_idx).collect();

    let n_agents = 32;
    let mut agents: Vec<MetabolicAgent> = (0..n_agents)
        .map(|_| {
            let pick = rng.random_range(0..allowed_indices.len());
            MetabolicAgent {
                idx: allowed_indices[pick],
                energy: 1.0,
                age_steps: 0,
                sum_c01: 0.0,
            }
        })
        .collect();

    let (_erb_scan, du_scan) = erb_grid_for_space(space);
    let workspace = build_c01_workspace(space);

    let gain = 0.06f32;
    let decay = 0.04f32;
    let mut deaths: Vec<(usize, u32, f32)> = Vec::new();
    let mut death_id: usize = 0;

    let min_deaths = 200usize;
    let base_steps = 2000usize;
    let hard_cap = 6000usize;
    let mut step = 0usize;

    while step < hard_cap && (step < base_steps || deaths.len() < min_deaths) {
        let agent_indices: Vec<usize> = agents.iter().map(|a| a.idx).collect();
        let (env_scan, density_scan) = build_env_scans(space, anchor_idx, &agent_indices, &du_scan);
        let c01_scan = compute_c01_scan(space, &workspace, &env_scan, &density_scan);

        for agent in agents.iter_mut() {
            let c = c01_scan[agent.idx];
            agent.energy += gain * c - decay;
            agent.age_steps += 1;
            agent.sum_c01 += c;

            if agent.energy <= 0.0 {
                let avg_c01 = if agent.age_steps > 0 {
                    agent.sum_c01 / agent.age_steps as f32
                } else {
                    0.0
                };
                deaths.push((death_id, agent.age_steps, avg_c01));
                death_id += 1;

                let pick = rng.random_range(0..allowed_indices.len());
                agent.idx = allowed_indices[pick];
                agent.energy = 1.0;
                agent.age_steps = 0;
                agent.sum_c01 = 0.0;
            }
        }

        let mut updated_indices: Vec<usize> = agents.iter().map(|a| a.idx).collect();
        update_agent_indices(
            &mut updated_indices,
            &c01_scan,
            &log2_ratio_scan,
            min_idx,
            max_idx,
            2,
            0.15,
            0.06,
        );
        for (agent, idx) in agents.iter_mut().zip(updated_indices.into_iter()) {
            agent.idx = idx;
        }

        step += 1;
    }

    let mut csv = String::from("death_id,lifetime_steps,avg_c01\n");
    for (id, lifetime, avg_c01) in &deaths {
        csv.push_str(&format!("{id},{lifetime},{avg_c01:.6}\n"));
    }
    write(out_dir.join("paper_e3_lifetimes.csv"), csv)?;

    let scatter_path = out_dir.join("paper_e3_consonance_vs_lifetime.png");
    render_consonance_lifetime_scatter(&scatter_path, &deaths)?;

    let survival_path = out_dir.join("paper_e3_survival_curve.png");
    render_survival_curve(&survival_path, &deaths)?;

    Ok(())
}

fn plot_e5_rhythmic_entrainment(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut rng = seeded_rng(0xC0FFEE_u64 + 2);
    let n = 32usize;
    let dt = 0.02f32;
    let steps = 2000usize;
    let omega0 = 2.0 * PI * 2.0;
    let jitter = 0.02f32;
    let k_global = 1.2f32;
    let k_kick = 0.8f32;

    let mut thetas: Vec<f32> = (0..n)
        .map(|_| rng.random_range(0.0f32..(2.0 * PI)))
        .collect();
    let omegas: Vec<f32> = (0..n)
        .map(|_| {
            let jitter_scale = rng.random_range(-jitter..jitter);
            omega0 * (1.0 + jitter_scale)
        })
        .collect();

    let mut series: Vec<(f32, f32, f32)> = Vec::with_capacity(steps);
    let mut thetas_last: Option<Vec<f32>> = None;
    for step in 0..steps {
        let t = step as f32 * dt;
        let theta_kick = omega0 * t;

        let (mut mean_cos, mut mean_sin) = (0.0f32, 0.0f32);
        let (mut mean_dcos, mut mean_dsin) = (0.0f32, 0.0f32);
        for &theta in &thetas {
            mean_cos += theta.cos();
            mean_sin += theta.sin();

            let d = theta - theta_kick;
            mean_dcos += d.cos();
            mean_dsin += d.sin();
        }
        let inv_n = 1.0 / n as f32;
        mean_cos *= inv_n;
        mean_sin *= inv_n;
        mean_dcos *= inv_n;
        mean_dsin *= inv_n;

        let r = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();
        let plv = (mean_dcos * mean_dcos + mean_dsin * mean_dsin).sqrt();
        series.push((t, r, plv));

        if step + 1 == steps {
            thetas_last = Some(thetas.clone());
        }

        let mut next = vec![0.0f32; n];
        for i in 0..n {
            let theta_i = thetas[i];
            let mut coupling = 0.0f32;
            for &theta_j in &thetas {
                coupling += (theta_j - theta_i).sin();
            }
            let dtheta =
                omegas[i] + (k_global * inv_n) * coupling + k_kick * (theta_kick - theta_i).sin();
            next[i] = theta_i + dtheta * dt;
        }
        thetas = next;
    }

    let mut csv = String::from("t,r,plv\n");
    for (t, r, plv) in &series {
        csv.push_str(&format!("{t:.4},{r:.6},{plv:.6}\n"));
    }
    write(out_dir.join("paper_e5_order_plv.csv"), csv)?;

    let order_path = out_dir.join("paper_e5_order_plv_over_time.png");
    render_order_plv_plot(&order_path, &series)?;

    let final_t = (steps.saturating_sub(1)) as f32 * dt;
    let theta_kick_final = omega0 * final_t;
    let phase_source = thetas_last.as_deref().unwrap_or(&thetas);
    let phase_diffs: Vec<f32> = phase_source
        .iter()
        .map(|&theta| wrap_to_pi(theta - theta_kick_final))
        .collect();
    let phase_path = out_dir.join("paper_e5_phase_histogram.png");
    render_phase_histogram(&phase_path, &phase_diffs, 48)?;

    Ok(())
}

fn render_e1_plot(
    out_path: &Path,
    anchor_hz: f32,
    log2_ratio_scan: &[f32],
    perc_h_pot_scan: &[f32],
    perc_r_state01_scan: &[f32],
    perc_c_raw_scan: &[f32],
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
        c_points.push((x, perc_c_raw_scan[i]));
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

    let mut c_min = f32::INFINITY;
    let mut c_max = f32::NEG_INFINITY;
    for &(_, y) in &c_points {
        if y.is_finite() {
            c_min = c_min.min(y);
            c_max = c_max.max(y);
        }
    }
    if !c_min.is_finite() || !c_max.is_finite() || (c_max - c_min).abs() < 1e-6 {
        c_min = -1.0;
        c_max = 1.0;
    }
    let pad = 0.05f32;

    let mut chart_c = ChartBuilder::on(&panels[2])
        .caption("E1 Consonance Raw C(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, (c_min - pad)..(c_max + pad))?;

    chart_c
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("C_raw")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_c.draw_series(std::iter::once(PathElement::new(
                vec![(x, c_min - pad), (x, c_max + pad)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart_c.draw_series(LineSeries::new(c_points, &BLACK))?;

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
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(min_points, &RED))?
        .label("mass_m3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    let freqs_m0 = run_e4_condition(0.0, seed);
    let freqs_m1 = run_e4_condition(1.0, seed);
    let semitones_m0: Vec<f32> = freqs_m0
        .iter()
        .filter(|f| f.is_finite() && **f > 0.0)
        .map(|f| 12.0 * (f / anchor_hz).log2())
        .collect();
    let semitones_m1: Vec<f32> = freqs_m1
        .iter()
        .filter(|f| f.is_finite() && **f > 0.0)
        .map(|f| 12.0 * (f / anchor_hz).log2())
        .collect();

    let hist_m0_path = out_dir.join("paper_e4_interval_histogram_m0.png");
    render_interval_histogram(
        &hist_m0_path,
        "E4 Interval Histogram (mirror_weight=0.0)",
        &semitones_m0,
        0.0,
        12.0,
        0.5,
    )?;
    let hist_m1_path = out_dir.join("paper_e4_interval_histogram_m1.png");
    render_interval_histogram(
        &hist_m1_path,
        "E4 Interval Histogram (mirror_weight=1.0)",
        &semitones_m1,
        0.0,
        12.0,
        0.5,
    )?;
    Ok(())
}

struct C01Workspace {
    params: LandscapeParams,
    r_ref_peak: f32,
}

#[derive(Clone, Copy)]
struct MetabolicAgent {
    idx: usize,
    energy: f32,
    age_steps: u32,
    sum_c01: f32,
}

fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn build_log2_ratio_scan(space: &Log2Space, anchor_hz: f32) -> Vec<f32> {
    let anchor_log2 = anchor_hz.log2();
    space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect()
}

fn log2_ratio_bounds(log2_ratio_scan: &[f32], min: f32, max: f32) -> (usize, usize) {
    let mut min_idx: Option<usize> = None;
    let mut max_idx: Option<usize> = None;
    for (i, &v) in log2_ratio_scan.iter().enumerate() {
        if v >= min && v <= max {
            if min_idx.is_none() {
                min_idx = Some(i);
            }
            max_idx = Some(i);
        }
    }
    let fallback_min = 0;
    let fallback_max = log2_ratio_scan.len().saturating_sub(1);
    (
        min_idx.unwrap_or(fallback_min),
        max_idx.unwrap_or(fallback_max),
    )
}

fn build_env_scans(
    space: &Log2Space,
    anchor_idx: usize,
    agent_indices: &[usize],
    du_scan: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut env_scan = vec![0.0f32; space.n_bins()];
    let mut density_scan = vec![0.0f32; space.n_bins()];

    let mut add_source = |idx: usize| {
        env_scan[idx] += 1.0;
        let denom = du_scan[idx].max(1e-12);
        density_scan[idx] += 1.0 / denom;
    };

    add_source(anchor_idx);
    for &idx in agent_indices {
        add_source(idx);
    }

    (env_scan, density_scan)
}

fn build_c01_workspace(space: &Log2Space) -> C01Workspace {
    let roughness_kernel = RoughnessKernel::new(KernelParams::default(), 0.005);
    let harmonicity_kernel = HarmonicityKernel::new(space, HarmonicityParams::default());
    let params = LandscapeParams {
        fs: 48_000.0,
        max_hist_cols: 1,
        alpha: 0.0,
        roughness_kernel,
        harmonicity_kernel,
        roughness_scalar_mode: RoughnessScalarMode::Total,
        roughness_half: 0.1,
        consonance_harmonicity_deficit_weight: 1.0,
        consonance_roughness_weight_floor: 0.35,
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
    C01Workspace {
        params,
        r_ref_peak: r_ref.peak,
    }
}

fn compute_c01_scan(
    space: &Log2Space,
    workspace: &C01Workspace,
    env_scan: &[f32],
    density_scan: &[f32],
) -> Vec<f32> {
    let (perc_h_pot_scan, _) = workspace
        .params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(env_scan, space);
    let (perc_r_pot_scan, _) = workspace
        .params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(density_scan, space);

    let mut perc_r_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &perc_r_pot_scan,
        workspace.r_ref_peak,
        workspace.params.roughness_k,
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

    let mut c01_scan = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let (_c_signed, c01) = psycho_state::compose_c_statepm1(
            perc_h_state01_scan[i],
            perc_r_state01_scan[i],
            workspace.params.consonance_harmonicity_deficit_weight,
            workspace.params.consonance_roughness_weight_floor,
            workspace.params.consonance_roughness_weight,
        );
        c01_scan[i] = c01.clamp(0.0, 1.0);
    }
    c01_scan
}

fn update_agent_indices(
    indices: &mut [usize],
    c01_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    lambda: f32,
    sigma: f32,
) {
    if indices.is_empty() {
        return;
    }
    let sigma = sigma.max(1e-6);
    let prev = indices.to_vec();
    let prev_log2: Vec<f32> = prev.iter().map(|&idx| log2_ratio_scan[idx]).collect();

    for (agent_i, current_idx) in prev.iter().copied().enumerate() {
        let start = (current_idx as isize - k as isize).max(min_idx as isize) as usize;
        let end = (current_idx as isize + k as isize).min(max_idx as isize) as usize;
        let mut best_idx = current_idx;
        let mut best_score = f32::NEG_INFINITY;
        for cand in start..=end {
            let cand_log2 = log2_ratio_scan[cand];
            let mut repulsion = 0.0f32;
            for (j, &other_log2) in prev_log2.iter().enumerate() {
                if j == agent_i {
                    continue;
                }
                let dist = (cand_log2 - other_log2).abs();
                repulsion += (-dist / sigma).exp();
            }
            let score = c01_scan[cand] - lambda * repulsion;
            if score > best_score {
                best_score = score;
                best_idx = cand;
            }
        }
        indices[agent_i] = best_idx;
    }
}

fn mean_at_indices(values: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f32 = indices.iter().map(|&idx| values[idx]).sum();
    sum / indices.len() as f32
}

fn render_mean_consonance_plot(
    out_path: &Path,
    series: &[(f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = series.last().map(|(x, _)| *x).unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E2 Mean Consonance Over Time", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("mean C01")
        .draw()?;

    chart.draw_series(LineSeries::new(series.iter().copied(), &BLUE))?;
    root.present()?;
    Ok(())
}

fn render_interval_histogram(
    out_path: &Path,
    caption: &str,
    values: &[f32],
    min: f32,
    max: f32,
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    let counts = histogram_counts(values, min, max, bin_width);
    let y_max = counts
        .iter()
        .map(|(_, count)| *count as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("count")
        .x_labels(25)
        .draw()?;

    for (bin_start, count) in counts {
        let x0 = bin_start;
        let x1 = bin_start + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, count as f32)],
            BLUE.mix(0.6).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_consonance_lifetime_scatter(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    let y_max = deaths
        .iter()
        .map(|(_, lifetime, _)| *lifetime as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E3 Consonance vs Lifetime", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(y_max * 1.05))?;

    chart
        .configure_mesh()
        .x_desc("avg C01")
        .y_desc("lifetime (steps)")
        .draw()?;

    let points = deaths
        .iter()
        .map(|(_, lifetime, avg_c01)| (*avg_c01, *lifetime as f32));
    chart.draw_series(points.map(|(x, y)| Circle::new((x, y), 3, RED.filled())))?;

    root.present()?;
    Ok(())
}

fn render_survival_curve(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    if deaths.is_empty() {
        return Ok(());
    }
    let mut lifetimes: Vec<u32> = deaths.iter().map(|(_, lifetime, _)| *lifetime).collect();
    lifetimes.sort_unstable();
    let total = lifetimes.len() as f32;
    let max_t = *lifetimes.last().unwrap_or(&0) as usize;

    let mut series: Vec<(f32, f32)> = Vec::with_capacity(max_t + 1);
    let mut idx = 0usize;
    let total_usize = lifetimes.len();
    for t in 0..=max_t {
        let t_u32 = t as u32;
        while idx < total_usize && lifetimes[idx] < t_u32 {
            idx += 1;
        }
        let survivors = (total_usize - idx) as f32;
        series.push((t as f32, survivors / total));
    }

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E3 Survival Curve", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..(max_t as f32), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival S(t)")
        .draw()?;

    chart.draw_series(LineSeries::new(series, &BLACK))?;
    root.present()?;
    Ok(())
}

fn render_order_plv_plot(
    out_path: &Path,
    series: &[(f32, f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let x_max = series.last().map(|(x, _, _)| *x).unwrap_or(0.0);
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Order Parameter and PLV", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("value")
        .draw()?;

    let r_points = series.iter().map(|(t, r, _)| (*t, *r));
    let plv_points = series.iter().map(|(t, _, plv)| (*t, *plv));

    chart
        .draw_series(LineSeries::new(r_points, &BLUE))?
        .label("r(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(plv_points, &RED))?
        .label("plv(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_phase_histogram(
    out_path: &Path,
    phase_diffs: &[f32],
    bins: usize,
) -> Result<(), Box<dyn Error>> {
    if bins == 0 {
        return Ok(());
    }
    let min = -PI;
    let max = PI;
    let bin_width = (max - min) / bins as f32;
    let counts = histogram_counts(phase_diffs, min, max, bin_width);
    let y_max = counts
        .iter()
        .map(|(_, count)| *count as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("E5 Phase Difference Histogram", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Δθ (rad)")
        .y_desc("count")
        .x_labels(9)
        .draw()?;

    for (bin_start, count) in counts {
        let x0 = bin_start;
        let x1 = bin_start + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, count as f32)],
            GREEN.mix(0.6).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn histogram_counts(values: &[f32], min: f32, max: f32, bin_width: f32) -> Vec<(f32, usize)> {
    if bin_width <= 0.0 {
        return Vec::new();
    }
    let bins = ((max - min) / bin_width).ceil().max(1.0) as usize;
    let mut counts = vec![0usize; bins];
    for &value in values {
        if value < min || value > max {
            continue;
        }
        let mut idx = ((value - min) / bin_width).floor() as isize;
        if idx as usize >= bins {
            idx = (bins - 1) as isize;
        }
        if idx >= 0 {
            counts[idx as usize] += 1;
        }
    }
    (0..bins)
        .map(|i| (min + i as f32 * bin_width, counts[i]))
        .collect()
}

fn wrap_to_pi(theta: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut x = theta.rem_euclid(two_pi);
    if x > PI {
        x -= two_pi;
    }
    x
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
