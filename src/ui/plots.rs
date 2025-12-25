use crate::ui::viewdata::AgentStateInfo;
use egui::{Align2, Color32, FontId, Id, Stroke, Vec2, Vec2b};
use egui_plot::{
    Bar, BarChart, GridInput, GridMark, Line, LineStyle, Plot, PlotPoints, Points, Polygon, VLine,
    log_grid_spacer,
};
use std::collections::VecDeque;

/// Histogram on a log2 frequency axis (auto bin width).
#[allow(clippy::too_many_arguments)]
pub fn log2_hist_hz(
    ui: &mut egui::Ui,
    title: &str,
    xs_hz: &[f32],
    ys: &[f32],
    y_label: &str,
    y_min: f64,
    y_max: f64,
    height: f32,
) {
    assert_eq!(xs_hz.len(), ys.len());
    if xs_hz.is_empty() {
        return;
    }

    // Use a sparser set to keep labels from colliding.
    const HZ_TICKS: [f64; 5] = [20.0, 100.0, 1_000.0, 10_000.0, 20_000.0];
    let tick_marks_log2: Vec<f64> = HZ_TICKS.iter().map(|hz| hz.log2()).collect();

    // Choose a bar width per bin based on neighbor spacing in log2 frequency.
    let mut bars: Vec<Bar> = Vec::with_capacity(xs_hz.len());
    for i in 0..xs_hz.len() {
        let f = xs_hz[i].max(1.0); // Avoid log2(0).
        let f_left = if i > 0 { xs_hz[i - 1].max(1.0) } else { f };
        let f_right = if i + 1 < xs_hz.len() {
            xs_hz[i + 1].max(1.0)
        } else {
            f
        };

        // Estimate bar width in log2 space from neighbor centers.
        let left = (f_left.log2() + f.log2()) * 0.5;
        let right = (f_right.log2() + f.log2()) * 0.5;
        let width = (right - left).abs().max(0.001);

        bars.push(
            Bar::new(f.log2() as f64, ys[i] as f64)
                .width(width as f64)
                .fill(Color32::from_rgb(240, 120, 120)) // match Roughness accent
                .stroke(egui::Stroke::NONE),
        );
    }

    let chart = BarChart::new(y_label, bars);

    let y_max_fixed = if y_max <= y_min { y_min + 1.0 } else { y_max };

    let tick_marks_log2_for_grid = tick_marks_log2.clone();
    Plot::new(title)
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .include_y(y_min)
        .include_y(y_max_fixed)
        .include_x((20.0f64).log2())
        .include_x((20_000.0f64).log2())
        .default_x_bounds((20.0f64).log2(), (20_000.0f64).log2())
        .default_y_bounds(y_min, y_max_fixed)
        .x_grid_spacer(move |_input: GridInput| {
            tick_marks_log2_for_grid
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let step_size = if i + 1 < tick_marks_log2_for_grid.len() {
                        tick_marks_log2_for_grid[i + 1] - v
                    } else {
                        tick_marks_log2_for_grid[i] - tick_marks_log2_for_grid[i - 1]
                    };
                    GridMark {
                        value: v,
                        step_size,
                    }
                })
                .collect()
        })
        .x_axis_formatter(|mark, _range| {
            let hz = 2f64.powf(mark.value);
            if hz < 1000.0 {
                format!("{:.0} Hz", hz)
            } else {
                format!("{:.1} kHz", hz / 1000.0)
            }
        })
        .y_axis_formatter(|mark, _| format!("{:.2}", mark.value))
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(chart);
        });
}

/// log₂スケールで周波数を描画する汎用プロット関数
#[allow(clippy::too_many_arguments)]
pub fn log2_plot_hz(
    ui: &mut egui::Ui,
    title: &str,
    xs_hz: &[f32],
    ys: &[f32],
    y_label: &str,
    y_min: f64,
    y_max: f64,
    height: f32,
    link_group: Option<&str>,
    line_color: Option<Color32>,
    overlay: Option<(&[f32], &str, Color32)>,
) {
    assert_eq!(
        xs_hz.len(),
        ys.len(),
        "x/y length mismatch: {} vs {}",
        xs_hz.len(),
        ys.len()
    );
    debug_assert!(
        xs_hz
            .windows(2)
            .all(|w| w[0].is_finite() && w[1].is_finite() && w[1] > w[0]),
        "log2_plot_hz expects strictly increasing finite x values"
    );

    // === X軸を log2(Hz) に変換 ===
    let points: PlotPoints = xs_hz
        .iter()
        .zip(ys.iter())
        .map(|(&xx, &yy)| [xx.log2() as f64, yy as f64])
        .collect();

    // === egui_plot用 Line オブジェクト ===
    let mut line = Line::new(y_label, points);
    if let Some(color) = line_color {
        line = line.color(color);
    }
    let overlay_line = overlay.map(|(ys2, label, color)| {
        assert_eq!(
            xs_hz.len(),
            ys2.len(),
            "x/overlay length mismatch: {} vs {}",
            xs_hz.len(),
            ys2.len()
        );
        let points: PlotPoints = xs_hz
            .iter()
            .zip(ys2.iter())
            .map(|(&xx, &yy)| [xx.log2() as f64, yy as f64])
            .collect();
        Line::new(label, points).color(color)
    });

    // === X軸範囲（20〜20kHz）を log2に変換 ===
    let x_min = (20.0f64).log2();
    let x_max = (20_000.0f64).log2();

    // === 描画 ===
    let mut plot = Plot::new(title)
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .include_x(x_min)
        .include_x(x_max)
        .include_y(y_min)
        .include_y(y_max)
        .x_grid_spacer(log_grid_spacer(10))
        .x_axis_formatter(|mark, _range| {
            let hz = 2f64.powf(mark.value);
            format!("{:.0}", hz)
        })
        .y_axis_formatter(|mark, _range| format!("{:.2}", mark.value));
    if let Some(link) = link_group {
        plot = plot.link_axis(Id::new(link), Vec2b::new(true, false));
    }

    plot.show(ui, |plot_ui| {
        plot_ui.line(line);
        if let Some(line) = overlay_line {
            plot_ui.line(line);
        }
    });
}

/// Draw harmonicity above zero and roughness below zero on a log2 axis.
pub fn draw_roughness_harmonicity(
    ui: &mut egui::Ui,
    title: &str,
    xs_hz: &[f32],
    harmonicity: &[f32],
    roughness: &[f32],
    height: f32,
    link_group: Option<&str>,
) {
    assert_eq!(
        xs_hz.len(),
        harmonicity.len(),
        "x/h length mismatch: {} vs {}",
        xs_hz.len(),
        harmonicity.len()
    );
    assert_eq!(
        xs_hz.len(),
        roughness.len(),
        "x/r length mismatch: {} vs {}",
        xs_hz.len(),
        roughness.len()
    );
    if xs_hz.is_empty() {
        return;
    }

    let points_h: PlotPoints = xs_hz
        .iter()
        .zip(harmonicity.iter())
        .map(|(&xx, &yy)| {
            let v = if yy.is_finite() { yy } else { 0.0 };
            [xx.log2() as f64, v.clamp(0.0, 1.0) as f64]
        })
        .collect();
    let points_r: PlotPoints = xs_hz
        .iter()
        .zip(roughness.iter())
        .map(|(&xx, &yy)| {
            let v = if yy.is_finite() { yy } else { 0.0 };
            [xx.log2() as f64, -(v.clamp(0.0, 1.0) as f64)]
        })
        .collect();

    let mut plot = Plot::new(title)
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .include_x((20.0f64).log2())
        .include_x((20_000.0f64).log2())
        .include_y(-1.0)
        .include_y(1.0)
        .default_y_bounds(-1.0, 1.0)
        .x_grid_spacer(log_grid_spacer(10))
        .x_axis_formatter(|mark, _range| {
            let hz = 2f64.powf(mark.value);
            format!("{:.0}", hz)
        })
        .y_axis_formatter(|mark, _range| format!("{:.2}", mark.value));
    if let Some(link) = link_group {
        plot = plot.link_axis(Id::new(link), Vec2b::new(true, false));
    }

    plot.show(ui, |plot_ui| {
        plot_ui.line(Line::new("H", points_h));
        plot_ui.line(Line::new("R", points_r));
    });
}

pub fn time_plot(
    ui: &mut egui::Ui,
    title: &str,
    fs: f64,
    samples: &[f32],
    height: f32,
    show_y_axis: bool,
) {
    let points: PlotPoints = samples
        .iter()
        .enumerate()
        .map(|(i, s)| [i as f64 / fs, *s as f64])
        .collect();
    let line = Line::new("wave", points);

    ui.vertical(|ui| {
        let mut plot = Plot::new(title)
            .height(height)
            .allow_scroll(false)
            .allow_drag(false)
            .include_y(-1.1)
            .include_y(1.1)
            .default_y_bounds(-1.1, 1.1)
            .x_axis_formatter(|mark, _| format!("{:.3} s", mark.value));
        if show_y_axis {
            plot = plot.y_axis_formatter(|mark, _| format!("{:.2}", mark.value));
        } else {
            plot = plot.y_axis_formatter(|_, _| String::new());
        }
        plot.show(ui, |plot_ui| {
            plot_ui.line(line);
        });
    });
}

/// Show magnitude history for neural rhythms to avoid phase aliasing.
pub fn neural_activity_plot(
    ui: &mut egui::Ui,
    history: &VecDeque<(f64, crate::core::modulation::NeuralRhythms)>,
    height: f32,
    window_start: f64,
    window_end: f64,
    link_group: Option<&str>,
) {
    if history.len() < 2 {
        ui.label("No rhythm data");
        return;
    }

    let mut plot = Plot::new("neural_activity")
        .height(height)
        .allow_drag(true)
        .allow_scroll(true)
        .include_y(0.0)
        .include_y(1.0)
        .include_x(window_start)
        .include_x(window_end)
        .default_x_bounds(window_start, window_end)
        .y_axis_formatter(|_, _| String::new())
        .x_axis_formatter(|mark, _| format!("{:.1} s", mark.value));
    if let Some(link) = link_group {
        plot = plot.link_axis(Id::new(link), Vec2b::new(true, false));
    }
    plot.show(ui, |plot_ui| {
        let colors = [
            (Color32::from_rgb(80, 180, 255), "Delta (Phase)"),
            (Color32::from_rgb(70, 225, 135), "Theta (Phase)"),
            (Color32::from_rgb(255, 215, 60), "Alpha (Mag)"),
            (Color32::from_rgb(255, 110, 90), "Beta (Mag)"),
        ];

        for (idx, (color, name)) in colors.iter().enumerate() {
            let is_phase = idx <= 1;
            let mut segments: Vec<(Vec<[f64; 2]>, Color32)> = Vec::new();
            let mut current_pts: Vec<[f64; 2]> = Vec::new();
            let mut current_opacity: f32 = 1.0;
            let mut prev_val = 0.0f64;
            let mut prev_t = 0.0f64;
            let mut has_prev = false;

            for (t, r) in history {
                let (val, mag) = match idx {
                    0 => (
                        (r.delta.phase as f64).rem_euclid(std::f64::consts::TAU)
                            / std::f64::consts::TAU,
                        r.delta.mag,
                    ),
                    1 => (
                        (r.theta.phase as f64).rem_euclid(std::f64::consts::TAU)
                            / std::f64::consts::TAU,
                        r.theta.mag,
                    ),
                    2 => (r.alpha.mag.clamp(0.0, 1.0) as f64, r.alpha.mag),
                    3 => (r.beta.mag.clamp(0.0, 1.0) as f64, r.beta.mag),
                    _ => (0.0, 0.0),
                };

                let raw_opacity = 0.3 + 0.7 * mag.clamp(0.0, 1.0);
                let new_opacity = (raw_opacity * 10.0).round() / 10.0;

                if !has_prev {
                    current_opacity = new_opacity;
                    current_pts.push([*t, val]);
                    prev_val = val;
                    prev_t = *t;
                    has_prev = true;
                    continue;
                }

                let is_wrap = is_phase && (val - prev_val < -0.5);
                let is_opacity_change = (new_opacity - current_opacity).abs() > 0.05;

                if is_wrap {
                    let dt = *t - prev_t;
                    let v_diff = (val + 1.0) - prev_val;
                    if v_diff > 1e-5 {
                        let frac = (1.0 - prev_val) / v_diff;
                        let t_cross = prev_t + dt * frac;
                        current_pts.push([t_cross, 1.0]);
                        segments.push((current_pts, color.gamma_multiply(current_opacity)));
                        current_pts = vec![[t_cross, 0.0]];
                    } else {
                        segments.push((current_pts, color.gamma_multiply(current_opacity)));
                        current_pts = Vec::new();
                    }

                    current_pts.push([*t, val]);
                    current_opacity = new_opacity;
                } else if is_opacity_change {
                    segments.push((current_pts.clone(), color.gamma_multiply(current_opacity)));
                    let last_pt = *current_pts.last().unwrap();
                    current_pts = vec![last_pt, [*t, val]];
                    current_opacity = new_opacity;
                } else {
                    current_pts.push([*t, val]);
                }

                prev_val = val;
                prev_t = *t;
            }

            if !current_pts.is_empty() {
                segments.push((current_pts, color.gamma_multiply(current_opacity)));
            }

            for (seg_i, (pts, col)) in segments.into_iter().enumerate() {
                let series = PlotPoints::from(pts);
                let mut line = Line::new("", series).color(col).width(2.0);
                if seg_i == 0 {
                    line = line.name(*name);
                }
                plot_ui.line(line);
            }
        }
    });
}

/// Show current vs target frequency for each agent with intent arrows.
pub fn plot_population_dynamics(
    ui: &mut egui::Ui,
    agents: &[AgentStateInfo],
    spec_hz: &[f32],
    spec_amps: &[f32],
    height: f32,
) {
    let x_min = (20.0f64).log2();
    let x_max = (20_000.0f64).log2();
    let plot = Plot::new("population_dynamics")
        .height(height)
        .allow_scroll(true)
        .allow_drag(true)
        .include_y(-1.0)
        .include_y(1.1)
        .include_x(x_min)
        .include_x(x_max)
        .x_axis_formatter(|mark, _| format!("{:.0} Hz", 2f64.powf(mark.value)))
        .y_axis_formatter(|mark, _| format!("{:.2}", mark.value))
        .link_axis(Id::new("landscape_group"), Vec2b::new(true, false));

    plot.show(ui, |plot_ui| {
        if spec_hz.len() > 1 && spec_amps.len() > 1 {
            let mut bars: Vec<Bar> = Vec::with_capacity(spec_hz.len().saturating_sub(1));
            for i in 1..spec_hz.len().min(spec_amps.len()) {
                let f = spec_hz[i].max(1.0);
                let f_left = if i > 1 { spec_hz[i - 1].max(1.0) } else { f };
                let f_right = if i + 1 < spec_hz.len() {
                    spec_hz[i + 1].max(1.0)
                } else {
                    f
                };
                let left = (f_left.log2() + f.log2()) * 0.5;
                let right = (f_right.log2() + f.log2()) * 0.5;
                let width = (right - left).abs().max(0.001);
                bars.push(Bar::new(f.log2() as f64, spec_amps[i] as f64).width(width as f64));
            }
            plot_ui.bar_chart(BarChart::new("Sound bodies", bars));
        }

        for agent in agents {
            let y = agent.consonance as f64;
            let x = agent.freq_hz.max(1.0).log2() as f64;
            let xt = agent.target_freq.max(1.0).log2() as f64;
            if (x - xt).abs() > f64::EPSILON {
                plot_ui.line(
                    Line::new(format!("intent-{}", agent.id), vec![[x, y], [xt, y]])
                        .color(Color32::from_rgb(80, 140, 255))
                        .style(LineStyle::Dashed { length: 4.0 }),
                );
            }
            let t = agent.habituation.clamp(0.0, 1.0);
            let r = (50.0 + 205.0 * t) as u8;
            let g = (120.0 + 40.0 * (1.0 - t)) as u8;
            let b = (230.0 * (1.0 - t) + 30.0 * t) as u8;
            // Use log2(window) so point size varies smoothly on a log-frequency x-axis.
            let radius = (agent.integration_window.max(1.0).log2() * 4.0).clamp(3.0, 20.0);
            plot_ui.points(
                Points::new(format!("agent-{}", agent.id), vec![[x, y]])
                    .radius(radius)
                    .color(Color32::from_rgb(r, g, b)),
            );
        }
    });
}

pub fn spectrum_time_freq_axes(
    ui: &mut egui::Ui,
    history: &VecDeque<(f64, crate::ui::viewdata::DorsalFrame)>,
    height: f32,
    window_start: f64,
    window_end: f64,
    link_group: Option<&str>,
) {
    let mut plot = Plot::new("time_freq_spectrum")
        .height(height)
        .allow_drag(true)
        .allow_scroll(true)
        .include_x(window_start)
        .include_x(window_end)
        .include_y(0.0)
        .include_y(3.0)
        .default_x_bounds(window_start, window_end)
        .default_y_bounds(0.0, 3.0)
        .x_axis_formatter(|mark, _| format!("{:.1} s", mark.value))
        .y_axis_formatter(|_, _| String::new());
    if let Some(link) = link_group {
        plot = plot.link_axis(Id::new(link), Vec2b::new(true, false));
    }

    plot.show(ui, |plot_ui| {
        let samples: Vec<(f64, crate::ui::viewdata::DorsalFrame)> = history
            .iter()
            .filter(|(t, _)| *t >= window_start && *t <= window_end)
            .map(|(t, d)| (*t, *d))
            .collect();
        if samples.len() < 2 {
            return;
        }

        let mut max_energy = 0.0f32;
        let mut max_flux = 0.0f32;
        for (_, d) in &samples {
            max_energy = max_energy.max(d.e_low).max(d.e_mid).max(d.e_high);
            max_flux = max_flux.max(d.flux);
        }
        let energy_scale = if max_energy > 0.0 {
            1.0 / max_energy
        } else {
            0.0
        };
        let flux_scale = if max_flux > 0.0 { 3.0 / max_flux } else { 0.0 };

        let band_specs = [
            (2.0f64, Color32::from_rgb(220, 60, 60), "high"),
            (1.0f64, Color32::from_rgb(220, 180, 60), "mid"),
            (0.0f64, Color32::from_rgb(60, 180, 80), "low"),
        ];

        for i in 0..(samples.len() - 1) {
            let (t0, d0) = samples[i];
            let (t1, _) = samples[i + 1];
            let x0 = t0;
            let x1 = t1.max(t0 + 0.0001);

            let vals = [d0.e_high, d0.e_mid, d0.e_low];
            for (band_idx, (y_base, base_color, _name)) in band_specs.iter().enumerate() {
                let intensity = (vals[band_idx] * energy_scale).clamp(0.0, 1.0);
                let alpha = (30.0 + 225.0 * intensity) as u8;
                let color = Color32::from_rgba_unmultiplied(
                    base_color.r(),
                    base_color.g(),
                    base_color.b(),
                    alpha,
                );
                let y0 = *y_base;
                let y1 = y0 + 1.0;
                let poly = vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
                plot_ui.polygon(
                    Polygon::new("", poly)
                        .fill_color(color)
                        .stroke(Stroke::NONE),
                );
            }
        }

        let flux_points: PlotPoints = samples
            .iter()
            .map(|(t, d)| [*t, (d.flux * flux_scale) as f64])
            .collect();
        plot_ui.line(
            Line::new("flux", flux_points)
                .color(Color32::from_rgb(140, 140, 140))
                .width(1.5),
        );
    });
}

/// Visualize neural rhythms (Delta/Theta/Alpha/Beta) as radial gauges.
pub fn neural_compass(
    ui: &mut egui::Ui,
    rhythms: &crate::core::modulation::NeuralRhythms,
    height: f32,
) {
    let color_delta = Color32::from_rgb(80, 180, 255);
    let color_theta = Color32::from_rgb(70, 225, 135);
    let color_alpha = Color32::from_rgb(255, 215, 60);
    let color_beta = Color32::from_rgb(255, 110, 90);
    let bands = [
        ("Delta", color_delta, rhythms.delta),
        ("Theta", color_theta, rhythms.theta),
        ("Alpha", color_alpha, rhythms.alpha),
        ("Beta", color_beta, rhythms.beta),
    ];

    let cheight = (height / 4.0) * 0.95;
    ui.vertical(|ui| {
        for (label, color, rhythm) in bands {
            let desired = egui::vec2(130.0, cheight);
            let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
            let painter = ui.painter_at(rect);
            let center = rect.center();
            let radius = rect.width().min(rect.height()) * 0.45;

            // Background circle
            painter.circle_stroke(center, radius, Stroke::new(1.0, color.gamma_multiply(0.35)));
            painter.circle_filled(center, 2.0, Color32::WHITE);
            painter.circle_stroke(center, 2.0, Stroke::new(1.0, color.gamma_multiply(0.6)));
            let top = center + egui::vec2(0.0, -radius);
            painter.line_segment(
                [center, top],
                Stroke::new(1.0, Color32::WHITE.gamma_multiply(0.3)),
            );

            // Needle with log-style gain and a visible floor.
            let vis_mag =
                ((1.0f32 + (rhythm.mag * 50.0)).ln() / (1.0f32 + 50.0).ln()).clamp(0.35, 1.0);
            let length = radius * (0.3 + 0.7 * vis_mag);
            let angle = rhythm.phase - std::f32::consts::FRAC_PI_2;
            let tip = center + egui::vec2(angle.cos(), angle.sin()) * length;
            let thickness = 2.0 + 3.8 * vis_mag;
            painter.line_segment([center, tip], Stroke::new(thickness, color));

            // Label
            painter.text(
                rect.center_top() + egui::vec2(-40.0, 10.0),
                Align2::CENTER_TOP,
                label,
                FontId::proportional(12.0),
                color,
            );
        }
    });
}

/// Draw a mandala-style rhythm compass combining meter, beat, stability, and error without fast rotation.
pub fn draw_rhythm_mandala(
    ui: &mut egui::Ui,
    rhythms: &crate::core::modulation::NeuralRhythms,
    size: Vec2,
) {
    let side = size.x.min(size.y).clamp(80.0, 200.0);
    let scale = (side / 150.0).clamp(0.6, 1.2);
    let (rect, _resp) = ui.allocate_exact_size(Vec2::splat(side), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    let center = rect.center();
    let radius = rect.width().min(rect.height()) * 0.39;
    let start_angle = -std::f32::consts::FRAC_PI_2;
    let wrap = |p: f32| {
        let mut v = p % std::f32::consts::TAU;
        if v < 0.0 {
            v += std::f32::consts::TAU;
        }
        v
    };

    // Delta: orbiting marker outside the base ring.
    let get_visuals = |mag: f32, color: Color32| {
        let m = mag.clamp(0.0, 1.0);
        let alpha = 0.3 + 0.7 * m;
        let weight = m;
        (color.gamma_multiply(alpha), weight)
    };

    let color_delta = Color32::from_rgb(80, 180, 255);
    let color_theta = Color32::from_rgb(70, 225, 135);
    let color_alpha = Color32::from_rgb(255, 215, 60);
    let color_beta = Color32::from_rgb(255, 110, 90);

    // Delta: progress arc as the meter with magnitude-driven opacity/width.
    let delta_phase = wrap(rhythms.delta.phase);
    let outer_r = radius * 1.15;
    let (d_color, d_weight) = get_visuals(rhythms.delta.mag, color_delta);
    painter.circle_stroke(
        center,
        outer_r,
        Stroke::new(1.0, d_color.gamma_multiply(0.2)),
    );
    if delta_phase > 0.01 {
        let steps = (delta_phase * 12.0).max(2.0) as usize;
        let points: Vec<egui::Pos2> = (0..=steps)
            .map(|i| {
                let angle = start_angle + (delta_phase * i as f32 / steps as f32);
                center + egui::vec2(angle.cos(), angle.sin()) * outer_r
            })
            .collect();
        let width = 1.0 + d_weight * 3.0;
        painter.add(egui::Shape::line(points, Stroke::new(width, d_color)));
    }
    let delta_tip_angle = start_angle + delta_phase;
    let delta_tip_pos = center + egui::vec2(delta_tip_angle.cos(), delta_tip_angle.sin()) * outer_r;
    painter.circle_filled(delta_tip_pos, 2.0 + d_weight * 2.0, d_color);

    // Theta: base ring plus a slow cursor.
    let base_circle = Color32::from_gray(180);
    painter.circle_stroke(center, radius, Stroke::new(2.5, base_circle));
    let (t_color, t_weight) = get_visuals(rhythms.theta.mag, color_theta);
    let theta_angle = start_angle + wrap(rhythms.theta.phase);
    let theta_pos = center + egui::vec2(theta_angle.cos(), theta_angle.sin()) * radius;
    let dot_radius = 2.0 + t_weight * 3.0;
    painter.circle_filled(theta_pos, dot_radius, t_color);
    painter.circle_stroke(
        theta_pos,
        dot_radius + 1.0,
        Stroke::new(1.0, t_color.gamma_multiply(0.5)),
    );
    painter.circle_filled(center, 3.0, base_circle);

    // Alpha: stability axis along the vertical.
    let alpha_mag = rhythms.alpha.mag.clamp(0.0, 1.0);
    let alpha_vis = 0.25 + 0.75 * alpha_mag;
    let alpha_color = color_alpha.gamma_multiply(alpha_vis);
    let alpha_width = 2.0 + 6.0 * alpha_mag;
    painter.line_segment(
        [
            center + egui::vec2(0.0, -radius),
            center + egui::vec2(0.0, radius),
        ],
        Stroke::new(alpha_width, alpha_color),
    );

    // Beta: prediction error cross-grid.
    let beta_mag = rhythms.beta.mag.clamp(0.0, 1.0);
    let beta_vis = 0.2 + 0.8 * beta_mag;
    let beta_color = color_beta.gamma_multiply(beta_vis);
    let beta_width = 1.5 + 6.0 * beta_mag;
    painter.line_segment(
        [
            center + egui::vec2(0.0, -radius),
            center + egui::vec2(0.0, radius),
        ],
        Stroke::new(beta_width, beta_color),
    );
    painter.line_segment(
        [
            center + egui::vec2(-radius, 0.0),
            center + egui::vec2(radius, 0.0),
        ],
        Stroke::new(beta_width, beta_color),
    );
}
