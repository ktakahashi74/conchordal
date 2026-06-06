use crate::ui::viewdata::VoiceStateInfo;
use egui::{Color32, Id, Stroke, Vec2, Vec2b};
use egui_plot::{
    Bar, BarChart, GridInput, GridMark, Line, LineStyle, Plot, PlotPoints, Points, Polygon,
    log_grid_spacer,
};
use std::collections::VecDeque;

const LOG2_HZ_MIN: f64 = 20.0;
const LOG2_HZ_MAX: f64 = 20_000.0;

#[inline]
fn log2_plot_x_bounds() -> (f64, f64) {
    (LOG2_HZ_MIN.log2(), LOG2_HZ_MAX.log2())
}

#[inline]
fn log2_bar_center_width(f_left_hz: f32, f_hz: f32, f_right_hz: f32) -> (f64, f64) {
    let f_left = f_left_hz.max(1.0);
    let f = f_hz.max(1.0);
    let f_right = f_right_hz.max(1.0);
    let center = f.log2();
    let left = (f_left.log2() + center) * 0.5;
    let right = (f_right.log2() + center) * 0.5;
    let width = (right - left).abs().max(0.001);
    (center as f64, width as f64)
}

/// Histogram on a log2 frequency axis (auto bin width).
#[allow(dead_code)]
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
        let f = xs_hz[i];
        let f_left = if i > 0 { xs_hz[i - 1] } else { f };
        let f_right = if i + 1 < xs_hz.len() { xs_hz[i + 1] } else { f };
        let (center, width) = log2_bar_center_width(f_left, f, f_right);

        bars.push(
            Bar::new(center, ys[i] as f64)
                .width(width)
                .fill(Color32::from_rgb(240, 120, 120)) // match Roughness accent
                .stroke(egui::Stroke::NONE),
        );
    }

    let chart = BarChart::new(y_label, bars);

    let y_max_fixed = if y_max <= y_min { y_min + 1.0 } else { y_max };

    let tick_marks_log2_for_grid = tick_marks_log2.clone();
    let (x_min, x_max) = log2_plot_x_bounds();
    Plot::new(title)
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_double_click_reset(false)
        .include_y(y_min)
        .include_y(y_max_fixed)
        .include_x(x_min)
        .include_x(x_max)
        .default_x_bounds(x_min, x_max)
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
            plot_ui.set_plot_bounds_x(x_min..=x_max);
            plot_ui.bar_chart(chart);
        });
}

/// Generic log2-frequency plot.
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

    // === Convert X axis to log2(Hz) ===
    let points: PlotPoints = xs_hz
        .iter()
        .zip(ys.iter())
        .map(|(&xx, &yy)| [xx.log2() as f64, yy as f64])
        .collect();

    // === Line object for egui_plot ===
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

    // === Convert X range (20-20 kHz) to log2 ===
    let (x_min, x_max) = log2_plot_x_bounds();

    // === Render ===
    let y_max_fixed = if y_max <= y_min { y_min + 1.0 } else { y_max };

    let mut plot = Plot::new(title)
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_double_click_reset(false)
        .include_x(x_min)
        .include_x(x_max)
        .include_y(y_min)
        .include_y(y_max_fixed)
        .default_x_bounds(x_min, x_max)
        .default_y_bounds(y_min, y_max_fixed)
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
        plot_ui.set_plot_bounds_x(x_min..=x_max);
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

    let (x_min, x_max) = log2_plot_x_bounds();
    let mut plot = Plot::new(title)
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_double_click_reset(false)
        .include_x(x_min)
        .include_x(x_max)
        .include_y(-1.0)
        .include_y(1.0)
        .default_x_bounds(x_min, x_max)
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
        plot_ui.set_plot_bounds_x(x_min..=x_max);
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
    history: &VecDeque<(f64, crate::core::meter::MeterState)>,
    height: f32,
    window_start: f64,
    window_end: f64,
    link_group: Option<&str>,
) {
    let fallback_history;
    let history = if history.len() < 2 {
        if let Some((t, meter)) = history.back() {
            fallback_history = VecDeque::from([
                ((*t - 0.1).max(window_start), *meter),
                ((*t).min(window_end), *meter),
            ]);
            &fallback_history
        } else {
            ui.label("No meter data");
            return;
        }
    } else {
        history
    };

    let mut plot = Plot::new("neural_activity")
        .height(height)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_zoom(false)
        .allow_double_click_reset(false)
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
        plot_ui.set_plot_bounds_x(window_start..=window_end);
        plot_ui.set_plot_bounds_y(0.0..=1.0);
        let colors = [
            (Color32::from_rgb(80, 180, 255), "Beat (Phase)"),
            (Color32::from_rgb(70, 225, 135), "Sub (Phase)"),
            (Color32::from_rgb(255, 215, 60), "Beat conf"),
            (Color32::from_rgb(255, 110, 90), "Measure conf"),
        ];

        for (idx, (color, name)) in colors.iter().enumerate() {
            let is_phase = idx <= 1;
            let mut segments: Vec<(Vec<[f64; 2]>, Color32, f32)> = Vec::new();
            let mut current_pts: Vec<[f64; 2]> = Vec::new();
            let mut current_opacity: f32 = 1.0;
            let mut prev_val = 0.0f64;
            let mut prev_t = 0.0f64;
            let mut has_prev = false;

            for (t, m) in history {
                let (val, mag) = match idx {
                    0 => (
                        (m.beat.phase as f64).rem_euclid(std::f64::consts::TAU)
                            / std::f64::consts::TAU,
                        m.beat.confidence.clamp(0.0, 1.0),
                    ),
                    1 => (
                        (m.subdivision.phase as f64).rem_euclid(std::f64::consts::TAU)
                            / std::f64::consts::TAU,
                        m.subdivision.confidence.clamp(0.0, 1.0),
                    ),
                    2 => {
                        let c = m.beat.confidence.clamp(0.0, 1.0);
                        (c as f64, c)
                    }
                    3 => {
                        let c = m.measure.confidence.clamp(0.0, 1.0);
                        (c as f64, c)
                    }
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
                        segments.push((
                            current_pts,
                            color.gamma_multiply(current_opacity),
                            0.8 + 2.2 * current_opacity,
                        ));
                        current_pts = vec![[t_cross, 0.0]];
                    } else {
                        segments.push((
                            current_pts,
                            color.gamma_multiply(current_opacity),
                            0.8 + 2.2 * current_opacity,
                        ));
                        current_pts = Vec::new();
                    }

                    current_pts.push([*t, val]);
                    current_opacity = new_opacity;
                } else if is_opacity_change {
                    segments.push((
                        current_pts.clone(),
                        color.gamma_multiply(current_opacity),
                        0.8 + 2.2 * current_opacity,
                    ));
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
                segments.push((
                    current_pts,
                    color.gamma_multiply(current_opacity),
                    0.8 + 2.2 * current_opacity,
                ));
            }

            for (seg_i, (pts, col, width)) in segments.into_iter().enumerate() {
                let series = PlotPoints::from(pts);
                let mut line = Line::new("", series).color(col).width(width);
                if seg_i == 0 {
                    line = line.name(*name);
                }
                plot_ui.line(line);
            }
        }
    });
}

/// Show current vs target frequency for each voice with target arrows.
pub fn plot_population_dynamics(
    ui: &mut egui::Ui,
    voices: &[VoiceStateInfo],
    spec_hz: &[f32],
    spec_amps: &[f32],
    height: f32,
) {
    let (x_min, x_max) = log2_plot_x_bounds();
    let plot = Plot::new("population_dynamics")
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_double_click_reset(false)
        .include_y(-1.0)
        .include_y(1.1)
        .include_x(x_min)
        .include_x(x_max)
        .default_x_bounds(x_min, x_max)
        .x_axis_formatter(|mark, _| format!("{:.0} Hz", 2f64.powf(mark.value)))
        .y_axis_formatter(|mark, _| format!("{:.2}", mark.value))
        .link_axis(Id::new("landscape_group"), Vec2b::new(true, false));

    plot.show(ui, |plot_ui| {
        plot_ui.set_plot_bounds_x(x_min..=x_max);
        if spec_hz.len() > 1 && spec_amps.len() > 1 {
            let mut bars: Vec<Bar> = Vec::with_capacity(spec_hz.len().saturating_sub(1));
            for i in 1..spec_hz.len().min(spec_amps.len()) {
                let f = spec_hz[i];
                let f_left = if i > 1 { spec_hz[i - 1] } else { f };
                let f_right = if i + 1 < spec_hz.len() {
                    spec_hz[i + 1]
                } else {
                    f
                };
                let (center, width) = log2_bar_center_width(f_left, f, f_right);
                bars.push(Bar::new(center, spec_amps[i] as f64).width(width));
            }
            plot_ui.bar_chart(BarChart::new("Sound bodies", bars));
        }

        for voice in voices {
            let y = voice.consonance as f64;
            let x = voice.freq_hz.max(1.0).log2() as f64;
            let xt = voice.target_freq.max(1.0).log2() as f64;
            if (x - xt).abs() > f64::EPSILON {
                plot_ui.line(
                    Line::new(format!("target-{}", voice.id), vec![[x, y], [xt, y]])
                        .color(Color32::from_rgb(80, 140, 255))
                        .style(LineStyle::Dashed { length: 4.0 }),
                );
            }
            let t = voice.consonance.clamp(0.0, 1.0);
            let r = (60.0 + 180.0 * (1.0 - t)) as u8;
            let g = (80.0 + 140.0 * t) as u8;
            let b = (180.0 - 80.0 * t) as u8;
            // Use log2(window) so point size varies smoothly on a log-frequency x-axis.
            let radius = (voice.integration_window.max(1.0).log2() * 4.0).clamp(3.0, 20.0);
            plot_ui.points(
                Points::new(format!("voice-{}", voice.id), vec![[x, y]])
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
    let fallback_history;
    let history = if history.len() < 2 {
        if let Some((t, dorsal)) = history.back() {
            fallback_history = VecDeque::from([
                ((*t - 0.1).max(window_start), *dorsal),
                ((*t).min(window_end), *dorsal),
            ]);
            &fallback_history
        } else {
            ui.label("No salience data");
            return;
        }
    } else {
        history
    };

    let mut plot = Plot::new("time_freq_spectrum")
        .height(height)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_zoom(false)
        .allow_double_click_reset(false)
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
        plot_ui.set_plot_bounds_x(window_start..=window_end);
        plot_ui.set_plot_bounds_y(0.0..=3.0);
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

#[derive(Clone, Copy, Debug, Default)]
struct ListenerMandalaLevels {
    attention: Option<f32>,
    stability: Option<f32>,
    tension: Option<f32>,
}

/// Draw listener-side meter and salience state as an at-a-glance mandala.
///
/// `entrain_phases` are the live voices' offset-removed rhythm phases and
/// `entrain_order_r` their Kuramoto order; they are overlaid as dots on the
/// subdivision ring with a mean-resultant vector so voice entrainment reads on
/// the same compass as the listener meter.
pub fn draw_listener_mandala(
    ui: &mut egui::Ui,
    listener: &crate::ui::viewdata::ListenerFrame,
    entrain_phases: &[f32],
    entrain_order_r: Option<f32>,
    size: Vec2,
) {
    let levels = ListenerMandalaLevels {
        attention: listener
            .has_fast_state
            .then_some(listener.attention_level.clamp(0.0, 1.0)),
        stability: listener
            .has_state
            .then_some(listener.stability_level.clamp(0.0, 1.0)),
        tension: listener
            .has_state
            .then_some(listener.tension_level.clamp(0.0, 1.0)),
    };
    draw_mandala(
        ui,
        &listener.meter,
        size,
        levels,
        entrain_phases,
        entrain_order_r,
    );
}

fn draw_mandala(
    ui: &mut egui::Ui,
    meter: &crate::core::meter::MeterState,
    size: Vec2,
    listener_levels: ListenerMandalaLevels,
    entrain_phases: &[f32],
    entrain_order_r: Option<f32>,
) {
    let side = size.x.min(size.y).clamp(80.0, 200.0);
    let _scale = (side / 150.0).clamp(0.6, 1.2);
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

    // Confidence (entrainment PLV) drives opacity/weight, not drive amplitude.
    let get_visuals = |confidence: f32, color: Color32| {
        let c = confidence.clamp(0.0, 1.0);
        let alpha = 0.3 + 0.7 * c;
        let weight = c;
        (color.gamma_multiply(alpha), weight)
    };

    let color_beat = Color32::from_rgb(80, 180, 255);
    let color_sub = Color32::from_rgb(70, 225, 135);
    let color_measure = Color32::from_rgb(255, 215, 60);
    let color_error = Color32::from_rgb(255, 110, 90);
    let outer_r = radius * 1.15;

    if let Some(attention) = listener_levels.attention {
        let attention = attention.clamp(0.0, 1.0);
        if attention > 0.01 {
            let halo_r = outer_r * (1.12 + 0.12 * attention);
            let halo_color =
                Color32::from_rgb(90, 180, 230).gamma_multiply(0.06 + 0.20 * attention);
            painter.circle_filled(center, halo_r, halo_color);
        }
    }

    if let Some(tension) = listener_levels.tension {
        let tension = tension.clamp(0.0, 1.0);
        if tension > 0.01 {
            let tension_color =
                Color32::from_rgb(255, 95, 65).gamma_multiply(0.18 + 0.72 * tension);
            painter.circle_stroke(
                center,
                outer_r * 1.08,
                Stroke::new(1.0 + 4.0 * tension, tension_color),
            );
            painter.circle_stroke(
                center,
                outer_r * 1.18,
                Stroke::new(0.5 + 2.5 * tension, tension_color.gamma_multiply(0.45)),
            );
        }
    }

    // Beat (tactus): progress arc as the meter with confidence-driven opacity/width.
    let beat_phase = wrap(meter.beat.phase);
    let (d_color, d_weight) = get_visuals(meter.beat.confidence, color_beat);
    painter.circle_stroke(
        center,
        outer_r,
        Stroke::new(1.0, d_color.gamma_multiply(0.2)),
    );
    if beat_phase > 0.01 {
        let steps = (beat_phase * 12.0).max(2.0) as usize;
        let points: Vec<egui::Pos2> = (0..=steps)
            .map(|i| {
                let angle = start_angle + (beat_phase * i as f32 / steps as f32);
                center + egui::vec2(angle.cos(), angle.sin()) * outer_r
            })
            .collect();
        let width = 1.0 + d_weight * 3.0;
        painter.add(egui::Shape::line(points, Stroke::new(width, d_color)));
    }
    let beat_tip_angle = start_angle + beat_phase;
    let beat_tip_pos = center + egui::vec2(beat_tip_angle.cos(), beat_tip_angle.sin()) * outer_r;
    painter.circle_filled(beat_tip_pos, 2.0 + d_weight * 2.0, d_color);

    // Subdivision (tatum): base ring plus a fast cursor.
    let base_circle = Color32::from_gray(180);
    painter.circle_stroke(center, radius, Stroke::new(2.5, base_circle));
    let (t_color, t_weight) = get_visuals(meter.subdivision.confidence, color_sub);
    let sub_angle = start_angle + wrap(meter.subdivision.phase);
    let sub_pos = center + egui::vec2(sub_angle.cos(), sub_angle.sin()) * radius;
    let dot_radius = 2.0 + t_weight * 3.0;
    painter.circle_filled(sub_pos, dot_radius, t_color);
    painter.circle_stroke(
        sub_pos,
        dot_radius + 1.0,
        Stroke::new(1.0, t_color.gamma_multiply(0.5)),
    );
    let (center_radius, center_color) = if let Some(stability) = listener_levels.stability {
        let stability = stability.clamp(0.0, 1.0);
        (
            2.8 + 3.8 * stability,
            Color32::from_rgb(210, 255, 225).gamma_multiply(0.35 + 0.65 * stability),
        )
    } else {
        (3.0, base_circle)
    };
    painter.circle_filled(center, center_radius, center_color);
    painter.circle_stroke(
        center,
        center_radius + 1.0,
        Stroke::new(1.0, center_color.gamma_multiply(0.7)),
    );

    // Measure: a downbeat hand pointing at the measure phase, shown only when an
    // accent grouping holds. Length and opacity track the measure confidence.
    if meter.measure_ratio > 0 {
        let measure_conf = meter.measure.confidence.clamp(0.0, 1.0);
        let measure_vis = 0.25 + 0.75 * measure_conf;
        let measure_color = color_measure.gamma_multiply(measure_vis);
        let measure_width = 2.0 + 4.0 * measure_conf;
        let measure_angle = start_angle + wrap(meter.measure.phase);
        let measure_len = radius * (0.35 + 0.6 * measure_conf);
        let tip = center + egui::vec2(measure_angle.cos(), measure_angle.sin()) * measure_len;
        painter.line_segment([center, tip], Stroke::new(measure_width, measure_color));
        painter.circle_filled(tip, 2.0 + 2.0 * measure_conf, measure_color);
    }

    // Prediction error cross-grid: intensifies as beat entrainment confidence drops.
    let error_mag = (1.0 - meter.beat.confidence).clamp(0.0, 1.0);
    let error_vis = 0.2 + 0.8 * error_mag;
    let error_color = color_error.gamma_multiply(error_vis);
    let error_width = 1.5 + 6.0 * error_mag;
    painter.line_segment(
        [
            center + egui::vec2(0.0, -radius),
            center + egui::vec2(0.0, radius),
        ],
        Stroke::new(error_width, error_color),
    );
    painter.line_segment(
        [
            center + egui::vec2(-radius, 0.0),
            center + egui::vec2(radius, 0.0),
        ],
        Stroke::new(error_width, error_color),
    );

    // Voice entrainment: each live Entrain voice's phase as a dot on the
    // subdivision ring, plus the Kuramoto mean-resultant vector. Clustered dots
    // and a long vector mean the voices are phase-locked. Distinct violet so it
    // reads apart from the beat/subdivision/measure/error meter marks.
    if !entrain_phases.is_empty() {
        let voice_color = Color32::from_rgb(196, 150, 255);
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut n = 0u32;
        for &phase in entrain_phases {
            if !phase.is_finite() {
                continue;
            }
            let angle = start_angle + wrap(phase);
            let dot = center + egui::vec2(angle.cos(), angle.sin()) * radius;
            painter.circle_filled(dot, 2.2, voice_color.gamma_multiply(0.9));
            sx += phase.cos();
            sy += phase.sin();
            n += 1;
        }
        if n > 0 && (sx != 0.0 || sy != 0.0) {
            let r = entrain_order_r.unwrap_or(0.0).clamp(0.0, 1.0);
            let mean_angle = start_angle + wrap(sy.atan2(sx));
            let tip = center + egui::vec2(mean_angle.cos(), mean_angle.sin()) * (radius * r);
            painter.line_segment([center, tip], Stroke::new(2.0, voice_color));
            painter.circle_filled(tip, 2.8, voice_color);
        }
    }
}
