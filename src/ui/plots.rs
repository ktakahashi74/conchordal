use crate::viewdata::VoiceStateInfo;
use egui::{Color32, Id, Stroke, Vec2, Vec2b};
use egui_plot::{
    Bar, BarChart, Line, LineStyle, Plot, PlotPoints, Points, Polygon, log_grid_spacer,
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
    link_population: Option<&str>,
    line_color: Option<Color32>,
    overlay: Option<(&[f32], &str, Color32)>,
) {
    debug_assert!(
        xs_hz
            .windows(2)
            .all(|w| w[0].is_finite() && w[1].is_finite() && w[1] > w[0]),
        "log2_plot_hz expects strictly increasing finite x values"
    );

    // === Convert X axis to log2(Hz) ===
    // zip truncates to the shorter slice, so a length mismatch degrades the
    // drawing instead of panicking.
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
    if let Some(link) = link_population {
        plot = plot.link_axis(Id::unique(link), Vec2b::new(true, false));
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
    link_population: Option<&str>,
) {
    if xs_hz.is_empty() {
        return;
    }

    // zip truncates to the shorter slice, so a length mismatch degrades the
    // drawing instead of panicking.
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
    if let Some(link) = link_population {
        plot = plot.link_axis(Id::unique(link), Vec2b::new(true, false));
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

// Divisible by 2, 3 and 4; at the dashboard width each bin remains visible.
const STROBE_BINS: usize = 96;
const STROBE_ROW_HEIGHT: f32 = 6.0;
const STROBE_AXIS_HEIGHT: f32 = 20.0;
// Reuse the existing meter readout's confidence floor for supported display.
pub(super) const METER_DISPLAY_CONFIDENCE_MIN: f32 = 0.08;

struct StrobeRow {
    start_sec: f64,
    unknown: bool,
    flux: [Option<f32>; STROBE_BINS],
    subdivision_ratio: Option<u8>,
}

fn supported_subdivision(meter: &crate::core::meter::MeterState) -> Option<u8> {
    (meter.beat.confidence >= METER_DISPLAY_CONFIDENCE_MIN
        && meter.subdivision.confidence >= METER_DISPLAY_CONFIDENCE_MIN
        && matches!(meter.subdivision_ratio, 2..=4))
    .then_some(meter.subdivision_ratio)
}

/// Fold observed phases only. Unsupported spans occupy one unknown row without
/// inventing beat boundaries; reacquisition starts a new, possibly partial beat.
fn strobe_rows(
    history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    window_end: f64,
    max_rows: usize,
) -> VecDeque<StrobeRow> {
    let mut rows: VecDeque<StrobeRow> = VecDeque::new();
    let mut previous_phase = None;
    for hop in history.iter().filter(|hop| {
        hop.time_sec >= window_end - crate::viewdata::RHYTHM_HISTORY_SEC
            && hop.time_sec <= window_end
    }) {
        let supported = hop.meter.beat.confidence >= METER_DISPLAY_CONFIDENCE_MIN
            && hop.meter.beat.phase.is_finite();
        let phase = hop.meter.beat.phase.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
        let wrap = previous_phase.is_some_and(|prev| phase - prev < -0.5);
        let new_row = rows.back().is_none_or(|row| row.unknown == supported) || (supported && wrap);
        if new_row {
            rows.push_back(StrobeRow {
                start_sec: hop.time_sec,
                unknown: !supported,
                flux: [None; STROBE_BINS],
                subdivision_ratio: None,
            });
            if rows.len() > max_rows {
                rows.pop_front();
            }
        }
        if supported && let Some(row) = rows.back_mut() {
            let bin = ((phase * STROBE_BINS as f32) as usize).min(STROBE_BINS - 1);
            if hop.attention.flux.is_finite() {
                let flux = hop.attention.flux.max(0.0);
                row.flux[bin] = Some(row.flux[bin].map_or(flux, |prev| prev.max(flux)));
            }
            row.subdivision_ratio = supported_subdivision(&hop.meter);
        }
        previous_phase = supported.then_some(phase);
    }
    rows
}

pub(crate) fn beat_strobe_plot(
    ui: &mut egui::Ui,
    history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    height: f32,
    window_end: f64,
) {
    let max_rows = ((height - STROBE_AXIS_HEIGHT) / STROBE_ROW_HEIGHT)
        .floor()
        .max(1.0) as usize;
    let rows = strobe_rows(history, window_end, max_rows);
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), height),
        egui::Sense::hover(),
    );
    response.on_hover_text("Each row is one observed beat; newest below. Blue: auditory flux, normalized over visible rows. Gray: unknown phase or unobserved bin. Unsupported spans are not divided into guessed beats.");
    let painter = ui.painter_at(rect);
    let left = rect.left() + 36.0;
    let width = (rect.right() - left - 8.0).max(1.0);
    let x = |phase: f32| left + width * phase;
    let font = egui::FontId::monospace(10.0);
    let ticks = rows
        .back()
        .and_then(|row| row.subdivision_ratio)
        .unwrap_or(1);
    for tick in 0..=ticks {
        let phase = tick as f32 / ticks as f32;
        // Keep the edge labels inside the strobe instead of clipping at its border.
        let align = if tick == 0 {
            egui::Align2::LEFT_TOP
        } else if tick == ticks {
            egui::Align2::RIGHT_TOP
        } else {
            egui::Align2::CENTER_TOP
        };
        painter.text(
            egui::pos2(x(phase), rect.top() + 2.0),
            align,
            format!("{phase:.2}"),
            font.clone(),
            Color32::LIGHT_GRAY,
        );
    }
    let max_flux = rows
        .iter()
        .flat_map(|row| row.flux.iter().flatten())
        .copied()
        .fold(0.0_f32, f32::max);
    for (index, row) in rows.iter().enumerate() {
        // Stack from just under the phase labels; a full strobe ends at the
        // bottom, and a young one leaves no gap under the labels.
        let y = rect.top() + STROBE_AXIS_HEIGHT + index as f32 * STROBE_ROW_HEIGHT;
        let row_rect = egui::Rect::from_min_size(
            egui::pos2(left, y),
            Vec2::new(width, STROBE_ROW_HEIGHT - 1.0),
        );
        painter.rect_filled(
            row_rect,
            0.0,
            Color32::from_gray(if row.unknown { 70 } else { 38 }),
        );
        for (bin, flux) in row.flux.iter().enumerate() {
            if let Some(flux) = flux {
                let intensity = if max_flux > 0.0 { flux / max_flux } else { 0.0 };
                let cell = egui::Rect::from_min_max(
                    egui::pos2(x(bin as f32 / STROBE_BINS as f32), y),
                    egui::pos2(x((bin + 1) as f32 / STROBE_BINS as f32), row_rect.bottom()),
                );
                painter.rect_filled(
                    cell,
                    0.0,
                    Color32::from_rgb(80, 180, 255).gamma_multiply(0.12 + 0.88 * intensity),
                );
            }
        }
        if let Some(ratio) = row.subdivision_ratio {
            for tick in 1..ratio {
                let x = x(tick as f32 / ratio as f32);
                painter.line_segment(
                    [egui::pos2(x, y), egui::pos2(x, row_rect.bottom())],
                    Stroke::new(0.6, Color32::from_rgb(70, 225, 135)),
                );
            }
        }
    }
    if let Some(row) = rows.front() {
        painter.text(
            egui::pos2(rect.left(), rect.top() + STROBE_AXIS_HEIGHT),
            egui::Align2::LEFT_TOP,
            format!("{:.0}s", row.start_sec - window_end),
            font,
            Color32::GRAY,
        );
    } else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No meter observations",
            font,
            Color32::GRAY,
        );
    }
}

/// Slow meter quantities keep independent physical units on a fixed 30 s axis.
pub(crate) fn rhythm_slow_plots(
    ui: &mut egui::Ui,
    history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    window_end: f64,
) {
    let blue = Color32::from_rgb(80, 180, 255);
    let green = Color32::from_rgb(70, 225, 135);
    let yellow = Color32::from_rgb(255, 215, 60);
    let red = Color32::from_rgb(255, 110, 90);
    for (panel, (id, axis, height, upper)) in [
        ("rhythm_tempo", "BPM", 40.0, 240.0),
        ("rhythm_ratio", "Sub ratio", 34.0, 4.0),
        ("rhythm_confidence", "Confidence", 54.0, 1.0),
    ]
    .into_iter()
    .enumerate()
    {
        Plot::new(id)
            .height(height)
            .allow_drag(false)
            .allow_scroll(false)
            .allow_zoom(false)
            .allow_double_click_reset(false)
            .y_axis_min_width(36.0)
            .show_axes([panel == 2, true])
            .x_axis_formatter(|mark, _| format!("{:.0} s", mark.value))
            .show(ui, |plot_ui| {
                plot_ui.set_plot_bounds_x(-crate::viewdata::RHYTHM_HISTORY_SEC..=0.0);
                plot_ui.set_plot_bounds_y(0.0..=upper);
                // A rotated axis title is taller than these short plots and overlaps
                // its neighbors, so the quantity is named inside the plot instead.
                plot_ui.text(
                    egui_plot::Text::new(
                        "",
                        egui_plot::PlotPoint::new(-crate::viewdata::RHYTHM_HISTORY_SEC, upper),
                        axis,
                    )
                    .anchor(egui::Align2::LEFT_TOP)
                    .color(Color32::GRAY),
                );
                let traces: &[(Color32, &str)] = match panel {
                    0 => &[(blue, "Beat tempo (BPM)")],
                    1 => &[(yellow, "Subdivision ratio")],
                    _ => &[
                        (blue, "Beat confidence"),
                        (green, "Subdivision confidence"),
                        (red, "Measure confidence"),
                    ],
                };
                for (trace, &(color, name)) in traces.iter().enumerate() {
                    let mut points = Vec::new();
                    for hop in history.iter().filter(|hop| hop.time_sec <= window_end) {
                        let meter = hop.meter;
                        let value = match panel {
                            0 => (meter.beat.confidence >= METER_DISPLAY_CONFIDENCE_MIN
                                && meter.beat.freq_hz > 0.0)
                                .then_some(meter.beat.freq_hz as f64 * 60.0),
                            1 => supported_subdivision(&meter).map(f64::from),
                            _ => Some(match trace {
                                0 => meter.beat.confidence,
                                1 => meter.subdivision.confidence,
                                _ => meter.measure.confidence,
                            } as f64),
                        };
                        if let Some(value) = value.filter(|value| value.is_finite()) {
                            points.push([hop.time_sec - window_end, value]);
                        } else if !points.is_empty() {
                            plot_ui.line(Line::new(name, std::mem::take(&mut points)).color(color));
                        }
                    }
                    if !points.is_empty() {
                        plot_ui.line(Line::new(name, points).color(color));
                    }
                }
            });
    }
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
        .link_axis(Id::unique("landscape_population"), Vec2b::new(true, false));

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
    history: &VecDeque<(f64, crate::viewdata::DorsalFrame)>,
    height: f32,
    window_start: f64,
    window_end: f64,
    link_population: Option<&str>,
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

    // Relative seconds keep the time grid stationary.
    let span = window_start - window_end;
    let mut plot = Plot::new("time_freq_spectrum")
        .height(height)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_zoom(false)
        .allow_double_click_reset(false)
        .include_x(span)
        .include_x(0.0)
        .include_y(0.0)
        .include_y(3.0)
        .default_x_bounds(span, 0.0)
        .default_y_bounds(0.0, 3.0)
        .x_axis_formatter(|mark, _| format!("{:.1} s", mark.value))
        .y_axis_formatter(|_, _| String::new());
    if let Some(link) = link_population {
        plot = plot.link_axis(Id::unique(link), Vec2b::new(true, false));
    }

    plot.show(ui, |plot_ui| {
        plot_ui.set_plot_bounds_x(span..=0.0);
        plot_ui.set_plot_bounds_y(0.0..=3.0);
        // The caller keeps one sample before the window; keep one after its end
        // so both edge segments reach the plot border.
        let samples: Vec<(f64, crate::viewdata::DorsalFrame)> = history
            .iter()
            .scan(f64::NEG_INFINITY, |prev, &(t, d)| {
                let keep = *prev <= window_end;
                *prev = t;
                keep.then_some((t - window_end, d))
            })
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
                .width(1.5_f32),
        );
    });
}

#[derive(Clone, Copy, Debug, Default)]
struct ListenerMandalaLevels {
    attention: Option<f32>,
    stability: Option<f32>,
    uncertainty: Option<f32>,
}

/// Draw listener-side meter and salience state as an at-a-glance mandala.
///
/// `entrain_phases` are the live voices' offset-removed rhythm phases and
/// `entrain_order_r` their Kuramoto order; they are overlaid as dots on the
/// subdivision ring with a mean-resultant vector so voice entrainment reads on
/// the same compass as the listener meter.
pub fn draw_listener_mandala(
    ui: &mut egui::Ui,
    listener: &crate::viewdata::ListenerFrame,
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
        uncertainty: listener
            .prediction
            .forecast
            .and_then(|f| f.entropy_nats)
            .map(|h| (h / 5.0_f32.ln()).clamp(0.0, 1.0)),
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

    if let Some(uncertainty) = listener_levels.uncertainty {
        let uncertainty = uncertainty.clamp(0.0, 1.0);
        if uncertainty > 0.01 {
            let uncertainty_color =
                Color32::from_rgb(255, 95, 65).gamma_multiply(0.18 + 0.72 * uncertainty);
            painter.circle_stroke(
                center,
                outer_r * 1.08,
                Stroke::new(1.0 + 4.0 * uncertainty, uncertainty_color),
            );
            painter.circle_stroke(
                center,
                outer_r * 1.18,
                Stroke::new(
                    0.5 + 2.5 * uncertainty,
                    uncertainty_color.gamma_multiply(0.45),
                ),
            );
        }
    }

    // Beat (tactus): progress arc as the meter with confidence-driven opacity/width.
    let beat_phase = wrap(meter.beat.phase);
    let (d_color, d_weight) = get_visuals(meter.beat.confidence, color_beat);
    painter.circle_stroke(
        center,
        outer_r,
        Stroke::new(1.0_f32, d_color.gamma_multiply(0.2)),
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
    painter.circle_stroke(center, radius, Stroke::new(2.5_f32, base_circle));
    let (t_color, t_weight) = get_visuals(meter.subdivision.confidence, color_sub);
    let sub_angle = start_angle + wrap(meter.subdivision.phase);
    let sub_pos = center + egui::vec2(sub_angle.cos(), sub_angle.sin()) * radius;
    let dot_radius = 2.0 + t_weight * 3.0;
    painter.circle_filled(sub_pos, dot_radius, t_color);
    painter.circle_stroke(
        sub_pos,
        dot_radius + 1.0,
        Stroke::new(1.0_f32, t_color.gamma_multiply(0.5)),
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
        Stroke::new(1.0_f32, center_color.gamma_multiply(0.7)),
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
            painter.line_segment([center, tip], Stroke::new(2.0_f32, voice_color));
            painter.circle_filled(tip, 2.8, voice_color);
        }
    }
}

#[cfg(test)]
mod rhythm_tests {
    use super::*;
    use crate::viewdata::ListenerRhythmHop;

    fn hop(time_sec: f64, phase: f32, flux: f32, confidence: f32) -> ListenerRhythmHop {
        let mut sample = ListenerRhythmHop::default();
        sample.time_sec = time_sec;
        sample.meter.beat.phase = phase * std::f32::consts::TAU;
        sample.meter.beat.confidence = confidence;
        sample.meter.beat.freq_hz = 3.7;
        sample.attention.flux = flux;
        sample
    }

    #[test]
    fn observed_phase_folds_at_wrap_and_exact_beat_boundary() {
        let history = VecDeque::from([
            hop(0.0, -0.125, 0.2, 1.0),
            hop(0.1, 0.0, 0.9, 1.0),
            hop(0.2, 0.25, 0.5, 1.0),
            hop(0.3, 0.875, 0.7, 1.0),
            hop(0.4, 0.0, 0.8, 1.0),
        ]);
        let rows = strobe_rows(&history, 0.4, 10);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].flux[84], Some(0.2));
        assert_eq!(rows[1].flux[0], Some(0.9));
        assert_eq!(rows[1].flux[24], Some(0.5));
        assert_eq!(rows[1].flux[84], Some(0.7));
        assert_eq!(rows[2].flux[0], Some(0.8));
        // Frequency is deliberately inconsistent with the observed phases.
        assert_eq!(rows[2].start_sec, 0.4);
    }

    #[test]
    fn unsupported_span_never_invents_beats_or_phase_bins() {
        let mut history = VecDeque::from([
            hop(0.0, 0.75, 0.4, 1.0),
            hop(0.1, 0.0, 1.0, 0.0),
            hop(2.0, 0.9, 1.0, 0.0),
            hop(4.0, 0.0, 1.0, 0.0),
            hop(4.1, 0.25, 0.8, 1.0),
        ]);
        history[1].meter.subdivision_ratio = 3;
        history[1].meter.subdivision.confidence = 1.0;
        let rows = strobe_rows(&history, 4.1, 10);
        assert_eq!(rows.len(), 3);
        assert!(rows[1].unknown);
        assert!(rows[1].flux.iter().all(Option::is_none));
        assert_eq!(rows[1].subdivision_ratio, None);
        assert_eq!(rows[2].flux[24], Some(0.8));
        assert_eq!(rows[2].flux[0], None);
        history[4].meter.beat.phase = f32::NAN;
        assert!(strobe_rows(&history, 4.1, 10).back().unwrap().unknown);
    }

    #[test]
    fn strobe_retains_visible_rows_within_thirty_seconds_and_hides_future() {
        let history = VecDeque::from([
            hop(0.0, 0.75, 1.0, 1.0),
            hop(1.0, 0.0, 0.5, 1.0),
            hop(30.0, 0.75, 0.4, 1.0),
            hop(30.1, 0.0, 0.7, 1.0),
            hop(30.2, 0.0, 0.8, 1.0),
            hop(30.3, 0.75, 0.9, 1.0),
            hop(30.4, 0.0, 1.0, 1.0),
        ]);
        let rows = strobe_rows(&history, 30.2, 10);
        assert_eq!(rows.front().unwrap().start_sec, 1.0);
        assert_eq!(rows.back().unwrap().flux[0], Some(0.8));
        assert_eq!(rows.back().unwrap().flux[72], None);
        assert_eq!(strobe_rows(&history, 30.4, 2).len(), 2);
    }

    #[test]
    fn subdivision_marks_require_supported_beat_and_ratio() {
        let mut sample = hop(0.0, 0.0, 0.5, 1.0);
        sample.meter.subdivision_ratio = 3;
        assert_eq!(supported_subdivision(&sample.meter), None);
        sample.meter.subdivision.confidence = 0.079;
        assert_eq!(supported_subdivision(&sample.meter), None);
        sample.meter.subdivision.confidence = 0.4;
        assert_eq!(supported_subdivision(&sample.meter), Some(3));
        sample.meter.beat.confidence = 0.0;
        assert_eq!(supported_subdivision(&sample.meter), None);
        sample.meter.beat.confidence = 1.0;
        sample.meter.subdivision_ratio = 0;
        assert_eq!(supported_subdivision(&sample.meter), None);
    }

    #[test]
    fn low_confidence_uses_the_existing_readout_floor_for_unknown_rows() {
        let history = VecDeque::from([
            hop(0.0, 0.5, 1.0, 0.079),
            hop(0.1, 0.75, 1.0, METER_DISPLAY_CONFIDENCE_MIN),
        ]);
        let rows = strobe_rows(&history, 0.1, 10);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].unknown);
        assert!(rows[0].flux.iter().all(Option::is_none));
        assert!(!rows[1].unknown);
    }

    #[test]
    fn unsupported_strobe_row_is_painted_gray() {
        let history = VecDeque::from([hop(0.0, 0.5, 1.0, 1.0), hop(0.1, 0.0, 1.0, 0.0)]);
        let mut output = egui::Context::default().run_ui(Default::default(), |ui| {
            beat_strobe_plot(ui, &history, 128.0, 0.1);
            rhythm_slow_plots(ui, &history, 0.1);
        });
        output.textures_delta.clear();
        assert!(output.shapes.iter().any(|shape| matches!(
            &shape.shape,
            egui::epaint::Shape::Rect(rect) if rect.fill == Color32::from_gray(70)
        )));
    }
}
