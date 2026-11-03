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

/// Display-only extrapolation of the latest supported beat phase to the display
/// time, so the newest row shows where "now" falls inside the current beat. The
/// listener itself is not advanced; this only spans the gap to the last hop.
fn strobe_playhead(
    history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    window_end: f64,
) -> Option<f32> {
    let hop = history
        .iter()
        .rev()
        .find(|hop| hop.time_sec <= window_end)?;
    let beat = hop.meter.beat;
    (beat.confidence >= METER_DISPLAY_CONFIDENCE_MIN
        && beat.phase.is_finite()
        && beat.freq_hz.is_finite()
        && beat.freq_hz > 0.0)
        .then(|| {
            let observed = beat.phase.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
            (observed + beat.freq_hz * (window_end - hop.time_sec) as f32).rem_euclid(1.0)
        })
}

/// One-line key for the strobe: what a row is and the three patterns to look for.
pub(crate) fn beat_strobe_legend(ui: &mut egui::Ui) {
    let small = |text: &str| egui::RichText::new(text).size(10.0).color(Color32::GRAY);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.label(small("1 row = 1 beat, newest below"));
        // Cell columns per icon row: aligned, drifting, scattered.
        for (name, cells) in [
            ("in time", [0, 0, 0]),
            ("drifting", [0, 2, 4]),
            ("no pulse", [4, 0, 3]),
        ] {
            let (icon, _) = ui.allocate_exact_size(Vec2::new(14.0, 10.0), egui::Sense::hover());
            let painter = ui.painter_at(icon);
            painter.rect_filled(icon, 0.0, Color32::from_gray(38));
            for (row, column) in cells.into_iter().enumerate() {
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(
                            icon.left() + column as f32 * 2.5,
                            icon.top() + row as f32 * 3.4,
                        ),
                        Vec2::new(2.5, 3.0),
                    ),
                    0.0,
                    Color32::from_rgb(80, 180, 255),
                );
            }
            ui.label(small(name));
        }
        let (swatch, _) = ui.allocate_exact_size(Vec2::new(10.0, 10.0), egui::Sense::hover());
        ui.painter()
            .rect_filled(swatch, 0.0, Color32::from_gray(70));
        ui.label(small("beat lost"));
    });
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
    let counts: &[&str] = match rows.back().and_then(|row| row.subdivision_ratio) {
        Some(2) => &["0", "½"],
        Some(3) => &["0", "⅓", "⅔"],
        Some(4) => &["0", "¼", "½", "¾"],
        _ => &["0"],
    };
    for (tick, count) in counts.iter().enumerate() {
        painter.text(
            egui::pos2(x(tick as f32 / counts.len() as f32), rect.top() + 2.0),
            if tick == 0 {
                egui::Align2::LEFT_TOP
            } else {
                egui::Align2::CENTER_TOP
            },
            *count,
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
    if let (Some(newest), Some(phase)) = (rows.back(), strobe_playhead(history, window_end))
        && !newest.unknown
    {
        let y = rect.top() + STROBE_AXIS_HEIGHT + (rows.len() - 1) as f32 * STROBE_ROW_HEIGHT;
        painter.line_segment(
            [
                egui::pos2(x(phase), y - 2.0),
                egui::pos2(x(phase), y + STROBE_ROW_HEIGHT + 1.0),
            ],
            Stroke::new(1.5, Color32::WHITE),
        );
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
        ("rhythm_tempo", "Tempo", 40.0, 240.0),
        ("rhythm_ratio", "Subdivision", 34.0, 4.0),
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
                // The current value of each trace, shown at the right ("now") edge.
                let mut current = egui::text::LayoutJob::default();
                for (trace, &(color, name)) in traces.iter().enumerate() {
                    let mut points = Vec::new();
                    let mut now = None;
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
                        now = value.filter(|value| value.is_finite());
                        if let Some(value) = now {
                            points.push([hop.time_sec - window_end, value]);
                        } else if !points.is_empty() {
                            plot_ui.line(Line::new(name, std::mem::take(&mut points)).color(color));
                        }
                    }
                    if !points.is_empty() {
                        plot_ui.line(Line::new(name, points).color(color));
                    }
                    let text = match (panel, now) {
                        (_, None) => "–".to_owned(),
                        (0, Some(bpm)) => format!("{bpm:.0} BPM"),
                        (1, Some(ratio)) => format!("x{ratio:.0}"),
                        (_, Some(confidence)) => format!("{confidence:.2}"),
                    };
                    current.append(
                        &text,
                        if trace == 0 { 0.0 } else { 6.0 },
                        egui::TextFormat::simple(egui::FontId::proportional(11.0), color),
                    );
                }
                plot_ui.text(
                    egui_plot::Text::new("", egui_plot::PlotPoint::new(0.0, upper), current)
                        .anchor(egui::Align2::RIGHT_TOP),
                );
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

// Display tessellation only; this is not a pitch-class or musical grid.
const OCTAVE_BINS: usize = 96;

fn circle_direction(phase: f32) -> Vec2 {
    let angle = phase * std::f32::consts::TAU;
    egui::vec2(angle.sin(), -angle.cos())
}

fn supported_band(band: crate::core::meter::MeterBand) -> bool {
    band.confidence >= METER_DISPLAY_CONFIDENCE_MIN
        && band.phase.is_finite()
        && band.freq_hz.is_finite()
        && band.freq_hz > 0.0
}

/// The listener's measure as a discrete position, shown around the beat clock.
/// The measure ratio can waver (for example 3 and 4 over a 2-against-3 texture),
/// so it never drives the hand.
struct MeasureMarks {
    beats: u8,
    current: u8,
    confidence: f32,
}

struct MeterCircle {
    /// Current beat phase in [0, 1); one turn of the hand is one beat.
    phase: f32,
    subdivision: Option<u8>,
    measure: Option<MeasureMarks>,
    current: [f32; STROBE_BINS],
    earlier: [f32; STROBE_BINS],
}

fn meter_circle(
    history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    window_end: f64,
) -> Option<MeterCircle> {
    let latest = history
        .iter()
        .rev()
        .find(|hop| hop.time_sec <= window_end)?;
    if latest.time_sec < window_end - crate::viewdata::RHYTHM_HISTORY_SEC
        || !supported_band(latest.meter.beat)
    {
        return None;
    }
    // Display-only extrapolation from the latest observation to the display time.
    let advance = |band: crate::core::meter::MeterBand| {
        band.phase / std::f32::consts::TAU + band.freq_hz * (window_end - latest.time_sec) as f32
    };
    let beat_now = advance(latest.meter.beat);
    let measure = (matches!(latest.meter.measure_ratio, 2..=4)
        && supported_band(latest.meter.measure))
    .then(|| {
        let beats = latest.meter.measure_ratio;
        MeasureMarks {
            beats,
            current: ((advance(latest.meter.measure).rem_euclid(1.0) * beats as f32) as u8)
                .min(beats - 1),
            confidence: latest.meter.measure.confidence.clamp(0.0, 1.0),
        }
    });
    let mut circle = MeterCircle {
        phase: beat_now.rem_euclid(1.0),
        subdivision: supported_subdivision(&latest.meter),
        measure,
        current: [0.0; STROBE_BINS],
        earlier: [0.0; STROBE_BINS],
    };
    let mut previous = None;
    for hop in history.iter().filter(|hop| {
        hop.time_sec >= window_end - crate::viewdata::RHYTHM_HISTORY_SEC
            && hop.time_sec <= window_end
    }) {
        if !supported_band(hop.meter.beat) {
            // A lost beat cannot locate earlier flux in the current beat.
            circle.current.fill(0.0);
            circle.earlier.fill(0.0);
            previous = None;
            continue;
        }
        let phase = (hop.meter.beat.phase / std::f32::consts::TAU).rem_euclid(1.0);
        if previous.is_some_and(|prev| phase - prev < -0.5) {
            for (old, recent) in circle.earlier.iter_mut().zip(&mut circle.current) {
                *old = old.max(*recent);
                *recent = 0.0;
            }
        }
        let bin = ((phase * STROBE_BINS as f32) as usize).min(STROBE_BINS - 1);
        if hop.attention.flux.is_finite() {
            circle.current[bin] = circle.current[bin].max(hop.attention.flux.max(0.0));
        }
        previous = Some(phase);
    }
    // Extrapolation may already have crossed into an as-yet unobserved beat.
    if (latest.meter.beat.phase / std::f32::consts::TAU).rem_euclid(1.0)
        + latest.meter.beat.freq_hz * (window_end - latest.time_sec) as f32
        >= 1.0
    {
        for (old, recent) in circle.earlier.iter_mut().zip(&mut circle.current) {
            *old = old.max(*recent);
            *recent = 0.0;
        }
    }
    Some(circle)
}

pub(crate) fn draw_meter_circle(
    ui: &mut egui::Ui,
    history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    window_end: f64,
    side: f32,
) {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), egui::Sense::hover());
    response.on_hover_text("Presentation meter. One turn of the hand is one beat. Inner marks: subdivision. Bright blue: auditory flux in the current beat; dim blue: earlier beats in the retained 30 s, both normalized to the visible maximum. Outer segments: the beats of the listener's measure, the current one lit, opacity from measure confidence; gray when no measure is supported. A changing segment count shows the listener wavering between measure lengths.");
    let painter = ui.painter_at(rect);
    // The title sits in the square's corner, outside the ring, to save a row.
    painter.text(
        rect.left_top(),
        egui::Align2::LEFT_TOP,
        "Meter",
        egui::FontId::proportional(10.0),
        ui.visuals().text_color(),
    );
    let center = rect.center();
    let radius = side * 0.36;
    let ring_inner = side * 0.40;
    let ring_outer = side * 0.47;
    let gray = ui.visuals().weak_text_color();
    let blue = Color32::from_rgb(90, 180, 230);
    let red = Color32::from_rgb(255, 110, 90);
    painter.circle_stroke(center, radius, Stroke::new(1.0, gray));
    let circle = meter_circle(history, window_end);
    if let Some(circle) = &circle {
        let subdivisions = circle.subdivision.unwrap_or(1) as usize;
        for i in 0..subdivisions {
            let direction = circle_direction(i as f32 / subdivisions as f32);
            painter.line_segment(
                [
                    center + direction * radius,
                    center + direction * (radius - if i == 0 { 7.0 } else { 4.0 }),
                ],
                Stroke::new(if i == 0 { 3.0 } else { 1.0 }, gray),
            );
        }
        let maximum = circle
            .current
            .iter()
            .chain(&circle.earlier)
            .copied()
            .fold(0.0_f32, f32::max);
        if maximum > 0.0 {
            for (values, color, width) in [
                (&circle.earlier, blue.gamma_multiply(0.3), 1.0),
                (&circle.current, blue, 2.0),
            ] {
                for (i, flux) in values.iter().enumerate().filter(|(_, flux)| **flux > 0.0) {
                    let direction = circle_direction(i as f32 / STROBE_BINS as f32);
                    painter.line_segment(
                        [
                            center + direction * radius,
                            center + direction * (radius - radius * 0.35 * flux / maximum),
                        ],
                        Stroke::new(width, color),
                    );
                }
            }
        }
    }
    // Measure segments around the beat clock.
    let measure = circle.as_ref().and_then(|c| c.measure.as_ref());
    let segments = measure.map_or(1, |m| m.beats as usize);
    for segment in 0..segments {
        let (color, width) = match measure {
            // Same confidence-to-opacity mapping as the meter's other displays.
            Some(m) => (
                red.gamma_multiply(
                    (0.3 + 0.7 * m.confidence)
                        * if segment == m.current as usize {
                            1.0
                        } else {
                            0.25
                        },
                ),
                if segment == m.current as usize {
                    4.0
                } else {
                    2.0
                },
            ),
            None => (gray.gamma_multiply(0.5), 2.0),
        };
        let gap = if segments > 1 { 0.03 } else { 0.0 };
        let steps = 24;
        let points: Vec<egui::Pos2> = (0..=steps)
            .map(|k| {
                let t = (segment as f32 + gap + (1.0 - 2.0 * gap) * k as f32 / steps as f32)
                    / segments as f32;
                center + circle_direction(t) * (ring_inner + ring_outer) * 0.5
            })
            .collect();
        painter.add(egui::Shape::line(points, Stroke::new(width, color)));
    }
    painter.arrow(
        center,
        circle_direction(circle.as_ref().map_or(0.0, |c| c.phase)) * radius * 0.86,
        Stroke::new(
            1.5,
            if circle.is_some() {
                Color32::WHITE
            } else {
                gray
            },
        ),
    );
    painter.text(
        center + egui::vec2(0.0, radius * 0.45),
        egui::Align2::CENTER_CENTER,
        match (&circle, measure) {
            (None, _) => "beat lost",
            (Some(_), None) => "measure ?",
            (Some(_), Some(_)) => "1 turn = 1 beat",
        },
        egui::FontId::proportional(9.0),
        gray,
    );
}

fn octave_bin(log2: f64) -> usize {
    (log2.rem_euclid(1.0) * OCTAVE_BINS as f64).round() as usize % OCTAVE_BINS
}

fn fold_octaves(samples: impl Iterator<Item = (f64, f64)>) -> [f64; OCTAVE_BINS] {
    let mut folded = [0.0; OCTAVE_BINS];
    for (log2, weight) in samples {
        if log2.is_finite() && weight.is_finite() && weight > 0.0 {
            let bin = octave_bin(log2);
            folded[bin] += weight;
        }
    }
    folded
}

fn circular_mean(folded: &[f64; OCTAVE_BINS]) -> Option<Vec2> {
    let mass: f64 = folded.iter().sum();
    if mass <= 0.0 {
        return None;
    }
    let mut mean = [0.0; 2];
    for (i, weight) in folded.iter().enumerate() {
        let angle = i as f64 * std::f64::consts::TAU / OCTAVE_BINS as f64;
        mean[0] += angle.sin() * weight / mass;
        mean[1] -= angle.cos() * weight / mass;
    }
    let mean = egui::vec2(mean[0] as f32, mean[1] as f32);
    Some(mean / mean.length().max(1.0))
}

fn folded_support(support: &crate::core::harmonic_support::Snapshot) -> [f64; OCTAVE_BINS] {
    let start = f64::from(support.geometry.root_start_log2);
    let step = f64::from(support.geometry.space.step());
    fold_octaves(
        support
            .p
            .iter()
            .enumerate()
            .map(|(i, &p)| (start + i as f64 * step, p)),
    )
}

fn harmonic_marks(root: f32) -> [f32; 4] {
    [1.0_f32, 3.0, 5.0, 7.0].map(|harmonic| (root + harmonic.log2()).rem_euclid(1.0))
}

pub(crate) fn observe_harmony(
    history: &mut VecDeque<(f64, Option<Vec2>)>,
    frame: &crate::viewdata::UiFrame,
) {
    let time = frame.meta.time_sec;
    if history.back().is_some_and(|&(last, _)| time < last) {
        history.clear();
    }
    if history.back().is_some_and(|&(last, _)| time == last) {
        history.pop_back();
    }
    let mean = frame
        .landscape
        .support
        .as_deref()
        .and_then(|s| circular_mean(&folded_support(s)));
    history.push_back((time, mean));
    while history
        .front()
        .is_some_and(|&(t, _)| t < time - super::windows::TIME_WINDOW_SEC)
    {
        history.pop_front();
    }
}

pub(crate) fn draw_harmony_circle(
    ui: &mut egui::Ui,
    landscape: &crate::core::landscape::Landscape,
    history: &VecDeque<(f64, Option<Vec2>)>,
    side: f32,
) {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.text(
        rect.left_top(),
        egui::Align2::LEFT_TOP,
        "Harmony · habitat",
        egui::FontId::proportional(10.0),
        ui.visuals().text_color(),
    );
    let center = rect.center();
    let radius = side * 0.35;
    let gray = ui.visuals().weak_text_color();
    let blue = Color32::from_rgb(90, 180, 230);
    let gold = Color32::from_rgb(235, 200, 75);
    painter.circle_stroke(center, radius, Stroke::new(1.0, gray));
    for (phase, label) in [(0.0, "0"), (0.25, "¼"), (0.5, "½"), (0.75, "¾")] {
        painter.text(
            center + circle_direction(phase) * radius * 1.25,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(10.0),
            gray,
        );
    }
    let heard = fold_octaves(
        landscape
            .space
            .centers_log2
            .iter()
            .zip(&landscape.subjective_intensity)
            .map(|(&f, &v)| (f64::from(f), f64::from(v))),
    );
    let maximum = heard.iter().copied().fold(0.0_f64, f64::max);
    if maximum > 0.0 {
        for (i, strength) in heard.iter().enumerate().filter(|(_, v)| **v > 0.0) {
            let direction = circle_direction(i as f32 / OCTAVE_BINS as f32);
            painter.line_segment(
                [
                    center + direction * radius * 0.86,
                    center + direction * radius * (0.86 + 0.14 * (strength / maximum) as f32),
                ],
                Stroke::new(2.0, blue),
            );
        }
    }
    let support = landscape.support.as_deref();
    if let Some(support) = support {
        let folded = folded_support(support);
        let maximum = folded.iter().copied().fold(0.0_f64, f64::max);
        if maximum > 0.0 {
            for (i, strength) in folded.iter().enumerate().filter(|(_, v)| **v > 0.0) {
                let direction = circle_direction(i as f32 / OCTAVE_BINS as f32);
                painter.line_segment(
                    [
                        center + direction * radius * 0.15,
                        center + direction * radius * (0.15 + 0.58 * (strength / maximum) as f32),
                    ],
                    Stroke::new(2.0, gold.gamma_multiply(0.65)),
                );
            }
            let strongest = folded
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            for phase in harmonic_marks(strongest as f32 / OCTAVE_BINS as f32) {
                painter.circle_filled(center + circle_direction(phase) * radius * 1.06, 1.8, gold);
            }
        }
        let mut previous = None;
        for &(time, mean) in history {
            if let Some(mean) = mean {
                let tip = center + mean * radius * 0.73;
                if let Some(prev) = previous {
                    let age = history.back().map_or(0.0, |&(now, _)| now - time);
                    painter.line_segment(
                        [prev, tip],
                        Stroke::new(
                            1.0,
                            gold.gamma_multiply(
                                (1.0 - age / super::windows::TIME_WINDOW_SEC).max(0.0) as f32 * 0.6,
                            ),
                        ),
                    );
                }
                previous = Some(tip);
            } else {
                previous = None;
            }
        }
        if let Some(mean) = circular_mean(&folded) {
            painter.arrow(
                center,
                mean * radius * 0.73,
                Stroke::new(1.5, Color32::WHITE),
            );
        }
    } else {
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            "no support data",
            egui::FontId::proportional(9.0),
            gray,
        );
    }
    response.on_hover_ui(|ui| {
        ui.label("habitat · clockwise frac(log2 Hz); top = 2ⁿ Hz. Outer blue: heard density. Inner gold: support p. Each is scaled to its own maximum.");
        if let Some(pointer) = ui.ctx().pointer_hover_pos() {
            let offset = pointer - center;
            let fraction = offset.x.atan2(-offset.y).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
            let bin = (fraction * OCTAVE_BINS as f32).round() as usize % OCTAVE_BINS;
            let strongest = landscape.space.centers_log2.iter().zip(&landscape.subjective_intensity)
                .enumerate().filter(|(_, (log2, v))| {
                    v.is_finite() && **v > 0.0 && (octave_bin(f64::from(**log2)) == bin)
                }).max_by(|a, b| (a.1).1.total_cmp((b.1).1));
            ui.label(format!("Angle: {fraction:.3} octave"));
            if let Some((i, _)) = strongest {
                ui.label(format!("Strongest heard component here: {:.1} Hz", landscape.space.centers_hz[i]));
            } else {
                ui.label("No heard component here");
            }
        }
        if let Some(support) = support {
            let length = circular_mean(&folded_support(support)).map_or(0.0, |m| m.length());
            ui.label(format!("C = {:.3} · D = {:.3} · M = {:.3} · resultant = {length:.3}", support.focus, support.conflict, support.mass));
        } else {
            ui.label("no support data");
        }
        ui.label("Arrow: circular mean of folded p; length = resultant (0–1), not C. Trail: last 5 s of received frames.");
        ui.label("Harmonic marks: positions of the harmonic series used by the H model (harmonics 1–8, octave duplicates merged). Inharmonic timbre partials can fall outside these marks even for a single sound source.");
    });
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
    fn meter_circle_turns_with_the_beat_and_shows_the_measure_as_marks() {
        let measure_hop = |time, beat_phase, measure_phase: f32, ratio| {
            let mut sample = hop(time, beat_phase, 0.8, 1.0);
            sample.meter.measure_ratio = ratio;
            sample.meter.measure.confidence = 1.0;
            sample.meter.measure.freq_hz = 0.5;
            sample.meter.measure.phase = measure_phase * std::f32::consts::TAU;
            sample
        };
        let mut history = VecDeque::from([
            measure_hop(0.0, 0.875, 0.1, 3),
            measure_hop(0.1, 0.0, 0.2, 4),
            measure_hop(0.2, 0.25, 0.6, 4),
        ]);
        let circle = meter_circle(&history, 0.2).unwrap();
        assert!((circle.phase - 0.25).abs() < 1e-5);
        // The measure ratio flips 3 -> 4, yet the beat-folded flux and hand persist.
        assert_eq!(circle.earlier[84], 0.8);
        assert_eq!(circle.current[0], 0.8);
        assert_eq!(circle.current[24], 0.8);
        let marks = circle.measure.as_ref().unwrap();
        assert_eq!((marks.beats, marks.current), (4, 2));
        // The hand advances from the latest hop at the beat rate (hop() uses 3.7 Hz).
        let ahead = meter_circle(&history, 0.25).unwrap();
        assert!((ahead.phase - 0.435).abs() < 1e-4, "{}", ahead.phase);
        // An unsupported measure keeps the beat clock and drops only the marks.
        history.back_mut().unwrap().meter.measure.confidence = 0.079;
        let unmeasured = meter_circle(&history, 0.2).unwrap();
        assert!(unmeasured.measure.is_none());
        assert_eq!(unmeasured.current[24], 0.8);
        history.back_mut().unwrap().meter.measure.confidence = 1.0;
        history.back_mut().unwrap().meter.measure_ratio = 0;
        assert!(meter_circle(&history, 0.2).unwrap().measure.is_none());
        // A lost beat draws no clock; hops outside the retained window are ignored.
        history.back_mut().unwrap().meter.beat.confidence = 0.079;
        assert!(meter_circle(&history, 0.2).is_none());
        assert!(meter_circle(&history, -32.0).is_none());
        assert!(meter_circle(&history, 31.0).is_none());
    }

    #[test]
    fn meter_reacquisition_and_extrapolated_wrap_do_not_reuse_current_flux() {
        let mut history = VecDeque::from([
            hop(0.0, 0.5, 1.0, 1.0),
            hop(0.1, 0.75, 1.0, 0.0),
            hop(0.2, 0.875, 0.5, 1.0),
        ]);
        let reacquired = meter_circle(&history, 0.2).unwrap();
        assert_eq!(reacquired.current[48], 0.0);
        assert_eq!(reacquired.current[84], 0.5);
        let wrapped = meter_circle(&history, 0.3).unwrap();
        assert!(wrapped.current.iter().all(|&v| v == 0.0));
        assert_eq!(wrapped.earlier[84], 0.5);
        history.back_mut().unwrap().meter.beat.phase = f32::NAN;
        assert!(meter_circle(&history, 0.2).is_none());
    }

    #[test]
    fn support_octaves_use_the_root_grid_origin_and_resultant_is_bounded() {
        let space = crate::core::log2space::Log2Space::new(55.0, 1760.0, 24);
        let kernel =
            crate::core::harmonicity_kernel::HarmonicityKernel::new(&space, Default::default());
        let geometry = crate::core::harmonic_support::Geometry::new(&space, &kernel);
        let mut support = crate::core::harmonic_support::Snapshot::new(geometry);
        support.p[0] = 2.0;
        support.p[24] = 3.0;
        let folded = folded_support(&support);
        let bin = (f64::from(support.geometry.root_start_log2).rem_euclid(1.0) * OCTAVE_BINS as f64)
            .round() as usize
            % OCTAVE_BINS;
        assert_eq!(folded[bin], 5.0);
        assert_eq!(folded.iter().filter(|&&p| p > 0.0).count(), 1);
        let mean = circular_mean(&folded).unwrap();
        assert!((mean.length() - 1.0).abs() < 1e-6);
        assert!((mean - circle_direction(bin as f32 / OCTAVE_BINS as f32)).length() < 1e-6);
        assert!(circular_mean(&[1.0; OCTAVE_BINS]).unwrap().length() < 1e-6);
        assert!(circular_mean(&[0.0; OCTAVE_BINS]).is_none());
        let folded = fold_octaves([(8.25, 1.0), (9.25, 2.0), (-0.75, 3.0)].into_iter());
        assert_eq!(folded[24], 6.0);
    }

    #[test]
    fn harmonic_marks_are_the_unique_octave_folds_of_harmonics_one_through_eight() {
        let root = 0.83;
        let marks = harmonic_marks(root);
        for harmonic in 1..=8 {
            let phase = (root + (harmonic as f32).log2()).rem_euclid(1.0);
            assert!(marks.iter().any(|&m| (m - phase).abs() < 1e-6));
        }
        assert!((marks[0] - root).abs() < 1e-6);
        assert!((harmonic_marks(0.0)[1] - 3.0_f32.log2().fract()).abs() < 1e-6);
        assert_eq!(circle_direction(0.0), egui::vec2(0.0, -1.0));
        assert!((circle_direction(0.25) - egui::vec2(1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn harmony_trail_is_frame_driven_bounded_and_keeps_missing_support_gaps() {
        let mut frame = crate::viewdata::UiFrame::default();
        let space = crate::core::log2space::Log2Space::new(55.0, 1760.0, 24);
        let kernel =
            crate::core::harmonicity_kernel::HarmonicityKernel::new(&space, Default::default());
        let mut support = crate::core::harmonic_support::Snapshot::new(
            crate::core::harmonic_support::Geometry::new(&space, &kernel),
        );
        support.p[0] = 1.0;
        frame.landscape.support = Some(std::sync::Arc::new(support));
        let mut history = VecDeque::new();
        observe_harmony(&mut history, &frame);
        frame.meta.time_sec = 1.0;
        observe_harmony(&mut history, &frame);
        frame.landscape.support = None;
        frame.meta.time_sec = 2.0;
        observe_harmony(&mut history, &frame);
        assert!(history[0].1.is_some());
        assert_eq!(history[2].1, None);
        observe_harmony(&mut history, &frame);
        assert_eq!(history.len(), 3);
        frame.meta.time_sec = 6.0;
        observe_harmony(&mut history, &frame);
        assert_eq!(history.front().unwrap().0, 1.0);
        frame.meta.time_sec = 0.0;
        observe_harmony(&mut history, &frame);
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn unsupported_circles_show_explicit_missing_labels() {
        let frame = crate::viewdata::UiFrame::default();
        let mut output = egui::Context::default().run_ui(Default::default(), |ui| {
            draw_meter_circle(ui, &VecDeque::new(), 0.0, 108.0);
            draw_harmony_circle(ui, &frame.landscape, &VecDeque::new(), 108.0);
        });
        output.textures_delta.clear();
        for label in ["beat lost", "no support data"] {
            assert!(output.shapes.iter().any(|shape| matches!(&shape.shape,
                egui::epaint::Shape::Text(text) if text.galley.text() == label)));
        }
    }

    #[test]
    fn playhead_advances_from_the_latest_hop_and_wraps_into_the_beat() {
        // hop() uses a 3.7 Hz beat; 0.1 s later the phase has moved by 0.37 beats.
        let history = VecDeque::from([hop(1.0, 0.75, 0.0, 1.0), hop(2.0, 0.5, 0.0, 1.0)]);
        let ahead = strobe_playhead(&history, 2.1).unwrap();
        assert!((ahead - 0.87).abs() < 1e-4, "{ahead}");
        let wrapped = strobe_playhead(&history, 2.2).unwrap();
        assert!((wrapped - 0.24).abs() < 1e-4, "{wrapped}");
        // Only hops at or before the display time are used.
        let earlier = strobe_playhead(&history, 1.1).unwrap();
        assert!((earlier - 0.12).abs() < 1e-4, "{earlier}");
        assert_eq!(strobe_playhead(&history, 0.5), None);
        let lost = VecDeque::from([hop(2.0, 0.5, 0.0, 0.0)]);
        assert_eq!(strobe_playhead(&lost, 2.1), None);
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
    fn strobe_labels_use_ratios_without_musical_counts() {
        let mut sample = hop(0.0, 0.5, 1.0, 1.0);
        sample.meter.subdivision_ratio = 3;
        sample.meter.subdivision.confidence = 1.0;
        let history = VecDeque::from([sample]);
        let mut output = egui::Context::default().run_ui(Default::default(), |ui| {
            beat_strobe_plot(ui, &history, 128.0, 0.0);
        });
        output.textures_delta.clear();
        for label in ["0", "⅓", "⅔"] {
            assert!(output.shapes.iter().any(|shape| matches!(&shape.shape,
                egui::epaint::Shape::Text(text) if text.galley.text() == label)));
        }
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
