use crate::core::db::amp_to_db;
use crate::ui::plots::{
    draw_listener_mandala, draw_roughness_harmonicity, log2_plot_hz, neural_activity_plot,
    plot_population_dynamics, spectrum_time_freq_axes, time_plot,
};
use crate::ui::viewdata::{PlaybackState, UiFrame};
use egui::{CentralPanel, Color32, Panel, Vec2};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn consonance_field_score_axis_bounds(
    kernel: crate::core::consonance_kernel::ConsonanceKernel,
) -> (f64, f64) {
    let mut y_min = -1.0f32;
    let mut y_max = 1.0f32;

    // Bound from kernel corners instead of live scan values, so the axis does not jitter.
    for score in [
        kernel.score(0.0, 0.0),
        kernel.score(1.0, 0.0),
        kernel.score(0.0, 1.0),
        kernel.score(1.0, 1.0),
    ] {
        if score.is_finite() {
            y_min = y_min.min(score);
            y_max = y_max.max(score);
        }
    }

    if (y_max - y_min).abs() < 1e-6 {
        y_min -= 0.5;
        y_max += 0.5;
    }

    let pad = ((y_max - y_min).abs() * 0.05).max(0.1);
    let step = 0.5;
    y_min = ((y_min - pad) / step).floor() * step;
    y_max = ((y_max + pad) / step).ceil() * step;

    if y_max <= y_min {
        y_max = y_min + step;
    }

    (y_min as f64, y_max as f64)
}

fn format_time(sec: f32) -> String {
    let total_secs = sec.max(0.0).floor() as u64;
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    format!("{minutes:02}:{seconds:02}")
}

fn draw_level_meters(
    ui: &mut egui::Ui,
    left_inst: f32,
    left_win: f32,
    right_inst: f32,
    right_win: f32,
    meter_height: f32,
    meter_width_scale: f32,
) {
    let meter_height = meter_height.max(42.0);
    let label_width = 20.0;
    let meter_width = 22.0 * meter_width_scale;
    let spacing = 8.0;
    let pad_y = 4.0;
    let total_width = label_width + spacing + meter_width * 2.0 + spacing;
    let (rect, _resp) =
        ui.allocate_exact_size(Vec2::new(total_width, meter_height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let window_stroke_color = ui.visuals().window_stroke().color;
    let text_color = ui.visuals().text_color();
    let min_db = -60.0;

    let inner = rect.shrink2(Vec2::new(0.0, pad_y));
    let to_height = |db: f32| ((db - min_db) / -min_db).clamp(0.0, 1.0) * inner.height();
    let color_for = |amp: f32| match amp_to_db(amp) {
        db if db >= 0.0 => egui::Color32::RED,
        db if db >= -6.0 => egui::Color32::YELLOW,
        _ => egui::Color32::GREEN,
    };

    // Tick labels
    for &db in &[0.0, -6.0, -12.0, -30.0, -60.0] {
        let h = to_height(db);
        let y = inner.bottom() - h;
        painter.text(
            egui::pos2(rect.left() + label_width - 2.0, y),
            egui::Align2::RIGHT_CENTER,
            format!("{db:.0}"),
            egui::FontId::proportional(11.0),
            ui.visuals().text_color(),
        );
    }

    let draw_meter = |p: &egui::Painter, base: egui::Pos2, inst: f32, win: f32, label: &str| {
        let inst_db = amp_to_db(inst).max(min_db);
        let win_db = amp_to_db(win).max(min_db);
        let inst_h = to_height(inst_db);
        let win_h = to_height(win_db);

        let m_rect = egui::Rect::from_min_size(
            base + egui::vec2(0.0, pad_y),
            Vec2::new(meter_width, inner.height()),
        );
        p.rect_stroke(
            m_rect,
            2.0,
            egui::Stroke::new(1.0, window_stroke_color),
            egui::StrokeKind::Inside,
        );
        let inst_rect = egui::Rect::from_min_size(
            egui::pos2(m_rect.left() + 3.0, m_rect.bottom() - inst_h),
            Vec2::new(meter_width - 6.0, inst_h),
        );
        p.rect_filled(inst_rect, 0.0, color_for(inst));

        let win_y = m_rect.bottom() - win_h;
        p.line_segment(
            [
                egui::pos2(m_rect.left() + 2.5, win_y),
                egui::pos2(m_rect.right() - 2.5, win_y),
            ],
            egui::Stroke::new(2.0, color_for(win)),
        );

        p.text(
            m_rect.center_top() + egui::vec2(0.0, 4.0),
            egui::Align2::CENTER_TOP,
            label,
            egui::FontId::proportional(12.0),
            text_color,
        );
    };

    let left_origin = rect.left_top() + egui::vec2(label_width + spacing, 0.0);
    let right_origin = left_origin + egui::vec2(meter_width + spacing, 0.0);
    draw_meter(&painter, left_origin, left_inst, left_win, "L");
    draw_meter(&painter, right_origin, right_inst, right_win, "R");
}

fn split_widths(
    ui: &egui::Ui,
    ratio: f32,
    min_left: f32,
    max_left: f32,
    min_right: f32,
) -> (f32, f32) {
    let sep = ui.spacing().item_spacing.x;
    let available = ui.available_width();
    let left_target = (available * ratio).clamp(min_left, max_left);
    let left_width = left_target.min((available - min_right - sep).max(0.0));
    let right_width = (available - left_width - sep).max(0.0);
    (left_width, right_width)
}

fn fixed_text_label(ui: &mut egui::Ui, text: impl Into<String>, width: f32, color: Color32) {
    ui.allocate_ui_with_layout(
        Vec2::new(width, 16.0),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.label(
                egui::RichText::new(text.into())
                    .monospace()
                    .size(11.0)
                    .color(color),
            );
        },
    );
}

fn mini_fixed_text_label(ui: &mut egui::Ui, text: impl Into<String>, width: f32, color: Color32) {
    ui.allocate_ui_with_layout(
        Vec2::new(width, 11.0),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.label(
                egui::RichText::new(text.into())
                    .monospace()
                    .size(9.0)
                    .color(color),
            );
        },
    );
}

fn mandala_state_meter(ui: &mut egui::Ui, label: &str, value: Option<f32>, color: Color32) {
    let level = value.unwrap_or(0.0).clamp(0.0, 1.0);
    let text = if value.is_some() {
        format!("{level:>4.2}")
    } else {
        "--.--".to_string()
    };
    ui.vertical(|ui| {
        ui.label(egui::RichText::new(label).size(9.0).color(color));
        ui.horizontal(|ui| {
            let desired = Vec2::new(42.0, 8.0);
            let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
            painter.rect_stroke(
                rect,
                2.0,
                ui.visuals().window_stroke(),
                egui::StrokeKind::Inside,
            );
            let fill_rect = egui::Rect::from_min_max(
                rect.left_top(),
                egui::pos2(rect.left() + rect.width() * level, rect.bottom()),
            )
            .shrink2(Vec2::splat(1.0));
            painter.rect_filled(fill_rect, 1.0, color.gamma_multiply(0.75));
            resp.on_hover_text(label);

            mini_fixed_text_label(ui, text, 29.0, color);
        });
    });
}

fn draw_mandala_state_readout(ui: &mut egui::Ui, frame: &UiFrame) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        mandala_state_meter(
            ui,
            "Attention",
            frame
                .listener
                .has_fast_state
                .then_some(frame.listener.attention_level),
            Color32::from_rgb(90, 180, 230),
        );
        mandala_state_meter(
            ui,
            "Stability",
            frame
                .listener
                .has_state
                .then_some(frame.listener.stability_level),
            Color32::from_rgb(80, 200, 130),
        );
        mandala_state_meter(
            ui,
            "Resolvability",
            frame
                .listener
                .has_state
                .then_some(frame.listener.resolvability_level),
            Color32::from_rgb(235, 200, 75),
        );
        mandala_state_meter(
            ui,
            "Tension",
            frame
                .listener
                .has_state
                .then_some(frame.listener.tension_level),
            Color32::from_rgb(235, 95, 85),
        );
    });
}

fn meter_band_label(name: &str, band: crate::core::meter::MeterBand) -> String {
    let confidence = band.confidence.clamp(0.0, 1.0);
    let freq_text = if confidence < 0.08 {
        "--.--".to_string()
    } else {
        let freq_hz = if band.freq_hz.is_finite() {
            band.freq_hz.clamp(0.0, 99.99)
        } else {
            0.0
        };
        format!("{freq_hz:05.2}")
    };
    format!("{name} {freq_text}Hz c{confidence:>4.2}")
}

fn meter_ratio_labels(meter: &crate::core::meter::MeterState) -> (String, String) {
    let sub = if meter.subdivision_ratio > 0 {
        format!("Sub ratio  x{}", meter.subdivision_ratio)
    } else {
        "Sub ratio  x--".to_string()
    };
    let meas = if meter.measure_ratio > 0 {
        let conf = meter.measure.confidence.clamp(0.0, 1.0);
        format!("Measure x{} c{conf:>4.2}", meter.measure_ratio)
    } else {
        "Measure x--".to_string()
    };
    (sub, meas)
}

fn rhythm_label_width(label: &str) -> f32 {
    label.chars().count() as f32 * 7.0 + 4.0
}

fn fixed_rhythm_label(ui: &mut egui::Ui, text: String, color: Color32) {
    let width = rhythm_label_width(&text);
    fixed_text_label(ui, text, width, color);
}

fn fixed_system_label(ui: &mut egui::Ui, text: String) {
    let color = ui.visuals().text_color();
    let width = rhythm_label_width(&text);
    fixed_text_label(ui, text, width, color);
}

fn peak_level_text(peak_level: f32) -> String {
    if peak_level > 0.0 {
        format!("{:>6.1} dB", 20.0 * peak_level.log10())
    } else {
        "  -inf dB".to_string()
    }
}

fn peak_level_color(ui: &egui::Ui, peak_level: f32) -> Color32 {
    if peak_level > 1.0 {
        egui::Color32::RED
    } else {
        ui.visuals().text_color()
    }
}

fn draw_system_diagnostics(ui: &mut egui::Ui, frame: &UiFrame) {
    ui.horizontal_wrapped(|ui| {
        ui.heading("System");
        ui.separator();
        fixed_system_label(ui, format!("listener_t={:07.2}s", frame.listener.time_sec));
        ui.separator();
        fixed_system_label(
            ui,
            format!(
                "frames gen={:06} ana={:06} lag={:04}",
                frame.listener.generated_frame_id,
                frame.listener.analysis_frame_id,
                frame.listener.analysis_lag_frames
            ),
        );
        ui.separator();
        fixed_system_label(ui, format!("events={:04}", frame.meta.event_queue_len));
    });
}

fn draw_kuramoto_order(ui: &mut egui::Ui, frame: &UiFrame) {
    if let Some(r) = frame.meta.kuramoto_order_r {
        let r = r.clamp(0.0, 1.0);
        ui.separator();
        ui.label("Kuramoto R");
        ui.add(
            egui::ProgressBar::new(r)
                .desired_width(110.0)
                .text(format!("{r:.2}  N={}", frame.meta.kuramoto_active_count)),
        );
    }
}

fn draw_listener_dashboard(
    ui: &mut egui::Ui,
    frame: &UiFrame,
    rhythm_history: &VecDeque<(f64, crate::core::meter::MeterState)>,
    dorsal_history: &VecDeque<(f64, crate::ui::viewdata::DorsalFrame)>,
    window_start: f64,
    window_end: f64,
    time_link_id: &str,
) {
    let (left_width, _) = split_widths(ui, 0.10, 200.0, 220.0, 640.0);
    let dashboard_height = 232.0;
    let audio_height = 60.0;
    let plot_height = 82.0;
    let neural_height = 82.0;
    let mandala_side = 104.0;

    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            Vec2::new(left_width, dashboard_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Audio");
                    ui.separator();
                    ui.label("Peak level:");
                    fixed_text_label(
                        ui,
                        peak_level_text(frame.meta.peak_level),
                        68.0,
                        peak_level_color(ui, frame.meta.peak_level),
                    );
                });
                ui.allocate_ui_with_layout(
                    Vec2::new(left_width, audio_height),
                    egui::Layout::left_to_right(egui::Align::Min),
                    |ui| {
                        draw_level_meters(
                            ui,
                            frame.meta.channel_peak[0],
                            frame.meta.window_peak[0],
                            frame.meta.channel_peak[1],
                            frame.meta.window_peak[1],
                            audio_height,
                            0.45,
                        );

                        ui.separator();
                        let available_wave_width = ui.available_width().max(0.0);
                        let wave_width = (available_wave_width * 0.5)
                            .max(120.0)
                            .min(available_wave_width);
                        ui.allocate_ui_with_layout(
                            Vec2::new(wave_width, audio_height),
                            egui::Layout::top_down(egui::Align::LEFT),
                            |ui| {
                                time_plot(
                                    ui,
                                    "Current Hop Wave",
                                    frame.wave.fs as f64,
                                    frame.wave.samples.as_ref(),
                                    audio_height,
                                    false,
                                );
                            },
                        );
                    },
                );

                ui.separator();
                let mandala_area_height = ui.available_height();
                ui.allocate_ui_with_layout(
                    Vec2::new(left_width, mandala_area_height),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        ui.heading("Neural Mandala");
                        let bottom_gap = (ui.available_height() - mandala_side).max(0.0);
                        ui.add_space(bottom_gap);
                        ui.horizontal_top(|ui| {
                            draw_listener_mandala(
                                ui,
                                &frame.listener,
                                &frame.meta.entrain_phases,
                                frame.meta.kuramoto_order_r,
                                Vec2::splat(mandala_side),
                            );
                            draw_mandala_state_readout(ui, frame);
                        });
                    },
                );
            },
        );

        ui.separator();
        let right_width = ui.available_width().max(0.0);
        ui.allocate_ui_with_layout(
            Vec2::new(right_width, dashboard_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                ui.heading("Listener Twin");
                let old_spacing = ui.spacing().item_spacing;
                ui.spacing_mut().item_spacing.y = 0.0;
                ui.label("Auditory salience");
                spectrum_time_freq_axes(
                    ui,
                    dorsal_history,
                    plot_height,
                    window_start,
                    window_end,
                    Some(time_link_id),
                );
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Neural Rhythm");
                    if frame.listener.has_fast_state {
                        let meter = frame.listener.meter;
                        let (sub_label, measure_label) = meter_ratio_labels(&meter);
                        ui.separator();
                        fixed_rhythm_label(
                            ui,
                            meter_band_label("Beat", meter.beat),
                            egui::Color32::from_rgb(80, 180, 255),
                        );
                        fixed_rhythm_label(
                            ui,
                            meter_band_label("Sub", meter.subdivision),
                            egui::Color32::from_rgb(70, 225, 135),
                        );
                        fixed_rhythm_label(ui, sub_label, egui::Color32::from_rgb(255, 215, 60));
                        fixed_rhythm_label(
                            ui,
                            measure_label,
                            egui::Color32::from_rgb(255, 110, 90),
                        );
                    }
                });
                neural_activity_plot(
                    ui,
                    rhythm_history,
                    neural_height,
                    window_start,
                    window_end,
                    Some(time_link_id),
                );
                ui.spacing_mut().item_spacing = old_spacing;
            },
        );
    });
}

/// === Main window ===
#[allow(clippy::too_many_arguments)]
pub fn main_window(
    root_ui: &mut egui::Ui,
    frame: &UiFrame,
    rhythm_history: &VecDeque<(f64, crate::core::meter::MeterState)>,
    dorsal_history: &VecDeque<(f64, crate::ui::viewdata::DorsalFrame)>,
    audio_error: Option<&str>,
    exit_flag: &Arc<AtomicBool>,
    start_flag: &Arc<AtomicBool>,
    show_raw_nsgt_power: &mut bool,
) {
    let ctx = root_ui.ctx().clone();
    Panel::top("top").show_inside(root_ui, |ui| {
        ui.horizontal(|ui| {
            ui.heading("Conchordal");
            ui.separator();
            let scenario = if frame.meta.scenario_name.is_empty() {
                "Unknown".to_string()
            } else {
                frame.meta.scenario_name.clone()
            };
            let scene = frame
                .meta
                .scene_name
                .clone()
                .unwrap_or_else(|| "—".to_string());
            ui.label(format!("Scenario: {scenario}"));
            ui.separator();
            ui.label(format!("Scene: {scene}"));
        });

        let progress = if frame.meta.duration_sec > 0.0 {
            (frame.meta.time_sec / frame.meta.duration_sec).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let progress_text = format!(
            "{} / {}",
            format_time(frame.meta.time_sec),
            format_time(frame.meta.duration_sec)
        );

        ui.horizontal(|ui| {
            let btn_size = 18.0;
            let bar_width =
                (ui.available_width() - btn_size - ui.spacing().item_spacing.x).max(50.0);
            ui.add(
                egui::ProgressBar::new(progress)
                    .text(progress_text)
                    .desired_width(bar_width),
            );

            let (rect, resp) = ui.allocate_exact_size(Vec2::splat(btn_size), egui::Sense::click());
            let painter = ui.painter_at(rect);
            let tooltip = match frame.meta.playback_state {
                PlaybackState::NotStarted => {
                    let points = vec![
                        rect.left_top(),
                        egui::pos2(rect.right(), rect.center().y),
                        rect.left_bottom(),
                    ];
                    painter.add(egui::Shape::convex_polygon(
                        points,
                        egui::Color32::GREEN,
                        egui::Stroke::NONE,
                    ));
                    "Start (Space)"
                }
                PlaybackState::Playing => {
                    painter.add(egui::Shape::circle_filled(
                        rect.center(),
                        rect.width() * 0.45,
                        egui::Color32::GREEN,
                    ));
                    "Exit"
                }
                PlaybackState::Finished => {
                    painter.add(egui::Shape::line_segment(
                        [rect.left_top(), rect.right_bottom()],
                        egui::Stroke::new(2.0, egui::Color32::RED),
                    ));
                    painter.add(egui::Shape::line_segment(
                        [rect.right_top(), rect.left_bottom()],
                        egui::Stroke::new(2.0, egui::Color32::RED),
                    ));
                    "Exit"
                }
            };
            let resp = resp.on_hover_text(tooltip);
            if resp.clicked() {
                match frame.meta.playback_state {
                    PlaybackState::NotStarted => start_flag.store(true, Ordering::SeqCst),
                    PlaybackState::Playing | PlaybackState::Finished => {
                        exit_flag.store(true, Ordering::SeqCst)
                    }
                }
            }
        });

        if frame.meta.playback_state == PlaybackState::NotStarted
            && ctx.input(|i| i.focused && i.key_pressed(egui::Key::Space))
        {
            start_flag.store(true, Ordering::SeqCst);
        }
    });

    CentralPanel::default().show_inside(root_ui, |ui| {
        if let Some(err) = audio_error {
            ui.colored_label(egui::Color32::RED, format!("Audio init failed: {err}"));
        }
        let time_link_id = "time_freq_link";
        let x_max = rhythm_history.back().map(|(t, _)| *t).unwrap_or(0.0);
        let window_start = x_max - 5.0;
        let window_end = (window_start + 5.0).max(window_start + 0.1);
        draw_system_diagnostics(ui, frame);
        ui.separator();
        draw_listener_dashboard(
            ui,
            frame,
            rhythm_history,
            dorsal_history,
            window_start,
            window_end,
            time_link_id,
        );

        ui.separator();
        ui.horizontal(|ui| {
            ui.heading("Population Dynamics");
            ui.separator();
            ui.label(format!("Voices: {}", frame.meta.voice_count));
            draw_kuramoto_order(ui, frame);
        });
        plot_population_dynamics(
            ui,
            &frame.voices,
            &frame.spec.spec_hz,
            &frame.spec.amps,
            72.0,
        );

        ui.separator();
        ui.horizontal(|ui| {
            ui.heading("Subjective Intensity");
            ui.checkbox(show_raw_nsgt_power, "Show raw NSGT power");
        });

        let intensity_max = frame
            .landscape
            .subjective_intensity
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        let y_min = 0.0;
        let y_max = 100.0;

        let overlay_vals = if *show_raw_nsgt_power {
            let raw_max = frame
                .landscape
                .nsgt_power
                .iter()
                .cloned()
                .fold(0.0f32, f32::max)
                .max(1e-6);
            let scale = intensity_max.max(1e-6) / raw_max;
            Some(
                frame
                    .landscape
                    .nsgt_power
                    .iter()
                    .map(|&p| p * scale)
                    .collect::<Vec<f32>>(),
            )
        } else {
            None
        };
        let overlay = overlay_vals.as_ref().map(|vals| {
            (
                vals.as_slice(),
                "NSGT power (linear)",
                Color32::from_rgb(80, 160, 220),
            )
        });

        log2_plot_hz(
            ui,
            "Subjective Intensity",
            &frame.landscape.space.centers_hz,
            &frame.landscape.subjective_intensity,
            "Intensity",
            y_min,
            y_max,
            102.0,
            Some("landscape_group"),
            None,
            overlay,
        );

        ui.separator();

        let consonance_field_score = &frame.landscape.consonance_field_score;
        let (y_min, y_max) = consonance_field_score_axis_bounds(frame.landscape.consonance_kernel);

        ui.columns(1, |cols| {
            let ui = &mut cols[0];

            ui.heading("Consonance Landscape");
            // Combined Consonance Score C
            log2_plot_hz(
                ui,
                "Consonance Field Score",
                &frame.landscape.space.centers_hz,
                consonance_field_score,
                "C_field_score",
                y_min,
                y_max,
                102.0,
                Some("landscape_group"),
                None,
                None,
            );

            ui.heading("Roughness / Harmonicity");
            draw_roughness_harmonicity(
                ui,
                "Roughness / Harmonicity",
                &frame.landscape.space.centers_hz,
                &frame.landscape.harmonicity01,
                &frame.landscape.roughness01,
                102.0,
                Some("landscape_group"),
            );
        });

        ui.separator();
        egui::CollapsingHeader::new("Prediction")
            .default_open(false)
            .show_unindented(ui, |ui| {
                let fs = frame.wave.fs;
                let next_gate_tick = frame
                    .prediction
                    .next_gate_tick_est
                    .map(|tick| tick.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let next_gate_sec = frame.prediction.next_gate_tick_est.and_then(|tick| {
                    if fs > 0.0 {
                        Some(tick as f32 / fs)
                    } else {
                        None
                    }
                });
                let next_gate_sec = next_gate_sec
                    .map(|sec| format!("{sec:.3}"))
                    .unwrap_or_else(|| "None".to_string());
                let theta_hz = frame
                    .prediction
                    .theta_hz
                    .map(|hz| format!("{hz:.3}"))
                    .unwrap_or_else(|| "None".to_string());
                let delta_hz = frame
                    .prediction
                    .delta_hz
                    .map(|hz| format!("{hz:.3}"))
                    .unwrap_or_else(|| "None".to_string());
                let n_theta_per_delta = frame
                    .prediction
                    .pred_n_theta_per_delta
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let pred_tau_tick = frame.prediction.pred_tau_tick;
                let pred_horizon_tick = frame.prediction.pred_horizon_tick;
                let pred_tau = if let Some(tick) = pred_tau_tick {
                    if fs > 0.0 {
                        format!("{tick} ({:.1} ms)", tick as f32 / fs * 1000.0)
                    } else {
                        tick.to_string()
                    }
                } else {
                    "None".to_string()
                };
                let pred_horizon = if let Some(tick) = pred_horizon_tick {
                    if fs > 0.0 {
                        format!("{tick} ({:.1} ms)", tick as f32 / fs * 1000.0)
                    } else {
                        tick.to_string()
                    }
                } else {
                    "None".to_string()
                };

                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("next_gate_tick={next_gate_tick}"));
                    ui.separator();
                    ui.label(format!("next_gate_sec={next_gate_sec}"));
                    ui.separator();
                    ui.label(format!("theta_hz={theta_hz}"));
                    ui.separator();
                    ui.label(format!("delta_hz={delta_hz}"));
                    ui.separator();
                    ui.label(format!("n_theta_per_delta={n_theta_per_delta}"));
                    ui.separator();
                    ui.label(format!("pred_tau={pred_tau}"));
                    ui.separator();
                    ui.label(format!("pred_horizon={pred_horizon}"));
                });

                let pred_overlay =
                    frame
                        .prediction
                        .pred_c_field_level_next_gate
                        .as_ref()
                        .map(|scan| {
                            (
                                scan.as_ref(),
                                "Predicted C_field_level",
                                Color32::from_rgb(230, 170, 90),
                            )
                        });
                log2_plot_hz(
                    ui,
                    "Consonance Field Level (Observed/Predicted)",
                    &frame.landscape.space.centers_hz,
                    &frame.landscape.consonance_field_level,
                    "C_field_level",
                    0.0,
                    1.0,
                    102.0,
                    Some("landscape_group"),
                    None,
                    pred_overlay,
                );
                if let Some(scan) = frame.prediction.pred_c_field_level_next_gate.as_ref() {
                    let mut sum = 0.0f32;
                    let mut max = 0.0f32;
                    for &v in scan.iter() {
                        sum += v;
                        if v > max {
                            max = v;
                        }
                    }
                    let mean = sum / (scan.len().max(1) as f32);
                    ui.label(format!(
                        "pred_c_field_level_next_gate mean={mean:.3} max={max:.3}"
                    ));
                } else {
                    ui.label("pred_c_field_level_next_gate=None");
                }
            });

        ui.separator();
        egui::CollapsingHeader::new("Phonation")
            .default_open(false)
            .show(ui, |ui| {
                let gate_boundary = frame
                    .prediction
                    .gate_boundary_in_hop
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let pred_available = frame
                    .prediction
                    .pred_available_in_hop
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let phonation_onsets = frame
                    .prediction
                    .phonation_onsets_in_hop
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "None".to_string());
                ui.horizontal(|ui| {
                    ui.label(format!("gate_boundary_in_hop={gate_boundary}"));
                    ui.separator();
                    ui.label(format!("pred_available_in_hop={pred_available}"));
                    ui.separator();
                    ui.label(format!("phonation_onsets_in_hop={phonation_onsets}"));
                });
                if let (Some(min), Some(mean), Some(max)) = (
                    frame.prediction.pred_gain_raw_min,
                    frame.prediction.pred_gain_raw_mean,
                    frame.prediction.pred_gain_raw_max,
                ) {
                    ui.label(format!(
                        "pred_gain_raw min/mean/max={min:.3}/{mean:.3}/{max:.3}"
                    ));
                } else {
                    ui.label("pred_gain_raw=None");
                }
                if let (Some(min), Some(mean), Some(max)) = (
                    frame.prediction.pred_gain_mixed_min,
                    frame.prediction.pred_gain_mixed_mean,
                    frame.prediction.pred_gain_mixed_max,
                ) {
                    ui.label(format!(
                        "pred_gain_mixed min/mean/max={min:.3}/{mean:.3}/{max:.3}"
                    ));
                } else {
                    ui.label("pred_gain_mixed=None");
                }
                let pred_sync = frame
                    .prediction
                    .pred_sync_mean
                    .map(|val| format!("{val:.3}"))
                    .unwrap_or_else(|| "None".to_string());
                ui.label(format!("pred_sync_mean={pred_sync}"));
            });
    });
}
