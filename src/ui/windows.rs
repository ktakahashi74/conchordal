use crate::core::db::amp_to_db;
use crate::ui::plots::{
    draw_rhythm_mandala, draw_roughness_harmonicity, log2_plot_hz, neural_activity_plot,
    plot_population_dynamics, spectrum_time_freq_axes, time_plot,
};
use crate::ui::viewdata::{PlaybackState, UiFrame};
use egui::{CentralPanel, Color32, TopBottomPanel, Vec2};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    meter_width_scale: f32,
) {
    let meter_height = (ui.available_height() * 0.8).max(80.0);
    let label_width = 20.0;
    let meter_width = 22.0 * meter_width_scale;
    let spacing = 8.0;
    let pad_y = 4.0;
    let total_width = label_width + spacing + meter_width * 2.0 + spacing;
    let (rect, _resp) =
        ui.allocate_exact_size(Vec2::new(total_width, meter_height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
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
            egui::Stroke::new(1.0, p.ctx().style().visuals.window_stroke().color),
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
            p.ctx().style().visuals.text_color(),
        );
    };

    let left_origin = rect.left_top() + egui::vec2(label_width + spacing, 0.0);
    let right_origin = left_origin + egui::vec2(meter_width + spacing, 0.0);
    draw_meter(&painter, left_origin, left_inst, left_win, "L");
    draw_meter(&painter, right_origin, right_inst, right_win, "R");
}

fn split_widths(ui: &egui::Ui, ratio: f32, min_left: f32, min_right: f32) -> (f32, f32) {
    let sep = ui.spacing().item_spacing.x;
    let available = ui.available_width();
    let left_target = (available * ratio).max(min_left);
    let left_width = left_target.min((available - min_right - sep).max(0.0));
    let right_width = (available - left_width - sep).max(0.0);
    (left_width, right_width)
}

/// === Main window ===
#[allow(clippy::too_many_arguments)]
pub fn main_window(
    ctx: &egui::Context,
    frame: &UiFrame,
    rhythm_history: &VecDeque<(f64, crate::core::modulation::NeuralRhythms)>,
    dorsal_history: &VecDeque<(f64, crate::ui::viewdata::DorsalFrame)>,
    audio_error: Option<&str>,
    exit_flag: &Arc<AtomicBool>,
    start_flag: &Arc<AtomicBool>,
    show_raw_nsgt_power: &mut bool,
) {
    TopBottomPanel::top("top").show(ctx, |ui| {
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
            ui.separator();
            ui.label(format!("Events: {}", frame.meta.event_queue_len));
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

    CentralPanel::default().show(ctx, |ui| {
        if let Some(err) = audio_error {
            ui.colored_label(egui::Color32::RED, format!("Audio init failed: {err}"));
        }
        let time_link_id = "time_freq_link";
        let x_max = rhythm_history.back().map(|(t, _)| *t).unwrap_or(0.0);
        let window_start = x_max - 5.0;
        let window_end = (window_start + 5.0).max(window_start + 0.1);
        let (left_width, _) = split_widths(ui, 0.17, 100.0, 200.0);
        let row_height = 100.0;
        let height = 100.0;
        let legend_room = 12.0;
        let block_height = height + legend_room;
        let attention_height = row_height * 0.7;
        let neural_time_height = block_height * 0.7;
        let left_row_height = attention_height;
        let left_block_height = neural_time_height;
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                Vec2::new(left_width, left_row_height + left_block_height),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    let peak_db = if frame.meta.peak_level > 0.0 {
                        20.0 * frame.meta.peak_level.log10()
                    } else {
                        f32::NEG_INFINITY
                    };
                    let peak_text = if peak_db.is_infinite() {
                        "-inf dB".to_string()
                    } else {
                        format!("{:>5.1} dB", peak_db)
                    };
                    let peak_color = if frame.meta.peak_level > 1.0 {
                        egui::Color32::RED
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.horizontal(|ui| {
                        ui.heading("Audio");
                        ui.separator();
                        ui.colored_label(peak_color, format!("Peak level: {peak_text}"));
                    });
                    ui.allocate_ui_with_layout(
                        Vec2::new(left_width, left_row_height),
                        egui::Layout::left_to_right(egui::Align::Min),
                        |ui| {
                            // === Level meter ===
                            ui.vertical(|ui| {
                                draw_level_meters(
                                    ui,
                                    frame.meta.channel_peak[0],
                                    frame.meta.window_peak[0],
                                    frame.meta.channel_peak[1],
                                    frame.meta.window_peak[1],
                                    0.45,
                                );
                            });

                            ui.separator();
                            // === Waveform ===
                            ui.vertical(|ui| {
                                ui.label("Wave frame");
                                ui.allocate_ui_with_layout(
                                    Vec2::new(ui.available_width(), left_row_height),
                                    egui::Layout::top_down(egui::Align::LEFT),
                                    |ui| {
                                        time_plot(
                                            ui,
                                            "Current Hop Wave",
                                            frame.wave.fs as f64,
                                            frame.wave.samples.as_ref(),
                                            left_row_height,
                                            false,
                                        );
                                    },
                                );
                            });
                        },
                    );

                    ui.separator();
                    ui.heading("Neural Mandala");
                    ui.allocate_ui_with_layout(
                        Vec2::new(left_width, left_block_height),
                        egui::Layout::top_down(egui::Align::LEFT),
                        |ui| {
                            ui.set_min_height(left_block_height);
                            let side_len = ((left_block_height - legend_room) * 0.4).max(60.0);
                            let side = Vec2::splat(side_len);
                            ui.add_space(10.0);
                            ui.horizontal(|ui| {
                                ui.add_space(8.0);
                                draw_rhythm_mandala(ui, &frame.landscape.rhythm, side);
                                ui.vertical(|ui| {
                                    let labels = [
                                        ("Delta", egui::Color32::from_rgb(80, 180, 255)),
                                        ("Theta", egui::Color32::from_rgb(70, 225, 135)),
                                        ("Alpha", egui::Color32::from_rgb(255, 215, 60)),
                                        ("Beta", egui::Color32::from_rgb(255, 110, 90)),
                                    ];
                                    for (label, color) in labels {
                                        ui.label(
                                            egui::RichText::new(label).color(color).size(12.0),
                                        );
                                    }
                                });
                            });
                        },
                    );
                },
            );

            ui.separator();
            let right_width = ui.available_width().max(0.0);
            ui.allocate_ui_with_layout(
                Vec2::new(right_width, attention_height + neural_time_height),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    let old_spacing = ui.spacing().item_spacing;
                    ui.spacing_mut().item_spacing.y = 0.0;
                    ui.label("Auditory attention");
                    spectrum_time_freq_axes(
                        ui,
                        dorsal_history,
                        attention_height,
                        window_start,
                        window_end,
                        Some(time_link_id),
                    );
                    ui.spacing_mut().item_spacing = old_spacing;
                    ui.separator();
                    ui.label("Neural activity");
                    ui.allocate_ui_with_layout(
                        Vec2::new(right_width, neural_time_height),
                        egui::Layout::top_down(egui::Align::LEFT),
                        |ui| {
                            ui.set_min_width(right_width);
                            ui.set_min_height(neural_time_height);
                            neural_activity_plot(
                                ui,
                                rhythm_history,
                                neural_time_height,
                                window_start,
                                window_end,
                                Some(time_link_id),
                            );
                        },
                    );
                },
            );
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.heading("Population Dynamics");
            ui.separator();
            ui.label(format!("Agents: {}", frame.meta.agent_count));
        });
        plot_population_dynamics(
            ui,
            &frame.agents,
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

        let consonance_score = &frame.landscape.consonance_score;
        let (mut c_min, mut c_max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in consonance_score.iter() {
            if v.is_finite() {
                if v < c_min {
                    c_min = v;
                }
                if v > c_max {
                    c_max = v;
                }
            }
        }
        if !c_min.is_finite() || !c_max.is_finite() {
            c_min = -1.0;
            c_max = 1.0;
        }
        if (c_max - c_min).abs() < 1e-6 {
            c_min -= 0.5;
            c_max += 0.5;
        }
        let pad = (c_max - c_min).abs() * 0.1;
        let y_min = (c_min - pad) as f64;
        let y_max = (c_max + pad) as f64;

        ui.columns(1, |cols| {
            let ui = &mut cols[0];

            ui.heading("Consonance Landscape");
            // Combined Consonance Score C
            log2_plot_hz(
                ui,
                "Consonance Score Landscape",
                &frame.landscape.space.centers_hz,
                consonance_score,
                "C score",
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
            .show(ui, |ui| {
                let fs = frame.wave.fs;
                let next_gate_tick = frame
                    .next_gate_tick_est
                    .map(|tick| tick.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let next_gate_sec = frame.next_gate_tick_est.and_then(|tick| {
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
                    .theta_hz
                    .map(|hz| format!("{hz:.3}"))
                    .unwrap_or_else(|| "None".to_string());
                let delta_hz = frame
                    .delta_hz
                    .map(|hz| format!("{hz:.3}"))
                    .unwrap_or_else(|| "None".to_string());
                let n_theta_per_delta = frame
                    .pred_n_theta_per_delta
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let pred_tau_tick = frame.pred_tau_tick;
                let pred_horizon_tick = frame.pred_horizon_tick;
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

                let pred_overlay = frame.pred_c_level01_next_gate.as_ref().map(|scan| {
                    (
                        scan.as_ref(),
                        "Predicted C_level01",
                        Color32::from_rgb(230, 170, 90),
                    )
                });
                log2_plot_hz(
                    ui,
                    "Consonance Level (Observed/Predicted)",
                    &frame.landscape.space.centers_hz,
                    &frame.landscape.consonance_level01,
                    "C_level01",
                    0.0,
                    1.0,
                    102.0,
                    Some("landscape_group"),
                    None,
                    pred_overlay,
                );
                if let Some(scan) = frame.pred_c_level01_next_gate.as_ref() {
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
                        "pred_c_level01_next_gate mean={mean:.3} max={max:.3}"
                    ));
                } else {
                    ui.label("pred_c_level01_next_gate=None");
                }
            });

        ui.separator();
        egui::CollapsingHeader::new("Phonation")
            .default_open(false)
            .show(ui, |ui| {
                let gate_boundary = frame
                    .gate_boundary_in_hop
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let pred_available = frame
                    .pred_available_in_hop
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "None".to_string());
                let phonation_onsets = frame
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
                    frame.pred_gain_raw_min,
                    frame.pred_gain_raw_mean,
                    frame.pred_gain_raw_max,
                ) {
                    ui.label(format!(
                        "pred_gain_raw min/mean/max={min:.3}/{mean:.3}/{max:.3}"
                    ));
                } else {
                    ui.label("pred_gain_raw=None");
                }
                if let (Some(min), Some(mean), Some(max)) = (
                    frame.pred_gain_mixed_min,
                    frame.pred_gain_mixed_mean,
                    frame.pred_gain_mixed_max,
                ) {
                    ui.label(format!(
                        "pred_gain_mixed min/mean/max={min:.3}/{mean:.3}/{max:.3}"
                    ));
                } else {
                    ui.label("pred_gain_mixed=None");
                }
                let pred_sync = frame
                    .pred_sync_mean
                    .map(|val| format!("{val:.3}"))
                    .unwrap_or_else(|| "None".to_string());
                ui.label(format!("pred_sync_mean={pred_sync}"));
            });
    });
}
