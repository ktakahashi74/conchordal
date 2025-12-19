use crate::ui::plots::{
    draw_rhythm_mandala, log2_plot_hz, neural_activity_plot, plot_population_dynamics, time_plot,
};
use crate::ui::viewdata::{PlaybackState, UiFrame};
use egui::{CentralPanel, Key, TopBottomPanel, Vec2};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn format_time(sec: f32) -> String {
    let total_secs = sec.max(0.0).floor() as u64;
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    format!("{minutes:02}:{seconds:02}")
}

fn amp_to_db(a: f32) -> f32 {
    if a <= 0.0 { -60.0 } else { 20.0 * a.log10() }
}

fn draw_level_meters(
    ui: &mut egui::Ui,
    left_inst: f32,
    left_win: f32,
    right_inst: f32,
    right_win: f32,
) {
    let meter_height = ui.available_height().max(40.0);
    let label_width = 34.0;
    let meter_width = 22.0;
    let spacing = 8.0;
    let pad_y = 8.0;
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

/// === Main window ===
pub fn main_window(
    ctx: &egui::Context,
    frame: &UiFrame,
    rhythm_history: &VecDeque<(f64, crate::core::modulation::NeuralRhythms)>,
    audio_error: Option<&str>,
    exit_flag: &Arc<AtomicBool>,
    start_flag: &Arc<AtomicBool>,
) {
    TopBottomPanel::top("top").show(ctx, |ui| {
        ui.heading("Conchordal");

        ui.horizontal(|ui| {
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

        ui.horizontal(|ui| {
            ui.label(format!("Agents: {}", frame.meta.agent_count));
            ui.separator();
            ui.label(format!("Events: {}", frame.meta.event_queue_len));
        });
    });

    CentralPanel::default().show(ctx, |ui| {
        if let Some(err) = audio_error {
            ui.colored_label(egui::Color32::RED, format!("Audio init failed: {err}"));
        }
        ui.vertical(|ui| {
            ui.label("Audio");
            let row_height = 100.0;
            ui.horizontal(|ui| {
                ui.set_height(row_height);
                // === Level meter ===
                ui.vertical(|ui| {
                    //ui.separator();
                    let peak_db = if frame.meta.peak_level > 0.0 {
                        20.0 * frame.meta.peak_level.log10()
                    } else {
                        f32::NEG_INFINITY
                    };
                    let peak_text = if peak_db.is_infinite() {
                        "Peak: -inf dB".to_string()
                    } else {
                        format!("  Peak:  {:>6.1} dB", peak_db)
                    };
                    let peak_color = if frame.meta.peak_level > 1.0 {
                        egui::Color32::RED
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.colored_label(peak_color, peak_text);

                    ui.allocate_ui_with_layout(
                        Vec2::new(110.0, row_height),
                        egui::Layout::left_to_right(egui::Align::Min),
                        |ui| {
                            draw_level_meters(
                                ui,
                                frame.meta.channel_peak[0],
                                frame.meta.window_peak[0],
                                frame.meta.channel_peak[1],
                                frame.meta.window_peak[1],
                            );
                        },
                    );
                });

                ui.separator();
                // === Waveform ===
                ui.vertical(|ui| {
                    ui.label("Waveform");
                    ui.allocate_ui_with_layout(
                        Vec2::new(420.0, row_height),
                        egui::Layout::top_down(egui::Align::LEFT),
                        |ui| {
                            time_plot(
                                ui,
                                "Current Hop Wave",
                                frame.wave.fs as f64,
                                frame.wave.samples.as_ref(),
                                row_height,
                            );
                        },
                    );
                });

                ui.separator();
                // === Spectrum ===
                ui.vertical(|ui| {
                    ui.label("Sound bodies");
                    if frame.spec.spec_hz.len() > 1 && frame.spec.amps.len() > 1 {
                        crate::ui::plots::log2_hist_hz(
                            ui,
                            "Sound bodies",
                            &frame.spec.spec_hz[1..],
                            &frame.spec.amps[1..],
                            "A[k]",
                            0.0,
                            1.1,
                            row_height,
                        );
                    }
                });
            });
        });

        ui.separator();
        ui.heading("Neural Activity");
        let height = 100.0;
        let legend_room = 12.0;
        let block_height = height + legend_room;
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_min_height(block_height);
                let side_len = (height - legend_room).max(60.0);
                let side = Vec2::splat(side_len);
                draw_rhythm_mandala(ui, &frame.landscape.rhythm, side);
            });
            ui.vertical(|ui| {
                ui.set_min_width(ui.available_width());
                ui.set_min_height(block_height);
                neural_activity_plot(ui, rhythm_history, block_height);
            });
        });

        ui.separator();
        ui.heading("Population Dynamics");
        plot_population_dynamics(ui, &frame.agents, 140.0);

        ui.separator();
        ui.heading("Subjective Intensity");

        log2_plot_hz(
            ui,
            "Subjective Intensity",
            &frame.landscape.space.centers_hz,
            &frame.landscape.subjective_intensity,
            "Amplitude",
            0.0,
            11_f64,
            120.0,
            Some("landscape_group"),
        );

        ui.separator();

        let roughness: Vec<f32> = frame
            .landscape
            .roughness
            .iter()
            .map(|v| v.max(0.0))
            .collect();
        let harmonicity: Vec<f32> = frame
            .landscape
            .harmonicity
            .iter()
            .map(|v| v.clamp(0.0, 1.0))
            .collect();
        let consonance = &frame.landscape.consonance;

        ui.columns(1, |cols| {
            let ui = &mut cols[0];

            ui.heading("Consonance Landscape");
            // Combined Consonance C
            log2_plot_hz(
                ui,
                "Consonance Landscape",
                &frame.landscape.space.centers_hz,
                consonance,
                "C",
                -1.0,
                1.0,
                150.0,
                Some("landscape_group"),
            );

            ui.heading("Roughness");
            // Roughness R
            log2_plot_hz(
                ui,
                "Roughness",
                &frame.landscape.space.centers_hz,
                &roughness,
                "R",
                0.0,
                1.0,
                120.0,
                Some("landscape_group"),
            );

            ui.heading("Harmonicity");
            // Harmonicity H
            log2_plot_hz(
                ui,
                "Harmonicity",
                &frame.landscape.space.centers_hz,
                &harmonicity,
                "H",
                0.0,
                1.0,
                120.0,
                Some("landscape_group"),
            );
        });
    });
}
