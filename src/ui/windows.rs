use crate::ui::plots::{log2_plot_hz, neural_compass, neural_phase_plot, time_plot};
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
            ui.separator();
            let peak_db = if frame.meta.peak_level > 0.0 {
                20.0 * frame.meta.peak_level.log10()
            } else {
                f32::NEG_INFINITY
            };
            let peak_text = if peak_db.is_infinite() {
                "Peak: -inf dB".to_string()
            } else {
                format!("Peak: {:.1} dB", peak_db)
            };
            let peak_color = if frame.meta.peak_level > 1.0 {
                egui::Color32::RED
            } else {
                ui.visuals().text_color()
            };
            ui.colored_label(peak_color, peak_text);
        });
    });

    CentralPanel::default().show(ctx, |ui| {
        if let Some(err) = audio_error {
            ui.colored_label(egui::Color32::RED, format!("Audio init failed: {err}"));
        }
        ui.horizontal(|ui| {
            // === Waveform ===
            ui.vertical(|ui| {
                ui.heading("Waveform");
                ui.allocate_ui_with_layout(
                    Vec2::new(500.0, 180.0),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        time_plot(
                            ui,
                            "Current Hop Wave",
                            frame.wave.fs as f64,
                            &frame.wave.samples,
                        );
                    },
                );
            });

            ui.separator();
            // === Spectrum ===
            ui.vertical(|ui| {
                ui.heading("Synth Spectrum");
                if frame.spec.spec_hz.len() > 1 && frame.spec.amps.len() > 1 {
                    crate::ui::plots::log2_hist_hz(
                        ui,
                        "Amplitude Spectrum",
                        &frame.spec.spec_hz[1..],
                        &frame.spec.amps[1..],
                        "A[k]",
                        0.0,
                        20.0,
                    );
                }
            });
        });

        ui.separator();
        ui.heading("Neural Phases");
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                //ui.set_width(0.0);
                //ui.label("Neural Phases");
                neural_compass(ui, &frame.landscape.rhythm);
            });
            ui.vertical(|ui| {
                ui.set_min_width(ui.available_width());
                //                ui.label("Neural Phase Slope");
                neural_phase_plot(ui, rhythm_history);
            });
        });

        ui.separator();
        ui.heading("Analytic");

        log2_plot_hz(
            ui,
            "NSGT envelope",
            &frame.landscape.space.centers_hz,
            &frame.landscape.amps_last,
            "NSGT",
            0.0,
            11_f64,
            120.0,
        );

        ui.separator();

        let roughness: Vec<f32> = frame.landscape.r_last.iter().map(|v| v.max(0.0)).collect();
        let harmonicity: Vec<f32> = frame
            .landscape
            .h_last
            .iter()
            .map(|v| v.clamp(0.0, 1.0))
            .collect();
        let consonance = &frame.landscape.c_last;

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
            );
        });
    });
}
