use crate::core::db::amp_to_db;
use crate::ui::plots::{
    METER_DISPLAY_CONFIDENCE_MIN, beat_strobe_legend, beat_strobe_plot, draw_harmony_circle,
    draw_meter_circle, draw_roughness_harmonicity, log2_plot_hz, plot_population_dynamics,
    rhythm_slow_plots, spectrum_time_freq_axes, time_plot,
};
use crate::viewdata::{PlaybackState, UiFrame};
use egui::{CentralPanel, Color32, Panel, Vec2};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Span of the scrolling listener plots, in stream seconds.
pub const TIME_WINDOW_SEC: f64 = 5.0;

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

fn format_time(sec: f64) -> String {
    let total_secs = sec.max(0.0).floor() as u64;
    let days = total_secs / 86_400;
    let hours = total_secs / 3600 % 24;
    let minutes = total_secs / 60 % 60;
    let seconds = total_secs % 60;
    if days > 0 {
        format!("{days}d {hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{hours}:{minutes:02}:{seconds:02}")
    }
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
            egui::Stroke::new(1.0_f32, window_stroke_color),
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
            egui::Stroke::new(2.0_f32, color_for(win)),
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

fn listener_state_meter(
    ui: &mut egui::Ui,
    label: &str,
    value: Option<f32>,
    color: Color32,
) -> egui::Response {
    let level = value.unwrap_or(0.0).clamp(0.0, 1.0);
    let text = if value.is_some() {
        format!("{level:>4.2}")
    } else {
        " N/A".to_string()
    };
    ui.vertical(|ui| {
        ui.label(egui::RichText::new(label).size(9.0).color(color));
        ui.horizontal(|ui| {
            let desired = Vec2::new(42.0, 8.0);
            let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
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

            mini_fixed_text_label(ui, text, 29.0, color);
        });
    })
    .response
}

fn draw_listener_state_readout(ui: &mut egui::Ui, frame: &UiFrame) {
    let prediction = frame.listener.prediction;
    let forecast = prediction.forecast;
    // The E0 details belong to the two bars they explain, so they appear on hover
    // there instead of in a separate menu row.
    egui::Grid::new("listener_state_readout")
        .num_columns(2)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            listener_state_meter(
                ui,
                "Sensory salience",
                frame
                    .listener
                    .has_fast_state
                    .then_some(frame.listener.attention_level),
                Color32::from_rgb(90, 180, 230),
            )
            .on_hover_text("Bottom-up acoustic change (flux), smoothed.");
            listener_state_meter(
                ui,
                "Adapted stability",
                frame
                    .listener
                    .has_state
                    .then_some(frame.listener.stability_level),
                Color32::from_rgb(80, 200, 130),
            )
            .on_hover_text("Mass-weighted consonance field after habituation.");
            ui.end_row();
            listener_state_meter(
                ui,
                "Uncertainty H / ln 5",
                forecast
                    .and_then(|f| f.entropy_nats)
                    .map(|h| h / 5.0_f32.ln()),
                Color32::from_rgb(235, 95, 85),
            )
            .on_hover_ui(|ui| e0_details(ui, &prediction));
            listener_state_meter(
                ui,
                "Arrival score",
                forecast.and_then(|f| f.arrival_score),
                Color32::from_rgb(235, 200, 75),
            )
            .on_hover_ui(|ui| e0_details(ui, &prediction));
            ui.end_row();
        });
}

/// What the E0 uncertainty and arrival bars are based on.
fn e0_details(ui: &mut egui::Ui, prediction: &crate::viewdata::ListenerPrediction) {
    ui.set_max_width(400.0);
    let forecast = prediction.forecast;
    ui.label(egui::RichText::new("E0 prediction (uncalibrated scores)").strong());
    match forecast {
        Some(f) => {
            ui.label(
                f.entropy_nats
                    .map_or_else(|| "H: N/A".to_owned(), |h| format!("H: {h:.2} nat")),
            );
            ui.label(format!("Coverage score: {:.3}", f.coverage_score));
            ui.label("Target: new audible cell");
            ui.label(format!("Δ: {:.3} s (1 beat)", f.horizon_sec));
        }
        None => {
            ui.label("Prediction: N/A (meter or observation unavailable)");
        }
    }
    ui.label("Entropy uses p_known; all UNKNOWN has N/A entropy. Coverage is an uncalibrated support score.");
    ui.label("E0 q = coverage × p_known + remaining mass × marginal · uncalibrated scores");
    ui.label("Arrival target: the next support change enters a different audible cell. Later changes are outside this target.");
    if let Some(f) = forecast {
        ui.label(format!(
            "Samples: ({}, {}] · reference: {}",
            f.start_sample, f.end_sample, f.reference_sample
        ));
        ui.label(format!(
            "Reference cell: {:?} · retained successors: {}",
            f.reference_bin, f.examples
        ));
        if let Some(p) = f.q {
            for (name, score) in [
                "Unchanged",
                "Audible entry",
                "Lower cell",
                "Higher cell",
                "Silence",
            ]
            .into_iter()
            .zip(p)
            {
                ui.label(format!("{name}: {score:.3}"));
            }
        } else {
            ui.label("Next event: N/A (no same-support marginal at issuance)");
        }
    }
    if let Some(cancelled) = prediction.censored {
        ui.label(format!(
            "Censored window #{}: reference changed before or at start (sample {})",
            cancelled.forecast.version, cancelled.observed_sample
        ));
    }
    if let Some(done) = prediction.completed {
        let surprise = if done.zero_likelihood {
            "∞ (zero model likelihood)".to_owned()
        } else {
            done.surprise_nats
                .map_or_else(|| "N/A".to_owned(), |s| format!("{s:.3} nat"))
        };
        ui.label(format!(
            "Last completed window #{} · surprise: {surprise}",
            done.forecast.version
        ));
    } else {
        ui.label("Surprise: N/A (no completed window)");
    }
}

fn meter_band_label(name: &str, band: crate::core::meter::MeterBand) -> String {
    let confidence = band.confidence.clamp(0.0, 1.0);
    let freq_text = if confidence < METER_DISPLAY_CONFIDENCE_MIN {
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
        if let Some(script) = &frame.meta.script {
            if let Some(error) = &script.error {
                ui.label(format!("Script error {:?}: {error}", script.error_position));
            } else if script.stopped {
                ui.label("Script stopped");
            } else if script.done {
                ui.label("Script done");
            } else {
                ui.label(format!(
                    "Script evaluated through {:.3}s",
                    script.evaluated_through
                ));
            }
            for warning in &script.warnings {
                ui.label(format!(
                    "{:?}: {} ({} occurrences)",
                    warning.position, warning.message, warning.count
                ));
            }
        }
        fixed_system_label(
            ui,
            if frame.meta.estimator_freezes == 0 {
                "estimator freezes=0".into()
            } else {
                format!(
                    "estimator freezes={} frozen at issue {}",
                    frame.meta.estimator_freezes, frame.meta.estimator_last_frozen_issue
                )
            },
        );
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
        fixed_system_label(
            ui,
            format!(
                "events={:04} populations={} slow_hops={}",
                frame.meta.event_queue_len, frame.meta.population_count, frame.meta.slow_hops
            ),
        );
        ui.separator();
        fixed_system_label(
            ui,
            format!(
                "audio xrun={} rebuild={}{}",
                frame.meta.audio_xruns,
                frame.meta.audio_rebuilds,
                if frame.meta.audio_recovering {
                    " recovering"
                } else {
                    ""
                },
            ),
        );
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
    rhythm_history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    dorsal_history: &VecDeque<(f64, crate::viewdata::DorsalFrame)>,
    harmony_history: &VecDeque<(f64, Option<Vec2>)>,
    window_end: f64,
    time_link_id: &str,
) {
    let window_start = window_end - TIME_WINDOW_SEC;
    let (left_width, _) = split_widths(ui, 0.10, 250.0, 270.0, 640.0);
    let dashboard_height = 300.0;
    let audio_height = 60.0;
    let plot_height = 82.0;
    let strobe_height = 128.0;
    // Reserve the audio heading, separator and the two readout rows; circle titles
    // sit inside the circles' corners. At the default width the circles reach half
    // the column, and the Roughness/Harmonicity layout test bounds this height.
    let circle_side = (dashboard_height - audio_height - 110.0_f32)
        .min((left_width - ui.spacing().item_spacing.x) * 0.5);

    // Top-align both columns; centering left the shorter column floating.
    // `horizontal_top` would start the row at the full available height, which
    // inside the page ScrollArea is the whole viewport.
    let row_size = Vec2::new(ui.available_width(), ui.spacing().interact_size.y);
    ui.allocate_ui_with_layout(row_size, egui::Layout::left_to_right(egui::Align::Min), |ui| {
        ui.allocate_ui_with_layout(
            Vec2::new(left_width, dashboard_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
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
                ui.horizontal(|ui| {
                    draw_meter_circle(ui, rhythm_history, window_end, circle_side);
                    draw_harmony_circle(ui, &frame.landscape, harmony_history, circle_side);
                });
                draw_listener_state_readout(ui, frame);
            },
        );

        ui.separator();
        let right_width = ui.available_width().max(0.0);
        ui.allocate_ui_with_layout(
            Vec2::new(right_width, dashboard_height),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {

                if let Some(body) = &frame.body {
                    ui.heading("Private body descriptors (uncalibrated)");
                    if let Some(version) = body.prototype_model_version {
                        ui.label(format!("Prototype model {:02x}{:02x}{:02x}{:02x} (diagnostic)", version[0], version[1], version[2], version[3]));
                    }
                    ui.label(format!("{} frames · {} capture drops · {} outside-capacity voice hops · {} invalid bus hops",
                        body.processed_frames, body.capture_drops, body.outside_voice_hops, body.invalid_hops));
                    for (index, record) in body.records.iter().enumerate().filter(|(_, r)| r.active).take(4) {
                        ui.label(format!("Voice {} / {} · body {} · bus {} · support {}..{} · mask {:06b}",
                            record.source_id, record.source_generation, record.body_generation, record.bus,
                            record.start, record.end, record.mask));
                        if body.prototype_model_version.is_some() {
                            if let Some(assignment) = body.prototype_assignments[index] {
                                ui.label(format!("Descriptive medoid {} · distance {:.3} · {} coordinates", assignment.key.0, assignment.distance, assignment.common_coordinates));
                            } else {
                                ui.label("Descriptive medoid: unsupported");
                            }
                        }
                    }
                    ui.separator();
                }
                ui.heading("Listener Twin");
                if frame.temporal.iter().any(|s| {
                    s.state != crate::temporal_cognition::observation::ObservationState::Off
                }) {
                    ui.label("Temporal context: observing; relations unavailable");
                    for observation in frame.temporal.iter() {
                        let bus = if observation.bus == 0 {
                            "Habitat"
                        } else {
                            "Presentation"
                        };
                        ui.label(format!(
                            "{bus}: {:?} · {} frames · {} dropped",
                            observation.state,
                            observation.received_frames,
                            observation.delivery_dropped_frames
                        ));
                        if let Some(trajectories) = observation.trajectories {
                            ui.label(format!(
                                "{} spectral peaks · residual energy {:.3e}",
                                trajectories.peak_bins.iter().flatten().count(),
                                trajectories.energy[7]
                            ));
                        }
                        if observation.ridge_failed {
                            ui.label("Ridge diagnostics: failed for this source epoch");
                        } else if let Some(ridges) = observation.ridges {
                            if ridges.observed {
                                ui.label(format!(
                                    "{} acoustic ridges · research scales, uncalibrated",
                                    ridges.current.iter().flatten().count()
                                ));
                            } else {
                                ui.label("Ridge diagnostics: awaiting supported audio");
                            }
                        }
                        if let Some(context) = observation.group_prototypes {
                            ui.label(format!("Prototype/group matches: {} · {} descriptors · support end {}",
                                context.assignments.iter().flatten().count(), context.descriptors.iter().flatten().count(), context.end_sample));
                        }



                        if observation.acoustic_failed {
                            ui.label("Acoustic diagnostics: failed for this source epoch");
                        } else if let Some(acoustic) = observation.acoustic {
                            ui.label(format!(
                                "{} sound groups · {} raw descriptors · {} accents · uncalibrated",
                                acoustic.retained_groups.iter().flatten().count(),
                                acoustic.features.iter().flatten().count(),
                                acoustic
                                    .features
                                    .iter()
                                    .flatten()
                                    .filter(|f| { f.detector.is_some_and(|d| d.accent.is_some()) })
                                    .count(),
                            ));
                        }
                        if let Some(error) = observation.context_error {
                            ui.label(format!("Observed context: {error}"));
                        } else if let Some(context) = observation.context {
                            ui.label(format!("Observed context · coverage {:.0}% · {} groups{}",
                                100. * context.observed_coverage,
                                context.groups.iter().flatten().count(),
                                if context.censored { " · censored" } else { "" }));
                            for g in context.groups.iter().flatten() {
                                ui.label(format!("Group {} · {} hops · {} accents · weight {:.3} · arrival {:?}",
                                    g.group.generation, g.history_hops, g.accents, g.acoustic_weight,
                                    g.arrival_probability));
                            }
                        }
                        if let Some(error)=observation.period_error {
                            ui.label(format!("Period / arrival diagnostics: {error}"));
                        } else if let Some(period)=observation.period {
                            for group in period.groups.iter().flatten().filter(|g|g.active) {
                                ui.label(format!("Group {}: {} accents · {} period alternatives · uncalibrated",group.ledger.group.generation,group.ledger.retained_accents,group.peaks.iter().flatten().count()));
                                if let Some(f)=group.forecast {
                                    ui.label(format!("{:?} arrival {:?} · elapsed {:?} s · reset unknown {} · forecast {}–{} · evidence {}–{} / available {}",f.model,f.probability,f.elapsed_seconds,f.reset_unknown,f.issued_at,f.horizon_end,f.source_start,f.source_end,f.available));
                                }
                            }
                        }


                    }
                }
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
                    ui.label("Listener Meter");
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
                ui.columns(2, |columns| {
                    beat_strobe_legend(&mut columns[0]);
                    beat_strobe_plot(&mut columns[0], rhythm_history, strobe_height, window_end);
                    columns[1].horizontal(|ui| {
                        ui.label("30 s");
                        for (name, color) in [
                            ("Beat", Color32::from_rgb(80, 180, 255)),
                            ("Sub", Color32::from_rgb(70, 225, 135)),
                            ("Measure", Color32::from_rgb(255, 110, 90)),
                        ] {
                            ui.colored_label(color, name);
                        }
                    });
                    rhythm_slow_plots(&mut columns[1], rhythm_history, window_end);
                });
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
    rhythm_history: &VecDeque<crate::viewdata::ListenerRhythmHop>,
    dorsal_history: &VecDeque<(f64, crate::viewdata::DorsalFrame)>,
    harmony_history: &VecDeque<(f64, Option<Vec2>)>,
    display_time: f64,
    runtime_error: Option<&str>,
    exit_flag: &Arc<AtomicBool>,
    start_flag: &Arc<AtomicBool>,
    end_request: &Arc<AtomicBool>,
    show_raw_nsgt_power: &mut bool,
) {
    let ctx = root_ui.ctx().clone();
    Panel::top("top").show(root_ui, |ui| {
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

        ui.horizontal(|ui| {
            // Right to left: Exit sits at the window's right edge, the playback
            // state beside it, and the progress bar takes the remaining width.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Exit").clicked() {
                    exit_flag.store(true, Ordering::SeqCst);
                }
                match frame.meta.playback_state {
                    PlaybackState::NotStarted => {
                        if ui.button("Start").on_hover_text("Start (Space)").clicked() {
                            start_flag.store(true, Ordering::SeqCst);
                        }
                    }
                    PlaybackState::Playing => {
                        if ui.button("End").clicked() {
                            end_request.store(true, Ordering::SeqCst);
                        }
                    }
                    PlaybackState::Ending => {
                        ui.label("Ending (fade)…");
                    }
                    PlaybackState::Finished => {
                        ui.label("Finished");
                    }
                }
                if let Some(end) = frame.meta.end {
                    let progress = if end > 0.0 {
                        (frame.meta.time_sec / end).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    ui.add(
                        egui::ProgressBar::new(progress as f32)
                            .text(format!(
                                "{} / {}",
                                format_time(frame.meta.time_sec),
                                format_time(end)
                            ))
                            .desired_width(ui.available_width().max(50.0)),
                    );
                } else {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.label(format!("{} · open end", format_time(frame.meta.time_sec)));
                    });
                }
            });
        });

        if frame.meta.playback_state == PlaybackState::NotStarted
            && ctx.input(|i| i.focused && i.key_pressed(egui::Key::Space))
        {
            start_flag.store(true, Ordering::SeqCst);
        }
    });

    CentralPanel::default().show(root_ui, |ui| {
        // Scroll so the lower plots stay reachable at the minimum window height.
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        if let Some(err) = runtime_error {
            ui.colored_label(egui::Color32::RED, err);
        }
        let time_link_id = "time_freq_link";
        let window_end = display_time;
        draw_system_diagnostics(ui, frame);
        ui.separator();
        draw_listener_dashboard(
            ui,
            frame,
            rhythm_history,
            dorsal_history,
            harmony_history,
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
            Some("landscape_population"),
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
                Some("landscape_population"),
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
                Some("landscape_population"),
            );
        });

        ui.separator();
        egui::CollapsingHeader::new("Harmonic support — first-order fields")
            .show(ui, |ui| {
                if let Some(global) = &frame.landscape.support {
                    ui.label(format!("M {:.4} grid units · C {:.4} · D {:.4} · sample {}", global.mass, global.focus, global.conflict, global.end_sample));
                    ui.label(format!("Support includes sub-roots from {:.2} Hz; not root probabilities", global.geometry.root_start_log2.exp2()));
                    let id = egui::Id::unique("harmonic_support_voice");
                    let mut selected = ui.ctx().data_mut(|data| data.get_temp::<u64>(id)).unwrap_or_else(|| frame.voices.first().map_or(0, |v| v.id));
                    egui::ComboBox::from_id_salt(id).selected_text(format!("Voice {selected}"))
                        .show_ui(ui, |ui| { for voice in &frame.voices { ui.selectable_value(&mut selected, voice.id, format!("Voice {}", voice.id)); } });
                    ui.ctx().data_mut(|data| data.insert_temp(id, selected));
                    if let Some(voice) = frame.voices.iter().find(|v| v.id == selected) {
                        if let Some(reference) = &voice.support {
                            ui.label(if voice.support_template.is_some() { "Body partials reading first-order fields; PCM self-excluded reference" } else { "Single partial at the nominal pitch and level (no spectral self-report); PCM self-excluded reference." });
                            let mut scans = std::array::from_fn(|_| vec![0.; frame.landscape.space.n_bins()]);
                            if let Some(template) = &voice.support_template {
                                for i in 0..frame.landscape.space.n_bins() {
                                    for (scan, value) in scans.iter_mut().zip(reference.template_features(i, template)) { scan[i] = value; }
                                }
                            } else { reference.single_partial_scan(voice.support_single_density, &mut scans); }
                            for (name, scan) in ["dC (first order)", "dD (first order)", "N"].into_iter().zip(scans) {
                                let values: Vec<_> = scan.into_iter().map(|v| v as f32).collect();
                                let lo = values.iter().copied().fold(0., f32::min);
                                let hi = values.iter().copied().fold(0., f32::max);
                                log2_plot_hz(ui, name, &frame.landscape.space.centers_hz, &values, name, f64::from(lo), if hi > lo { f64::from(hi) } else { 1. }, 90., None, None, None);
                            }
                        } else { ui.label("No self-excluded support for this Voice; movement term unavailable."); }
                    }
                } else { ui.label("Support observation disabled or unavailable."); }
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
                        Some(tick as f64 / fs as f64)
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
                    Some("landscape_population"),
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
    });
}

#[cfg(test)]
mod time_tests {
    use super::format_time;

    #[test]
    fn elapsed_time_includes_hours_and_days() {
        for (seconds, expected) in [
            (-1.0, "0:00:00"),
            (59.9, "0:00:59"),
            (60.0, "0:01:00"),
            (3600.0, "1:00:00"),
            (86399.0, "23:59:59"),
            (86400.0, "1d 0:00:00"),
            (176461.0, "2d 1:01:01"),
        ] {
            assert_eq!(format_time(seconds), expected);
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    /// The author asked for the Roughness/Harmonicity plot to fit the minimum window.
    /// Inside the page ScrollArea a row that claims the available height would grow
    /// the dashboard to the whole viewport and push the lower plots off screen.
    #[test]
    fn roughness_harmonicity_fits_the_minimum_window() {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                Vec2::new(1200.0, 900.0),
            )),
            ..Default::default()
        };
        let frame = UiFrame::default();
        let flag = Arc::new(AtomicBool::new(false));
        let mut raw = false;
        let mut output = egui::Context::default().run_ui(input, |ui| {
            main_window(
                ui,
                &frame,
                &VecDeque::new(),
                &VecDeque::new(),
                &VecDeque::new(),
                0.0,
                None,
                &flag,
                &flag,
                &flag,
                &mut raw,
            )
        });
        output.textures_delta.clear();
        let heading_bottom = output.shapes.iter().find_map(|shape| match &shape.shape {
            egui::epaint::Shape::Text(text) if text.galley.text() == "Roughness / Harmonicity" => {
                Some(text.pos.y + text.galley.size().y)
            }
            _ => None,
        });
        // 102 is the R/H plot height passed by `main_window`.
        let bottom = heading_bottom.expect("R/H heading on screen") + 102.0;
        let anchors: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                egui::epaint::Shape::Text(text)
                    if matches!(
                        text.galley.text(),
                        "Audio"
                            | "Meter"
                            | "Harmony · habitat"
                            | "Population Dynamics"
                            | "Roughness / Harmonicity"
                    ) =>
                {
                    Some((text.galley.text(), text.pos.y))
                }
                _ => None,
            })
            .collect();
        assert!(
            bottom <= 900.0,
            "R/H plot ends at {bottom}; anchors: {anchors:?}"
        );
        // The author asked for Exit at the right edge, not after a fixed gap.
        let exit_right = output.shapes.iter().find_map(|shape| match &shape.shape {
            egui::epaint::Shape::Text(text) if text.galley.text() == "Exit" => {
                Some(text.pos.x + text.galley.size().x)
            }
            _ => None,
        });
        let exit_right = exit_right.expect("Exit button on screen");
        assert!(exit_right > 1150.0, "Exit ends at {exit_right}");
    }
}
