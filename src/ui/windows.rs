use crate::ui::plots::{log2_plot_hz, neural_compass, time_plot};
use crate::ui::viewdata::UiFrame;
use egui::{CentralPanel, TopBottomPanel, Vec2};

/// === Main window ===
pub fn main_window(ctx: &egui::Context, frame: &UiFrame, audio_error: Option<&str>) {
    TopBottomPanel::top("top").show(ctx, |ui| {
        ui.heading("Conchordal — NSGT Landscape Viewer");
        ui.label("Wave + Landscape (log₂-space R, PLV-based C)");
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
        ui.heading("Analytic");
        ui.label("Neural Rhythms");
        neural_compass(ui, &frame.landscape.rhythm);

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
