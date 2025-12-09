use crate::ui::plots::{log2_plot_hz, time_plot};
use crate::ui::viewdata::UiFrame;
use egui::{CentralPanel, TopBottomPanel, Vec2};

use egui::epaint::{ColorImage, TextureHandle};
use egui::{Color32, Ui};
use egui_plot::{Plot, PlotImage, PlotPoint};

/// === PLV heatmap ===
pub fn show_plv_heatmap(
    ui: &mut Ui,
    freqs_x: &[f32],
    freqs_y: &[f32],
    plv: &[Vec<f32>],
    tex: &mut Option<TextureHandle>,
) {
    let nx = freqs_x.len();
    let ny = freqs_y.len();
    if nx == 0 || ny == 0 {
        ui.label("PLV data empty.");
        return;
    }

    // --- build color pixels (flip vertically) ---
    let mut pixels = vec![Color32::BLACK; nx * ny];
    for j in 0..ny {
        let row = ny - 1 - j; // flip vertically (low freq bottom)
        for i in 0..nx {
            let v = plv
                .get(j)
                .and_then(|r| r.get(i))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let r = (v * 255.0) as u8;
            let b = ((1.0 - v) * 255.0) as u8;
            pixels[row * nx + i] = Color32::from_rgb(r, 0, b);
        }
    }

    // --- upload texture ---
    let img = ColorImage::new([nx, ny], pixels);
    let texture = tex.get_or_insert_with(|| {
        ui.ctx()
            .load_texture("plv_heatmap", img.clone(), egui::TextureOptions::LINEAR)
    });
    texture.set(img, egui::TextureOptions::LINEAR);

    // --- coordinate ranges ---
    let fx_min = *freqs_x.first().unwrap();
    let fx_max = *freqs_x.last().unwrap();
    let fy_min = *freqs_y.first().unwrap();
    let fy_max = *freqs_y.last().unwrap();

    // --- draw plot ---
    let plot = Plot::new("plv_plot")
        .data_aspect(1.0)
        .allow_zoom(false)
        .allow_scroll(false)
        .include_x(fx_min as f64)
        .include_x(fx_max as f64)
        .include_y(fy_min as f64)
        .include_y(fy_max as f64);

    plot.show(ui, |plot_ui| {
        let center_x = (fx_min + fx_max) * 0.5;
        let center_y = (fy_min + fy_max) * 0.5;
        let size_x = fx_max - fx_min;
        let size_y = fy_max - fy_min;

        let img = PlotImage::new(
            "plv_img",
            texture.id(),
            PlotPoint::new(center_x, center_y),
            Vec2::new(size_x, size_y),
        );
        plot_ui.image(img);
    });
}

/// === Main window ===
pub fn main_window(ctx: &egui::Context, frame: &UiFrame) {
    TopBottomPanel::top("top").show(ctx, |ui| {
        ui.heading("Conchordal — NSGT Landscape Viewer");
        ui.label("Wave + Landscape (log₂-space R, PLV-based C)");
    });

    CentralPanel::default().show(ctx, |ui| {
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

        log2_plot_hz(
            ui,
            "NSGT envelope",
            &frame.landscape.space.centers_hz,
            &frame.landscape.amps_last,
            "NSGT",
            0.0,
            (11.) as f64,
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
