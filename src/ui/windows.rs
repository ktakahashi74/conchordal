use crate::ui::plots::{log2_plot_hz, time_plot};
use crate::ui::viewdata::UiFrame;
use egui::{CentralPanel, TopBottomPanel, Vec2};
use egui_plot::{Line, Plot, PlotPoints};

use crate::core::landscape::LandscapeFrame;

use egui::epaint::{ColorImage, TextureHandle};
use egui::{Color32, Response, ScrollArea, Ui};
use egui_plot::{PlotImage, PlotPoint, PlotResponse};

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
            eprintln!(
                "DEBUG: len(freqs)={}, len(R)={}, len(C)={}, len(K)={}",
                frame.landscape.freqs_hz.len(),
                frame.landscape.r_last.len(),
                frame.landscape.h_last.len(),
                frame.landscape.c_last.len()
            );

            // === Spectrum ===
            ui.vertical(|ui| {
                ui.heading("Synth Spectrum");
                if frame.spec.spec_hz.len() > 1 && frame.spec.amps.len() > 1 {
                    let max_amp = frame.spec.amps.iter().cloned().fold(0.0, f32::max);
                    crate::ui::plots::log2_hist_hz(
                        ui,
                        "Amplitude Spectrum",
                        &frame.spec.spec_hz[1..],
                        &frame.spec.amps[1..],
                        "A[k]",
                        0.0,
                        (max_amp * 1.05) as f64,
                    );
                }
            });
        });

        ui.separator();
        ui.heading("Analytic");

        log2_plot_hz(
            ui,
            "NSGT envelope",
            &frame.landscape.freqs_hz,
            &frame.landscape.amps_last,
            "NSGT",
            0.0,
            (1.05) as f64,
        );

        ui.separator();

        ui.separator();
        ui.heading("Landscape");

        let max_r = frame.landscape.r_last.iter().cloned().fold(0.0, f32::max);
        let min_c = frame.landscape.c_last.iter().cloned().fold(0.0, f32::min);
        let max_c = frame.landscape.c_last.iter().cloned().fold(0.0, f32::max);

        ui.columns(1, |cols| {
            let ui = &mut cols[0];

            // Roughness R
            log2_plot_hz(
                ui,
                "Roughness Landscape (R)",
                &frame.landscape.freqs_hz,
                &frame.landscape.r_last,
                "R",
                0.0,
                (max_r * 1.05) as f64,
            );

            // Harmonicity H
            log2_plot_hz(
                ui,
                "Harmonicity Landscape (H)",
                &frame.landscape.freqs_hz,
                &frame.landscape.h_last,
                "H",
                0.0,
                1.0,
            );

            // Combined Consonance C
            log2_plot_hz(
                ui,
                "Consonance potential",
                &frame.landscape.freqs_hz,
                &frame.landscape.c_last,
                "C",
                (min_c * 1.1) as f64,
                (max_c * 1.1) as f64,
            );
        });
    });
}
