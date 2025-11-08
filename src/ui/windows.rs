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

/// === Draw envelope amplitude over log2 freq ===
pub fn draw_nsgt_envelope(ui: &mut egui::Ui, frame: &LandscapeFrame) -> PlotResponse<()> {
    Plot::new("nsgt_envelope_plot")
        .legend(egui_plot::Legend::default())
        .allow_scroll(false)
        .allow_drag(false)
        .height(150.0)
        .x_axis_formatter(|mark, _| {
            let hz = 2f64.powf(mark.value);
            format!("{:.0} Hz", hz)
        })
        .include_x((20.0f64).log2())
        .include_x((20_000.0f64).log2())
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            if !frame.freqs_hz.is_empty() && frame.amps_last.len() == frame.freqs_hz.len() {
                let pts: PlotPoints = frame
                    .freqs_hz
                    .iter()
                    .cloned()
                    .zip(frame.amps_last.iter().cloned())
                    .map(|(f, e)| [f.log2() as f64, e as f64])
                    .collect();
                plot_ui.line(Line::new("Envelope", pts));
            }
        })
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
                frame.landscape.c_last.len(),
                frame.landscape.k_last.len()
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

        // // === Envelope & PLV ===
        //        ui.horizontal(|ui| {
        // Envelope (NSGT)
        ui.vertical(|ui| {
            ui.heading("NSGT Envelope");
            ui.allocate_ui_with_layout(
                Vec2::new(750.0, 180.0),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    draw_nsgt_envelope(ui, &frame.landscape);
                },
            );
        });

        ui.separator();

        //     // PLV heatmap
        //     ui.vertical(|ui| {
        //         ui.heading("Phase Locking (PLV)");
        //         ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
        //             if let Some(plv) = &frame.landscape.plv_last {
        //                 let freqs = &frame.landscape.freqs_hz;
        //                 thread_local! {
        //                     static TEX_PLV: std::cell::RefCell<Option<TextureHandle>> =
        //                         std::cell::RefCell::new(None);
        //                 }
        //                 TEX_PLV.with(|tex| {
        //                     let mut tex_ref = tex.borrow_mut();
        //                     show_plv_heatmap(ui, freqs, freqs, plv, &mut *tex_ref);
        //                 });
        //             } else {
        //                 ui.label("PLV data not available yet.");
        //             }
        //         });
        //     });
        // });

        ui.separator();
        ui.heading("Landscape");

        let max_r = frame.landscape.r_last.iter().cloned().fold(0.0, f32::max);
        let min_k = frame.landscape.k_last.iter().cloned().fold(0.0, f32::min);
        let max_k = frame.landscape.k_last.iter().cloned().fold(0.0, f32::max);

        ui.columns(1, |cols| {
            let ui = &mut cols[0];

            // Roughness R (log2)
            log2_plot_hz(
                ui,
                "Roughness Landscape (R)",
                &frame.landscape.freqs_hz,
                &frame.landscape.r_last,
                "R",
                0.0,
                (max_r * 1.05) as f64,
            );

            // Consonance C (phase-based)
            log2_plot_hz(
                ui,
                "Consonance Landscape (C)",
                &frame.landscape.freqs_hz,
                &frame.landscape.c_last,
                "C",
                0.0,
                1.0,
            );

            // Combined potential K = αC − βR
            log2_plot_hz(
                ui,
                "Potential K = αC − βR",
                &frame.landscape.freqs_hz,
                &frame.landscape.k_last,
                "K",
                (min_k * 1.1) as f64,
                (max_k * 1.1) as f64,
            );
        });
    });
}
