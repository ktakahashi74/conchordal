use crate::ui::plots::{log2_plot_hz, time_plot};
use crate::ui::viewdata::UiFrame;
use egui::{CentralPanel, TopBottomPanel, Vec2};
use egui_plot::{Line, Plot, PlotPoints};

use crate::core::landscape::LandscapeFrame;

use egui::{Color32, Response, ScrollArea, Ui};

use egui::epaint::{ColorImage, TextureHandle};
use egui_plot::{PlotImage, PlotPoint, PlotResponse};

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

    // --- build color pixels vector (flip vertically for display) ---
    let mut pixels = vec![Color32::BLACK; nx * ny];
    for j in 0..ny {
        let row = ny - 1 - j; // flip vertically (so low freq bottom)
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

    // --- create ColorImage ---
    let img = ColorImage::new([nx, ny], pixels);

    // --- upload/update texture ---
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
        // PlotImage center and size (not bottom-left)
        let center_x = (fx_min + fx_max) * 0.5;
        let center_y = (fy_min + fy_max) * 0.5;
        let size_x = fx_max - fx_min;
        let size_y = fy_max - fy_min;

        let img = PlotImage::new(
            "plv_img",
            texture.id(),
            PlotPoint::new(center_x, center_y), // center point
            Vec2::new(size_x, size_y),
        );

        plot_ui.image(img);
    });
}

/// Plot cochlea state: line (R per channel) + heatmap (history).
pub fn draw_cochlea_state(ui: &mut egui::Ui, frame: &LandscapeFrame) -> PlotResponse<()> {
    Plot::new("cochlea_state_plot")
        .legend(egui_plot::Legend::default())
        .allow_scroll(false)
        .allow_drag(false)
        .height(150.0)
        // 横軸を log2(Hz) に変換した値で表示
        .x_axis_formatter(|mark, _range| {
            let hz = 2f64.powf(mark.value);
            format!("{:.0} Hz", hz)
        })
        .include_x((20.0f64).log2())
        .include_x((20_000.0f64).log2())
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            // --- 折れ線プロット（最新 R） ---
            if !frame.freqs_hz.is_empty() && frame.r_last.len() == frame.freqs_hz.len() {
                let pts: PlotPoints = frame
                    .freqs_hz
                    .iter()
                    .cloned()
                    .zip(frame.r_last.iter().cloned())
                    .map(|(f, r)| [f.log2() as f64, r as f64])
                    .collect();
                plot_ui.line(Line::new("R (latest)", pts));
            }
        })
}

pub fn main_window(ctx: &egui::Context, frame: &UiFrame) {
    TopBottomPanel::top("top").show(ctx, |ui| {
        ui.heading("Concord — Skeleton");
        ui.label("Wave + Landscape (Cochlea roughness R; C dummy; K=-R)");
    });

    CentralPanel::default().show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.heading("Waveform");
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(500.0, 180.0),
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
            ui.vertical(|ui| {
                ui.heading("Synth Spectrum");

                if frame.spec.spec_hz.len() > 1 && frame.spec.amps.len() > 1 {
                    let max_amp = frame.spec.amps.iter().cloned().fold(0.0, f32::max);

                    ui.vertical(|ui| {
                        crate::ui::plots::log2_hist_hz(
                            ui,
                            "Amplitude Spectrum",
                            &frame.spec.spec_hz[1..], // remove DC
                            &frame.spec.amps[1..],
                            "A[k]",
                            0.0,
                            (max_amp * 1.05) as f64,
                        );
                    });
                }
            });
        });

        ui.separator();

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.heading("Cochlea State");
                ui.allocate_ui_with_layout(
                    egui::Vec2::new(750.0, 180.0),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        draw_cochlea_state(ui, &frame.landscape);
                    },
                );
            });

            ui.separator();

            ui.vertical(|ui| {
                ui.heading("PLV Heatmap");

                ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
                    if let Some(plv) = &frame.landscape.plv_last {
                        let freqs = &frame.landscape.freqs_hz;

                        thread_local! {
                            static TEX_PLV: std::cell::RefCell<Option<TextureHandle>> =
                            std::cell::RefCell::new(None);
                        }

                        TEX_PLV.with(|tex| {
                            let mut tex_ref = tex.borrow_mut();
                            show_plv_heatmap(ui, freqs, freqs, plv, &mut *tex_ref);
                        });
                    } else {
                        ui.label("PLV data not available. Run landscape update first.");
                    }
                });
            });
        });

        ui.separator();
        ui.heading("Landscape");

        let max_r = frame.landscape.r_last.iter().cloned().fold(0.0, f32::max);
        let min_k = frame.landscape.k_last.iter().cloned().fold(0.0, f32::min);
        let max_k = frame.landscape.k_last.iter().cloned().fold(0.0, f32::max);

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
                //1.0,
            );

            // Consonance (dummy)
            log2_plot_hz(
                ui,
                "Consonance C (dummy 0)",
                &frame.landscape.freqs_hz,
                &frame.landscape.c_last,
                "C",
                0.0,
                1.0,
            );

            // K = alpha*C - beta*R
            log2_plot_hz(
                ui,
                "K = alpha*C - beta*R",
                &frame.landscape.freqs_hz,
                &frame.landscape.k_last,
                "K",
                (min_k * 1.1) as f64,
                (max_k * 1.1) as f64,
            );
        });
    });
}
