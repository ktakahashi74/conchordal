use crate::ui::plots::{log2_plot_hz, time_plot};
use crate::ui::viewdata::UiFrame;
use egui::{CentralPanel, TopBottomPanel};
use egui_plot::{Line, Plot, PlotPoints};

use crate::core::landscape::LandscapeFrame;

use egui::{Color32, ColorImage};

/// Plot cochlea state: line (R per channel) + heatmap (history).
pub fn draw_cochlea_state(ui: &mut egui::Ui, frame: &LandscapeFrame) {
    Plot::new("cochlea_state_plot")
        .legend(egui_plot::Legend::default())
        .allow_scroll(false)
        .allow_drag(false)
        .height(250.0)
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

            // --- ヒートマップ（履歴） ---
            // if !frame.r_hist.is_empty() {
            //     let rows = frame.r_hist.len();
            //     let cols = frame.r_hist[0].len();
            //     if rows > 0 && cols > 0 {
            //         let mut buf = vec![0.0f64; rows * cols];
            //         for (row, hist_row) in frame.r_hist.iter().enumerate() {
            //             for (col, &v) in hist_row.iter().enumerate() {
            //                 buf[row * cols + col] = v as f64;
            //             }
            //         }
            //         let hm = Heatmap::new(buf, cols, rows)
            //             .name("R history")
            //             .color_map(egui_plot::ColorMap::Viridis);
            //         plot_ui.heatmap(hm);
            //     }
            //}
        });

    ui.label("x: log2(freq) / time (heat), y: channel index, color: R amplitude");
}

pub fn main_window(ctx: &egui::Context, frame: &UiFrame) {
    TopBottomPanel::top("top").show(ctx, |ui| {
        ui.heading("Concord — Skeleton");
        ui.label("Wave + Landscape (Cochlea roughness R; C dummy; K=-R)");
    });

    CentralPanel::default().show(ctx, |ui| {
        ui.heading("Waveform");
        time_plot(
            ui,
            "Current Hop Wave",
            frame.wave.fs as f64,
            &frame.wave.samples,
        );

        ui.separator();
        ui.heading("Synth Spectrum");

        if frame.spec.spec_hz.len() > 1 && frame.spec.amps.len() > 1 {
            let max_amp = frame.spec.amps.iter().cloned().fold(0.0, f32::max);
            crate::ui::plots::log2_hist_hz(
                ui,
                "Amplitude Spectrum",
                &frame.spec.spec_hz[1..], // remove DC
                &frame.spec.amps[1..],
                "A[k]",
                0.0,
                (max_amp * 1.05) as f64,
            );
        }

        ui.separator();
        ui.heading("Cochlea State");
        crate::ui::windows::draw_cochlea_state(ui, &frame.landscape);

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
                //                (max_r * 1.05) as f64,
                1.0,
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
