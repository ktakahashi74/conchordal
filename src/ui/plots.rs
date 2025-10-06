use crate::core::landscape::LandscapeFrame;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints};

/// log2 軸で周波数を描画するヒストグラム（自動幅調整版）
pub fn log2_hist_hz(
    ui: &mut egui::Ui,
    title: &str,
    xs_hz: &[f32],
    ys: &[f32],
    y_label: &str,
    y_min: f64,
    y_max: f64,
) {
    assert_eq!(xs_hz.len(), ys.len());
    if xs_hz.is_empty() {
        return;
    }

    // 各ビンごとに棒の幅を決める
    let mut bars: Vec<Bar> = Vec::with_capacity(xs_hz.len());
    for i in 0..xs_hz.len() {
        let f = xs_hz[i].max(1.0); // 0Hz対策
        let f_left = if i > 0 { xs_hz[i - 1].max(1.0) } else { f };
        let f_right = if i + 1 < xs_hz.len() {
            xs_hz[i + 1].max(1.0)
        } else {
            f
        };

        // log2 軸上の幅を近傍から推定
        let left = (f_left.log2() + f.log2()) * 0.5;
        let right = (f_right.log2() + f.log2()) * 0.5;
        let width = (right - left).abs().max(0.001);

        bars.push(
            Bar::new(f.log2() as f64, ys[i] as f64)
                .width(width as f64)
                .fill(egui::Color32::DARK_RED)
                .stroke(egui::Stroke::NONE),
        );
    }

    let chart = BarChart::new(y_label, bars);

    let min_x = xs_hz.iter().cloned().fold(f32::MAX, f32::min).max(1.0);
    let max_x = xs_hz.iter().cloned().fold(f32::MIN, f32::max);

    Plot::new(title)
        .height(150.0)
        .allow_scroll(false)
        .allow_drag(false)
        .include_y(y_min)
        .include_y(y_max)
        .include_x((min_x as f64).log2())
        .include_x((max_x as f64).log2())
        .x_axis_formatter(|mark, _range| {
            let hz = 2f64.powf(mark.value);
            if hz < 1000.0 {
                format!("{:.0} Hz", hz)
            } else {
                format!("{:.1} kHz", hz / 1000.0)
            }
        })
        .y_axis_formatter(|mark, _| format!("{:.2}", mark.value))
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(chart);
        });
}

/// log2 軸で周波数を描画するスペクトル系プロット
pub fn log2_plot_hz(
    ui: &mut egui::Ui,
    title: &str,
    xs_hz: &[f32],
    ys: &[f32],
    y_label: &str,
    y_min: f64,
    y_max: f64,
) {
    assert_eq!(xs_hz.len(), ys.len());

    let points: PlotPoints = xs_hz
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| [x.log2() as f64, *y as f64])
        .collect();
    let line = Line::new(y_label, points);

    Plot::new(title)
        .height(150.0)
        //        .data_aspect(1.0)
        .allow_scroll(false)
        .allow_drag(false)
        .include_y(y_min)
        .include_y(y_max)
        .include_x((20.0f64).log2())
        .include_x((20_000.0f64).log2())
        .x_axis_formatter(|mark, _range| {
            let hz = 2f64.powf(mark.value);
            format!("{:.0} Hz", hz)
        })
        .y_axis_formatter(|mark, _range| format!("{:.2}", mark.value))
        .show(ui, |plot_ui| {
            // // 半音ごとのガイドライン
            // for note in 20..=120 {
            //     let f = 440.0 * 2f32.powf((note as f32 - 69.0) / 12.0); // MIDI→Hz
            //     if (20.0..=20_000.0).contains(&f) {
            //         let x = (f as f64).log2();
            //         plot_ui.vline(egui_plot::VLine::new("", x).color(egui::Color32::DARK_GRAY));
            //     }
            // }
            plot_ui.line(line);
        });
}

/// 波形表示（時間軸）
pub fn time_plot(ui: &mut egui::Ui, title: &str, fs: f64, samples: &[f32]) {
    let points: PlotPoints = samples
        .iter()
        .enumerate()
        .map(|(i, s)| [i as f64 / fs, *s as f64])
        .collect();
    let line = Line::new("wave", points);

    ui.vertical(|ui| {
        //ui.label(title);

        Plot::new(title)
            .height(150.0)
            .allow_scroll(false)
            .allow_drag(false)
            .x_axis_formatter(|mark, _| format!("{:.3} s", mark.value))
            .y_axis_formatter(|mark, _| format!("{:.2}", mark.value))
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });
    });
}
