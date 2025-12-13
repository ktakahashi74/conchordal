use egui::{Align2, Color32, FontId, Stroke};
use egui_plot::{Bar, BarChart, GridInput, GridMark, Line, Plot, PlotPoints, log_grid_spacer};

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

    // Use a sparser set to keep labels from colliding.
    const HZ_TICKS: [f64; 5] = [20.0, 100.0, 1_000.0, 10_000.0, 20_000.0];
    let tick_marks_log2: Vec<f64> = HZ_TICKS.iter().map(|hz| hz.log2()).collect();

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
                .fill(Color32::from_rgb(240, 120, 120)) // match Roughness accent
                .stroke(egui::Stroke::NONE),
        );
    }

    let chart = BarChart::new(y_label, bars);

    let y_max_fixed = if y_max <= y_min { y_min + 1.0 } else { y_max };

    let tick_marks_log2_for_grid = tick_marks_log2.clone();
    Plot::new(title)
        .height(150.0)
        .allow_scroll(false)
        .allow_drag(false)
        .include_y(y_min)
        .include_y(y_max_fixed)
        .include_x((20.0f64).log2())
        .include_x((20_000.0f64).log2())
        .default_x_bounds((20.0f64).log2(), (20_000.0f64).log2())
        .default_y_bounds(y_min, y_max_fixed)
        .x_grid_spacer(move |_input: GridInput| {
            tick_marks_log2_for_grid
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let step_size = if i + 1 < tick_marks_log2_for_grid.len() {
                        tick_marks_log2_for_grid[i + 1] - v
                    } else {
                        tick_marks_log2_for_grid[i] - tick_marks_log2_for_grid[i - 1]
                    };
                    GridMark {
                        value: v,
                        step_size,
                    }
                })
                .collect()
        })
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

/// log₂スケールで周波数を描画する汎用プロット関数
#[allow(clippy::too_many_arguments)]
pub fn log2_plot_hz(
    ui: &mut egui::Ui,
    title: &str,
    xs_hz: &[f32],
    ys: &[f32],
    y_label: &str,
    y_min: f64,
    y_max: f64,
    height: f32,
) {
    assert_eq!(
        xs_hz.len(),
        ys.len(),
        "x/y length mismatch: {} vs {}",
        xs_hz.len(),
        ys.len()
    );

    // === X軸を log2(Hz) に変換 ===
    let points: PlotPoints = xs_hz
        .iter()
        .zip(ys.iter())
        .map(|(&xx, &yy)| [xx.log2() as f64, yy as f64])
        .collect();

    // === egui_plot用 Line オブジェクト ===
    let line = Line::new(y_label, points);

    // === X軸範囲（20〜20kHz）を log2に変換 ===
    let x_min = (10.0f64).log2();
    //let x_min = 1.0;
    let x_max = (24_000.0f64).log2();

    // === 描画 ===
    Plot::new(title)
        .height(height)
        .allow_scroll(false)
        .allow_drag(false)
        .include_x(x_min)
        .include_x(x_max)
        .include_y(y_min)
        .include_y(y_max)
        .x_grid_spacer(log_grid_spacer(10))
        .x_axis_formatter(|mark, _range| {
            let hz = 2f64.powf(mark.value);
            format!("{:.0}", hz)
        })
        .y_axis_formatter(|mark, _range| format!("{:.2}", mark.value))
        .show(ui, |plot_ui| {
            plot_ui.line(line);

            // === 任意: 半音ガイドライン ===
            // for note in 21..=108 {
            //     let f = 440.0 * 2f32.powf((note as f32 - 69.0) / 12.0);
            //     if (20.0..=20_000.0).contains(&f) {
            //         let x = (f as f64).log2();
            //         plot_ui.vline(egui_plot::VLine::new(x).color(egui::Color32::DARK_GRAY));
            //     }
            // }
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
            .include_y(-1.1)
            .include_y(1.1)
            .default_y_bounds(-1.1, 1.1)
            .x_axis_formatter(|mark, _| format!("{:.3} s", mark.value))
            .y_axis_formatter(|mark, _| format!("{:.2}", mark.value))
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });
    });
}

/// Visualize neural rhythms (Delta/Theta/Alpha/Beta) as radial gauges.
pub fn neural_compass(ui: &mut egui::Ui, rhythms: &crate::core::modulation::NeuralRhythms) {
    let bands = [
        ("Delta", Color32::from_rgb(52, 152, 219), rhythms.delta),
        ("Theta", Color32::from_rgb(46, 204, 113), rhythms.theta),
        ("Alpha", Color32::from_rgb(241, 196, 15), rhythms.alpha),
        ("Beta", Color32::from_rgb(231, 76, 60), rhythms.beta),
    ];

    ui.horizontal(|ui| {
        for (label, color, rhythm) in bands {
            let desired = egui::vec2(90.0, 100.0);
            let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
            let painter = ui.painter_at(rect);
            let center = rect.center();
            let radius = rect.width().min(rect.height()) * 0.35;

            // Background circle
            painter.circle_stroke(center, radius, Stroke::new(1.0, color.gamma_multiply(0.35)));
            painter.circle_filled(center, 4.0, Color32::WHITE);
            painter.circle_stroke(center, 4.0, Stroke::new(1.0, color.gamma_multiply(0.6)));
            let top = center + egui::vec2(0.0, -radius);
            painter.line_segment(
                [center, top],
                Stroke::new(1.0, Color32::WHITE.gamma_multiply(0.3)),
            );

            // Needle
            let vis_mag = (rhythm.mag * 6.0).min(1.0); // boost low values for visibility
            let length = radius * (0.1 + 0.9 * vis_mag);
            let angle = rhythm.phase - std::f32::consts::FRAC_PI_2;
            let tip = center + egui::vec2(angle.cos(), angle.sin()) * length;
            let thickness = 1.5 + 3.0 * vis_mag;
            painter.line_segment([center, tip], Stroke::new(thickness, color));

            // Label
            painter.text(
                rect.center_top() + egui::vec2(0.0, 8.0),
                Align2::CENTER_TOP,
                label,
                FontId::proportional(12.0),
                color,
            );
        }
    });
}
