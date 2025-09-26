use egui::{CentralPanel, SidePanel, TopBottomPanel};
use crate::ui::plots::{log2_plot_hz, time_plot};
use crate::ui::viewdata::UiFrame;

pub fn main_window(ctx: &egui::Context, frame: &UiFrame) {
    TopBottomPanel::top("top").show(ctx, |ui| {
        ui.heading("Concord — Skeleton");
        ui.label("Wave + Landscape (R_v1 roughness only; C=0; K=-R)");
    });

    // SidePanel::left("left").resizable(true).show(ctx, |ui| {
    //     ui.heading("Spectrum (amps)");
    // });

    
    CentralPanel::default().show(ctx, |ui| {
        ui.heading("Waveform");
        time_plot(ui, "Current Hop Wave", frame.wave.fs, &frame.wave.samples);

	ui.heading("Spectrum");
	ui.separator();
	log2_plot_hz(ui, "Amplitude Spectrum",
		     &frame.spec_hz, &frame.amps,
		     "A[k]", 0.0, (frame.amps.iter().cloned().fold(0.0, f32::max) * 1.1 + 1e-6) as f64);
	    


	
        ui.separator();
        ui.heading("Landscape");
        let max_r = frame.landscape.r.iter().cloned().fold(0.0, f32::max);
        let min_k = frame.landscape.k.iter().cloned().fold(0.0, f32::min);
        let max_k = frame.landscape.k.iter().cloned().fold(0.0, f32::max);
        ui.columns(1, |cols| {
            let ui = &mut cols[0];
            log2_plot_hz(ui, "Roughness R", &frame.landscape.freqs_hz, &frame.landscape.r, "R", 0.0, (max_r*1.1+1e-6) as f64);
            log2_plot_hz(ui, "Consonance C (dummy 0)", &frame.landscape.freqs_hz, &frame.landscape.c, "C", 0.0, 1.0);
            log2_plot_hz(ui, "K = alpha*C - beta*R", &frame.landscape.freqs_hz, &frame.landscape.k, "K", min_k as f64 * 1.1, max_k as f64 * 1.1);
        });
    });
}
