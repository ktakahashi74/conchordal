use egui::{CentralPanel, SidePanel, TopBottomPanel};
use crate::core::gammatone::gammatone_filterbank;
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
        time_plot(ui, "Current Hop Wave", frame.wave.fs as f64, &frame.wave.samples);

	// ui.heading("Spectrum");
	// ui.separator();
	// log2_plot_hz(ui, "Amplitude Spectrum",
	// 	     &frame.spec.spec_hz, &frame.spec.amps,
	// 	     "A[k]", 0.0, (frame.spec.amps.iter().cloned().fold(0.0, f32::max) * 1.1 + 1e-6) as f64);

	use crate::core::hilbert::hilbert_envelope; 
	let outs = gammatone_filterbank(&frame.wave.samples, &frame.landscape.freqs_hz, frame.wave.fs);
	let amps: Vec<f32> = outs
        .iter()
        .map(|ch| {
            let env = hilbert_envelope(ch);
            let rms = (env.iter().map(|&v| v * v).sum::<f32>() / env.len() as f32).sqrt();
            rms
        })
        .collect();

	log2_plot_hz(ui, "Gamma Tone Filterbank Frequencies",
		     &frame.landscape.freqs_hz, 
		     amps.as_slice(),
		     "Freqs", 0.0, 2.0);



	ui.separator();
        ui.heading("Landscape");
        let max_r = frame.landscape.r.iter().cloned().fold(0.0, f32::max);
        let min_k = frame.landscape.k.iter().cloned().fold(0.0, f32::min);
        let max_k = frame.landscape.k.iter().cloned().fold(0.0, f32::max);

	
        ui.columns(1, |cols| {
            let ui = &mut cols[0];

            // Roughness R
            log2_plot_hz(
                ui,
                "Roughness Landscape (R)",
                &frame.landscape.freqs_hz,
                &frame.landscape.r,
                "R",
                0.0,
		(max_r * 1.05) as f64,
            );

	    println!("max_r = {:?}", max_r);
	    println!("frame.landscape.r = {:?}", frame.landscape.r);

            // Consonance (まだダミー)
            log2_plot_hz(
                ui,
                "Consonance C (dummy 0)",
                &frame.landscape.freqs_hz,
                &frame.landscape.c,
                "C",
                0.0,
                1.0,
            );

            // K = alpha*C - beta*R
            log2_plot_hz(
                ui,
                "K = alpha*C - beta*R",
                &frame.landscape.freqs_hz,
                &frame.landscape.k,
                "K",
                (min_k * 1.1) as f64,
                (max_k * 1.1) as f64,
            );
        });


    });
}
