// Entry point: launches the egui/eframe app and spawns worker threads.
mod app;
mod audio;
mod config;
mod core;
mod life;
mod synth;
mod ui;

use clap::Parser;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Parse a comma-separated list like "440:0.8,880:0.5"
fn parse_tones(s: &str) -> Vec<(f32, f32)> {
    s.split(',')
        .filter_map(|pair| {
            let mut parts = pair.split(':');
            let f = parts.next()?.trim().parse::<f32>().ok()?;
            let a = parts
                .next()
                .map(|x| x.trim().parse::<f32>().unwrap_or(100.))
                .unwrap_or(100.);
            Some((f, a))
        })
        .collect()
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Play audio in realtime
    #[arg(long, default_value_t = true)]
    play: bool,

    /// Write audio to wav file
    #[arg(long)]
    wav: Option<String>,

    /// Input tones in the form f1:amp,f2:amp,...
    #[arg(long)]
    tones: Option<String>,
}

fn main() -> eframe::Result<()> {
    let args = Args::parse();

    // Parse tones argument (default 440 Hz @ amp=1.0)
    let tones_parsed = args
        .tones
        .as_deref()
        .map(parse_tones)
        .unwrap_or_else(|| vec![(440.0, 500.0)]);

    println!("Using tones: {:?}", tones_parsed);

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_ctrlc = stop_flag.clone();

    ctrlc::set_handler(move || {
        stop_flag_for_ctrlc.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 1200.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Conchordal",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(app::App::new(
                cc,
                args,
                stop_flag.clone(),
                tones_parsed,
            )))
        }),
    )
}
